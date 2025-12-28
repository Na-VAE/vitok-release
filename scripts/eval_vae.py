#!/usr/bin/env python
"""Evaluate ViTok VAE reconstruction quality.

Computes rFID, rFDD, SSIM, and PSNR on various datasets.

Usage:
    # Evaluate on ImageNet validation
    python scripts/eval_vae.py \
        --checkpoint checkpoints/vae/final.pt \
        --data /path/to/imagenet/val

    # Evaluate on multiple datasets
    python scripts/eval_vae.py \
        --checkpoint checkpoints/vae/final.pt \
        --imagenet_root /path/to/imagenet/val \
        --div8k_root /path/to/DIV8K \
        --eval_configs imagenet_512 div8k_1024

    # Multi-GPU evaluation
    torchrun --nproc_per_node=4 scripts/eval_vae.py \
        --checkpoint checkpoints/vae/final.pt \
        --data /path/to/imagenet/val
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from PIL import Image

from vitok import AEConfig, load_ae
from vitok.naflex_io import postprocess_images
from vitok.data import create_dataloader
from vitok.evaluators import MetricCalculator


def setup_distributed():
    """Setup distributed if available."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed."""
    if dist.is_initialized():
        dist.destroy_process_group()


# Evaluation configurations
EVAL_CONFIGS = {
    'imagenet_256': {
        'max_size': 256,
        'val_size': 50000,
        'batch_size': 64,
        'metrics': ('fid', 'fdd', 'ssim', 'psnr'),
        'augmentation': 'square',
    },
    'imagenet_512': {
        'max_size': 512,
        'val_size': 50000,
        'batch_size': 32,
        'metrics': ('fid', 'fdd', 'ssim', 'psnr'),
        'augmentation': 'naflex',
    },
    'imagenet_square_256': {
        'max_size': 256,
        'val_size': 50000,
        'batch_size': 64,
        'metrics': ('fid', 'fdd', 'ssim', 'psnr'),
        'augmentation': 'square',
    },
    'imagenet_square_512': {
        'max_size': 512,
        'val_size': 50000,
        'batch_size': 32,
        'metrics': ('fid', 'fdd', 'ssim', 'psnr'),
        'augmentation': 'square',
    },
    'div8k_1024': {
        'max_size': 1024,
        'val_size': 1500,
        'batch_size': 8,
        'metrics': ('fid', 'fdd', 'ssim', 'psnr'),
        'augmentation': 'naflex',
    },
    'div8k_2048': {
        'max_size': 2048,
        'val_size': 1500,
        'batch_size': 2,
        'metrics': ('fid', 'fdd', 'ssim', 'psnr'),
        'augmentation': 'naflex',
    },
    'textvqa_1280': {
        'max_size': 1280,
        'val_size': 5000,
        'batch_size': 8,
        'metrics': ('ssim', 'psnr'),
        'augmentation': 'naflex',
    },
    'urban100_1024': {
        'max_size': 1024,
        'val_size': 100,
        'batch_size': 4,
        'metrics': ('ssim', 'psnr'),
        'augmentation': 'naflex',
    },
    'clic_2048': {
        'max_size': 2048,
        'val_size': 100,
        'batch_size': 2,
        'metrics': ('ssim', 'psnr'),
        'augmentation': 'naflex',
    },
}


def get_pp_string(cfg: dict, patch_size: int) -> str:
    """Build preprocessing string from config."""
    max_size = cfg['max_size']
    max_tokens = (max_size // patch_size) ** 2

    if cfg['augmentation'] == 'square':
        return (
            f"resize({max_size})|"
            f"center_crop({max_size})|"
            f"to_tensor|"
            f"normalize(minus_one_to_one)|"
            f"patchify({max_size}, {patch_size}, {max_tokens})"
        )
    else:  # naflex
        return (
            f"to_tensor|"
            f"normalize(minus_one_to_one)|"
            f"patchify({max_size}, {patch_size}, {max_tokens})"
        )


def evaluate_single(
    model,
    data_source: str,
    cfg: dict,
    patch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
) -> Dict:
    """Run evaluation on a single dataset configuration."""
    max_size = cfg['max_size']
    max_grid_size = max_size // patch_size

    # Build preprocessing
    pp_string = get_pp_string(cfg, patch_size)

    # Per-GPU batch size
    per_gpu_batch = max(1, cfg['batch_size'] // world_size)
    global_batch = per_gpu_batch * world_size
    total_steps = max(1, cfg['val_size'] // global_batch)

    # Create dataloader
    loader = create_dataloader(
        source=data_source,
        pp=pp_string,
        batch_size=per_gpu_batch,
        num_workers=4,
        seed=42,
    )

    # Initialize metrics
    metrics = MetricCalculator(metrics=cfg['metrics'])
    metrics.move_model_to_device(device)

    # Collect samples
    all_real = []
    all_fake = []

    if rank == 0:
        print(f"  Evaluating {total_steps} batches...")

    data_iter = iter(loader)
    for step in range(total_steps):
        try:
            batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch, _ = next(data_iter)

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        if 'patches' in batch:
            batch['patches'] = batch['patches'].to(dtype)

        # Forward
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=dtype):
                output = model(batch, sample_posterior=False)

            # Postprocess to images
            recon_images = postprocess_images(
                output,
                output_format="minus_one_to_one",
                current_format="minus_one_to_one",
                unpack=True,
                patch=patch_size,
                max_grid_size=max_grid_size,
            )
            ref_images = postprocess_images(
                batch,
                output_format="minus_one_to_one",
                current_format="minus_one_to_one",
                unpack=True,
                patch=patch_size,
                max_grid_size=max_grid_size,
            )

        # Store
        if isinstance(ref_images, list):
            all_real.extend([x.cpu() for x in ref_images])
        else:
            all_real.append(ref_images.cpu())

        if isinstance(recon_images, list):
            all_fake.extend([x.cpu() for x in recon_images])
        else:
            all_fake.append(recon_images.cpu())

    # Compute metrics
    metrics.update(all_real, all_fake)
    stats = metrics.gather()
    metrics.reset()
    metrics.move_model_to_device('cpu')

    return stats


def save_results(output_dir: Path, name: str, stats: Dict, rank: int):
    """Save evaluation results."""
    if rank != 0:
        return

    eval_dir = output_dir / name
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    metrics = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
    if metrics:
        df = pd.DataFrame([metrics])
        df.to_csv(eval_dir / "metrics.csv", index=False)

    # Print
    print(f"\n{'='*50}")
    print(f"Results: {name}")
    print(f"{'='*50}")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViTok VAE")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to VAE checkpoint")
    parser.add_argument("--variant", type=str, default="Ld2-Ld22/1x16x64")
    parser.add_argument("--variational", action="store_true")

    # Data sources
    parser.add_argument("--data", type=str, default=None,
                        help="Path to default dataset (used if specific root not provided)")
    parser.add_argument("--imagenet_root", type=str, default=None)
    parser.add_argument("--div8k_root", type=str, default=None)
    parser.add_argument("--textvqa_root", type=str, default=None)
    parser.add_argument("--urban100_root", type=str, default=None)
    parser.add_argument("--clic_root", type=str, default=None)

    # Evaluation configs
    parser.add_argument("--eval_configs", type=str, nargs='+',
                        default=['imagenet_512'],
                        help="Evaluation configs to run")

    # Output
    parser.add_argument("--output_dir", type=str, default="eval_results")

    # System
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--compile", action="store_true")

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if rank == 0:
        print(f"Evaluating checkpoint: {args.checkpoint}")
        print(f"Configs: {args.eval_configs}")

    # Load model
    config = AEConfig(
        variant=args.variant,
        variational=args.variational,
    )
    model = load_ae(args.checkpoint, config, device=device, dtype=dtype)
    model.eval()

    if args.compile and hasattr(torch, 'compile'):
        if rank == 0:
            print("Compiling model...")
        model = torch.compile(model)

    # Get spatial stride
    spatial_stride = model.spatial_stride if hasattr(model, 'spatial_stride') else args.patch_size

    if rank == 0:
        print(f"Model loaded. Spatial stride: {spatial_stride}")

    # Output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Map configs to data sources
    def get_data_source(config_name: str) -> str:
        if 'imagenet' in config_name:
            return args.imagenet_root or args.data
        elif 'div8k' in config_name:
            return args.div8k_root or args.data
        elif 'textvqa' in config_name:
            return args.textvqa_root or args.data
        elif 'urban100' in config_name:
            return args.urban100_root or args.data
        elif 'clic' in config_name:
            return args.clic_root or args.data
        return args.data

    # Run evaluations
    all_results = {}

    for config_name in args.eval_configs:
        if config_name not in EVAL_CONFIGS:
            if rank == 0:
                print(f"Unknown config: {config_name}. Skipping.")
            continue

        cfg = EVAL_CONFIGS[config_name]
        data_source = get_data_source(config_name)

        if not data_source:
            if rank == 0:
                print(f"No data source for {config_name}. Skipping.")
            continue

        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Evaluating: {config_name}")
            print(f"  Data: {data_source}")
            print(f"  Max size: {cfg['max_size']}")
            print(f"  Val size: {cfg['val_size']}")
            print(f"  Metrics: {cfg['metrics']}")
            print(f"{'='*50}")

        t_start = time.perf_counter()

        try:
            stats = evaluate_single(
                model=model,
                data_source=data_source,
                cfg=cfg,
                patch_size=spatial_stride,
                device=device,
                dtype=dtype,
                rank=rank,
                world_size=world_size,
            )
            stats['eval_time_s'] = time.perf_counter() - t_start
            all_results[config_name] = stats
            save_results(output_dir, config_name, stats, rank)

        except Exception as e:
            if rank == 0:
                print(f"Error evaluating {config_name}: {e}")
            continue

    # Print summary
    if rank == 0 and all_results:
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"{'Config':<25} {'FID':>10} {'FDD':>10} {'SSIM':>10} {'PSNR':>10}")
        print(f"{'-'*70}")
        for name, stats in all_results.items():
            fid = stats.get('fid', float('nan'))
            fdd = stats.get('fdd', float('nan'))
            ssim = stats.get('ssim', float('nan'))
            psnr = stats.get('psnr', float('nan'))
            print(f"{name:<25} {fid:>10.2f} {fdd:>10.2f} {ssim:>10.4f} {psnr:>10.2f}")
        print(f"{'='*70}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
