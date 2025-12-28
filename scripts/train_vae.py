#!/usr/bin/env python
"""Train ViTok VAE (Autoencoder).

Supports FSDP2/DDP distributed training with perceptual losses.

Usage:
    # Single GPU training
    python scripts/train_vae.py \
        --data /path/to/shards/*.tar \
        --output_dir checkpoints/vae

    # Multi-GPU with FSDP2
    torchrun --nproc_per_node=8 scripts/train_vae.py \
        --data hf://ILSVRC/imagenet-1k/train/*.tar \
        --fsdp \
        --output_dir checkpoints/vae

    # Multi-GPU with DDP
    torchrun --nproc_per_node=4 scripts/train_vae.py \
        --data hf://ILSVRC/imagenet-1k/train/*.tar \
        --output_dir checkpoints/vae
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from tqdm import tqdm

from vitok import AEConfig, create_ae
from vitok.data import create_dataloader
from vitok.naflex_io import postprocess_images, RandomTileSampler
from vitok import training_utils as tu
from vitok.evaluators import MetricCalculator

# Perceptual losses
from dino_perceptual import DINOPerceptual

# SSIM
from torchmetrics.image import structural_similarity_index_measure as SSIM

import wandb


def main():
    parser = argparse.ArgumentParser(description="Train ViTok VAE")

    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Data source (local path or hf://repo/pattern)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-GPU batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--square_crop_prob", type=float, default=0.5,
                        help="Probability of using square crop vs native aspect ratio")

    # Model
    parser.add_argument("--variant", type=str, default="Ld2-Ld22/1x16x64",
                        help="AE variant (e.g., B/1x16x64, Ld2-Ld22/1x16x64)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")

    # Training
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "muon"], help="Optimizer to use")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["cosine", "linear", "warmup_exp_decay"],
                        help="LR schedule type")

    # Loss weights
    parser.add_argument("--charbonnier", type=float, default=1.0)
    parser.add_argument("--charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--ssim", type=float, default=0.1)
    parser.add_argument("--dino_perceptual", type=float, default=250.0)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--n_tiles", type=int, default=2)

    # Distributed
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP2 instead of DDP")

    # Logging
    parser.add_argument("--output_dir", type=str, default="checkpoints/vae")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=5000,
                        help="Frequency for running evaluation")
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--marked_freq", type=int, default=25000,
                        help="Frequency for marked checkpoints (0 to disable)")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")

    args = parser.parse_args()

    # Handle compile flag (--compile is default True, --no_compile disables it)
    use_compile = args.compile and not args.no_compile

    # Setup distributed
    rank, world_size, local_rank, device, device_mesh = tu.setup_distributed(args.seed)
    dtype = torch.bfloat16
    use_fsdp = args.fsdp and world_size > 1

    # Output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb_enabled = args.wandb_project and rank == 0
    if wandb_enabled:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # Create model
    if rank == 0:
        print(f"Creating AE model: {args.variant}")
    config = AEConfig(variant=args.variant)
    model = create_ae(config)
    model.to(device=device, dtype=dtype)

    # Compile if requested (before FSDP/DDP wrapping)
    if use_compile:
        if rank == 0:
            print("Compiling model...")
        model = torch.compile(model, fullgraph=True)

    # Wrap with FSDP2 or DDP
    if world_size > 1:
        if use_fsdp:
            mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
            fully_shard(model, mesh=device_mesh, mp_policy=mp)
            if rank == 0:
                print("Using FSDP2")
        else:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)
            if rank == 0:
                print("Using DDP")

    # Count parameters
    model_ref = model.module if hasattr(model, 'module') else model
    n_params = sum(p.numel() for p in model_ref.parameters() if p.requires_grad)
    if rank == 0:
        print(f"Model parameters: {n_params / 1e6:.1f}M")

    # Optimizer
    if args.optimizer == "muon":
        from muon import Muon
        optimizer = Muon(model.parameters(), lr=args.lr, momentum=0.95)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=True)

    # LR Scheduler
    warmup_steps = int(args.warmup_ratio * args.steps)
    scheduler = tu.create_scheduler(
        optimizer=optimizer,
        schedule_type=args.schedule,
        steps=args.steps,
        lr=args.lr,
        warmup_steps=warmup_steps,
    )

    # Build train_state for DCP checkpointing
    train_state = {
        "app": tu.ModelOptimizerState(model, optimizer),
        "step": 1,
        "scheduler": scheduler,
    }

    # Resume if checkpoint provided
    start_step = 0
    if args.checkpoint:
        start_step = tu.load_checkpoint(train_state, args.checkpoint, rank)
        if rank == 0:
            print(f"Resumed at step {start_step}")

    # Create dataloader (on ALL ranks)
    if rank == 0:
        print(f"Loading data from: {args.data}")

    # Build preprocessing string with square crop probability
    # We use a mixed approach: sometimes square crop, sometimes native aspect ratio
    if args.square_crop_prob > 0:
        pp_string = (
            f"random_resized_crop({args.max_size})|"
            f"flip|"
            f"to_tensor|"
            f"normalize(minus_one_to_one)|"
            f"patchify({args.max_size}, {args.patch_size}, {args.max_tokens})"
        )
    else:
        # Native aspect ratio (naflex-style)
        pp_string = (
            f"to_tensor|"
            f"normalize(minus_one_to_one)|"
            f"patchify({args.max_size}, {args.patch_size}, {args.max_tokens})"
        )

    loader = create_dataloader(
        source=args.data, pp=pp_string, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed + rank + start_step,
    )

    # Perceptual losses
    tile_sampler = RandomTileSampler(
        n_tiles=args.n_tiles,
        tile_size=(args.tile_size, args.tile_size),
        spatial_stride=args.patch_size,
    )

    dino_loss_fn = None
    if args.dino_perceptual > 0:
        dino_loss_fn = DINOPerceptual(model_size='B', target_size=args.tile_size)
        dino_loss_fn = dino_loss_fn.to(device).eval()
        if use_compile:
            dino_loss_fn = torch.compile(dino_loss_fn, fullgraph=True)

    # Evaluation metric calculator
    eval_metrics = MetricCalculator(metrics=('ssim', 'psnr'))
    eval_metrics.move_model_to_device(device)

    # Training loop
    if rank == 0:
        print(f"\nStarting training for {args.steps} steps...")
    model.train()
    loader_iter = iter(loader)
    step = start_step

    log_metrics = {}
    log_count = 0
    t_log_start = time.perf_counter()

    max_grid_size = args.max_size // args.patch_size
    grad_params = [p for p in model.parameters() if p.requires_grad]

    pbar = tqdm(total=args.steps, initial=start_step, disable=rank != 0, desc="Training")

    while step < args.steps:
        try:
            batch, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch, _ = next(loader_iter)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if 'patches' in batch:
            batch['patches'] = batch['patches'].to(dtype)

        optimizer.zero_grad(set_to_none=True)

        # Step the scheduler (handles warmup + cosine internally)
        lr = scheduler.step()

        # Forward
        with torch.autocast(device_type='cuda', dtype=dtype):
            decode_dict = model(batch)

        ptype = batch['ptype']
        diff = decode_dict['patches'] - batch['patches']

        # Charbonnier loss
        if args.charbonnier > 0:
            charb_per_token = (diff.pow(2) + args.charbonnier_eps**2).sqrt().sum(dim=2)
            charb_per_token = charb_per_token * ptype
            actual_tokens = ptype.sum(dim=1).clamp_min(1)
            charb_loss = (charb_per_token.sum(dim=1) / actual_tokens).mean()
        else:
            charb_loss = torch.tensor(0.0, device=device)

        loss = args.charbonnier * charb_loss

        # Perceptual losses on tiles
        ssim_loss = torch.tensor(0.0, device=device)
        dino_loss = torch.tensor(0.0, device=device)

        if args.ssim > 0 or args.dino_perceptual > 0:
            with torch.no_grad():
                recon_images = postprocess_images(
                    decode_dict, output_format="minus_one_to_one",
                    current_format="minus_one_to_one", unpack=False,
                    patch=args.patch_size, max_grid_size=max_grid_size,
                )
                ref_images = postprocess_images(
                    batch, output_format="minus_one_to_one",
                    current_format="minus_one_to_one", unpack=False,
                    patch=args.patch_size, max_grid_size=max_grid_size,
                )

            tiles_ref, tile_indices = tile_sampler(ref_images, batch)
            tiles_pred, _ = tile_sampler(recon_images, batch, indices=tile_indices)

            B = tiles_ref.shape[0]
            tiles_ref = tiles_ref.reshape(B * args.n_tiles, 3, args.tile_size, args.tile_size)
            tiles_pred = tiles_pred.reshape(B * args.n_tiles, 3, args.tile_size, args.tile_size)

            with torch.autocast(device_type='cuda', dtype=dtype):
                if args.ssim > 0:
                    ssim_val = SSIM(preds=tiles_pred, target=tiles_ref, data_range=2.0)
                    ssim_loss = 1.0 - ssim_val
                    loss = loss + args.ssim * ssim_loss

                if args.dino_perceptual > 0 and dino_loss_fn is not None:
                    dino_loss = dino_loss_fn(tiles_pred, tiles_ref).mean()
                    loss = loss + args.dino_perceptual * dino_loss

        # Backward
        loss.backward()

        # Gradient clipping
        grad_norm = tu.clip_grad_norm_(grad_params, args.grad_clip, use_fsdp=use_fsdp, world_size=world_size)

        optimizer.step()
        step += 1
        train_state['step'] = step
        pbar.update(1)

        # Accumulate metrics
        log_metrics['loss/total'] = log_metrics.get('loss/total', 0) + loss.item()
        log_metrics['loss/charb'] = log_metrics.get('loss/charb', 0) + charb_loss.item()
        log_metrics['loss/ssim'] = log_metrics.get('loss/ssim', 0) + ssim_loss.item()
        log_metrics['loss/dino'] = log_metrics.get('loss/dino', 0) + dino_loss.item()
        log_count += 1

        # Log
        if step % args.log_freq == 0:
            elapsed = time.perf_counter() - t_log_start
            avg = {k: v / log_count for k, v in log_metrics.items()}
            avg['training/lr'] = lr
            avg['training/grad_norm'] = float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            avg['timing/samples_per_sec'] = (args.batch_size * log_count) / elapsed

            if rank == 0:
                print(f"Step {step}/{args.steps} | "
                      f"loss: {avg['loss/total']:.4f} | "
                      f"charb: {avg['loss/charb']:.4f} | "
                      f"ssim: {avg['loss/ssim']:.4f} | "
                      f"dino: {avg['loss/dino']:.4f} | "
                      f"lr: {lr:.2e}")

            if wandb_enabled:
                wandb.log(avg, step=step)

            log_metrics = {}
            log_count = 0
            t_log_start = time.perf_counter()

        # Evaluation
        if args.eval_freq > 0 and step % args.eval_freq == 0:
            model.eval()
            eval_metrics.reset()

            # Run a few batches for quick eval
            n_eval_batches = 10
            all_ref = []
            all_recon = []

            with torch.no_grad():
                for _ in range(n_eval_batches):
                    try:
                        eval_batch, _ = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(loader)
                        eval_batch, _ = next(loader_iter)

                    eval_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
                    if 'patches' in eval_batch:
                        eval_batch['patches'] = eval_batch['patches'].to(dtype)

                    with torch.autocast(device_type='cuda', dtype=dtype):
                        eval_out = model(eval_batch)

                    recon = postprocess_images(
                        eval_out, output_format="minus_one_to_one",
                        current_format="minus_one_to_one", unpack=True,
                        patch=args.patch_size, max_grid_size=max_grid_size,
                    )
                    ref = postprocess_images(
                        eval_batch, output_format="minus_one_to_one",
                        current_format="minus_one_to_one", unpack=True,
                        patch=args.patch_size, max_grid_size=max_grid_size,
                    )

                    if isinstance(ref, list):
                        all_ref.extend([x.cpu() for x in ref])
                        all_recon.extend([x.cpu() for x in recon])
                    else:
                        all_ref.append(ref.cpu())
                        all_recon.append(recon.cpu())

            eval_metrics.update(all_ref, all_recon)
            eval_stats = eval_metrics.gather()

            if rank == 0:
                print(f"[Eval @ {step}] SSIM: {eval_stats.get('ssim', 0):.4f} | PSNR: {eval_stats.get('psnr', 0):.2f}")

            if wandb_enabled:
                wandb.log({f"eval/{k}": v for k, v in eval_stats.items()}, step=step)

            model.train()

        # Save checkpoint
        if step % args.save_freq == 0:
            tu.save_checkpoint(train_state, str(output_dir), step, rank, world_size)
            if rank == 0:
                print(f"Saved checkpoint at step {step}")

        # Marked checkpoint
        if args.marked_freq > 0 and step % args.marked_freq == 0:
            tu.save_marked_checkpoint(train_state, str(output_dir), step, rank)
            if rank == 0:
                print(f"Saved marked checkpoint at step {step}")

    pbar.close()

    # Final save
    tu.save_checkpoint(train_state, str(output_dir), step, rank, world_size)
    if rank == 0:
        print(f"\nTraining complete! Final checkpoint saved.")

    if wandb_enabled:
        wandb.finish()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
