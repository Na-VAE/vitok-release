#!/usr/bin/env python
"""Evaluate ViTok VAE reconstruction quality.

Usage:
    python scripts/eval_vae.py --checkpoint model.pt --data /path/to/images
    python scripts/eval_vae.py --checkpoint model.pt --data hf://org/repo/val.tar --max-size 512
"""
import argparse
import torch

from vitok import AE, decode_variant
from safetensors.torch import load_file

from vitok.utils import setup_distributed
from vitok.data import create_dataloader
from vitok.pp.io import postprocess
from vitok.evaluators import MetricCalculator


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViTok VAE")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--variant", default="Ld2-Ld22/1x16x64", help="Model variant")
    parser.add_argument("--data", required=True, help="Path or hf:// URL to data")
    parser.add_argument("--max-size", type=int, default=512, help="Max image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--metrics", nargs="+", default=["fid", "ssim", "psnr"], help="Metrics to compute")
    args = parser.parse_args()

    # Setup distributed
    rank, world_size, _, device, _ = setup_distributed()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if rank == 0:
        print(f"Evaluating: {args.checkpoint}")
        print(f"Data: {args.data}")
        print(f"Max size: {args.max_size}, Batch size: {args.batch_size}")

    # Load model
    model = AE(**decode_variant(args.variant))
    model.to(device=device, dtype=dtype)
    model.load_state_dict(load_file(args.checkpoint), strict=True)
    model.eval()

    patch_size = model.spatial_stride
    max_tokens = (args.max_size // patch_size) ** 2

    # Create dataloader
    pp = f"resize_longest_side({args.max_size})|to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"
    loader = create_dataloader(args.data, pp, batch_size=args.batch_size // world_size)

    # Evaluate
    metrics = MetricCalculator(metrics=tuple(args.metrics))
    metrics.move_model_to_device(device)

    samples_seen = 0
    for batch in loader:
        if samples_seen >= args.num_samples:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if 'patches' in batch:
            batch['patches'] = batch['patches'].to(dtype)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=dtype):
            output = model(batch)

        ref = postprocess(batch, do_unpack=True, patch=patch_size)
        recon = postprocess(output, do_unpack=True, patch=patch_size)
        metrics.update(ref, recon)

        samples_seen += len(batch['patches'])

    # Gather and print results
    stats = metrics.gather()

    if rank == 0:
        print(f"\nResults ({samples_seen} samples):")
        for k, v in sorted(stats.items()):
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
