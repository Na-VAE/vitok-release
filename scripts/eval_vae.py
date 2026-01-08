#!/usr/bin/env python
"""Evaluate ViTok VAE reconstruction quality.

Usage:
    # Local evaluation with checkpoint path
    python scripts/eval_vae.py --checkpoint model.safetensors --data /path/to/images

    # With pretrained model name (downloads from HuggingFace)
    python scripts/eval_vae.py --model L-64 --data /path/to/images

    # Full options
    python scripts/eval_vae.py --model L-64 --data ./data/coco/val2017 --max-size 512 --num-samples 5000

    # Modal (recommended for GPU)
    modal run scripts/eval_vae.py --model L-64 --num-samples 5000
"""
import argparse
import os
import torch

from vitok import AE, decode_variant
from safetensors.torch import load_file

from vitok.utils import setup_distributed, cleanup_distributed
from vitok.data import create_dataloader
from vitok.pp.io import postprocess
from vitok.evaluators import MetricCalculator
from vitok.pretrained import download_pretrained, get_pretrained_info


def evaluate(
    checkpoint: str | None = None,
    model_name: str | None = None,
    variant: str | None = None,
    data: str = "",
    max_size: int = 512,
    batch_size: int = 16,
    num_samples: int = 5000,
    metrics: tuple[str, ...] = ("fid", "fdd", "ssim", "psnr"),
    compile: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    verbose: bool = True,
) -> dict:
    """Evaluate VAE reconstruction quality.

    Args:
        checkpoint: Path to checkpoint file (mutually exclusive with model_name)
        model_name: Pretrained model name, e.g. "L-64" (mutually exclusive with checkpoint)
        variant: Model variant string. Required if using checkpoint, inferred if using model_name
        data: Path to evaluation data (image folder or WebDataset)
        max_size: Maximum image size
        batch_size: Batch size for evaluation
        num_samples: Number of samples to evaluate
        metrics: Tuple of metrics to compute ("fid", "fdd", "ssim", "psnr")
        compile: Whether to use torch.compile
        device: Device to use (default: auto-detect)
        dtype: Data type (default: bf16 on CUDA, fp32 on CPU)
        verbose: Print progress

    Returns:
        Dictionary with computed metrics
    """
    # Resolve model and variant
    if model_name is not None:
        _, _, variant = get_pretrained_info(model_name)
        checkpoint = download_pretrained(model_name)
        if verbose:
            print(f"Model: {model_name}")
            print(f"  Variant: {variant}")
            print(f"  Checkpoint: {checkpoint}")
    elif checkpoint is None:
        raise ValueError("Either checkpoint or model_name must be provided")

    if variant is None:
        raise ValueError("variant must be provided when using checkpoint path")

    # Setup device and dtype
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if verbose:
        print(f"\nData: {data}")
        print(f"Max size: {max_size}, Batch size: {batch_size}")
        print(f"Device: {device}, Dtype: {dtype}")

    # Load encoder and decoder separately for better compilation
    weights = load_file(checkpoint)

    encoder = AE(**decode_variant(variant), decoder=False)
    encoder.to(device=device, dtype=dtype)
    encoder.load_state_dict(weights, strict=False)
    encoder.eval()

    decoder = AE(**decode_variant(variant), encoder=False)
    decoder.to(device=device, dtype=dtype)
    decoder.load_state_dict(weights, strict=False)
    decoder.eval()

    patch_size = encoder.spatial_stride

    # Compile for performance
    if compile and device.type == "cuda":
        encoder = torch.compile(encoder, fullgraph=True)
        decoder = torch.compile(decoder, fullgraph=True)

    max_tokens = (max_size // patch_size) ** 2

    if verbose:
        print(f"  Patch size: {patch_size}")
        print(f"  Max tokens: {max_tokens}")
        if compile and device.type == "cuda":
            print("  Compiling model (first run may be slow)...")

    # Create dataloader
    pp = f"resize_longest_side({max_size})|to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"
    loader = create_dataloader(data, pp, batch_size=batch_size)

    # Initialize metrics
    if verbose:
        print(f"\nEvaluating on {num_samples} samples...")
    metric_calc = MetricCalculator(metrics=metrics)
    metric_calc.move_model_to_device(device, dtype=dtype)

    samples_seen = 0
    for batch in loader:
        if samples_seen >= num_samples:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if "patches" in batch:
            batch["patches"] = batch["patches"].to(dtype)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            encoded = encoder.encode(batch)
            output = decoder.decode(encoded)

        ref = postprocess(batch, do_unpack=True, patch=patch_size)
        recon = postprocess(output, do_unpack=True, patch=patch_size)
        metric_calc.update(ref, recon)

        samples_seen += len(batch["patches"])
        if verbose and samples_seen % (batch_size * 25) == 0:
            print(f"  Processed {samples_seen}/{num_samples} samples")

    # Gather results
    stats = metric_calc.gather()
    stats["samples"] = samples_seen
    stats["model"] = model_name or checkpoint
    stats["variant"] = variant

    if verbose:
        print(f"\nResults ({samples_seen} samples):")
        for k, v in sorted(stats.items()):
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViTok VAE")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Path to checkpoint file")
    group.add_argument("--model", help="Pretrained model name (e.g., L-64)")
    parser.add_argument("--variant", default=None, help="Model variant (required if using --checkpoint)")
    parser.add_argument("--data", required=True, help="Path to evaluation data")
    parser.add_argument("--max-size", type=int, default=512, help="Max image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--metrics", nargs="+", default=["fid", "fdd", "ssim", "psnr"], help="Metrics to compute")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    # Setup distributed (for multi-GPU)
    rank, world_size, _, device, _ = setup_distributed()

    try:
        stats = evaluate(
            checkpoint=args.checkpoint,
            model_name=args.model,
            variant=args.variant,
            data=args.data,
            max_size=args.max_size,
            batch_size=args.batch_size // world_size,
            num_samples=args.num_samples,
            metrics=tuple(args.metrics),
            compile=not args.no_compile,
            device=device,
            verbose=(rank == 0),
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
