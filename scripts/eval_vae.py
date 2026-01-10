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
from pathlib import Path
import torch
import torchvision.transforms.functional as TF

from vitok import AE, decode_variant
from safetensors.torch import load_file

from vitok.utils import setup_distributed, cleanup_distributed
from vitok.data import create_dataloader
from vitok.pp.io import postprocess
from vitok.metrics import MetricCalculator
from vitok.pretrained import download_pretrained, get_pretrained_info


def save_comparison_grid(
    originals: list[torch.Tensor],
    reconstructions: list[torch.Tensor],
    output_path: Path,
    max_images: int = 8,
) -> None:
    """Save a grid of original | reconstruction | diff images.

    Args:
        originals: List of original images [C, H, W] in [0, 1]
        reconstructions: List of reconstructed images [C, H, W] in [0, 1]
        output_path: Path to save the grid image
        max_images: Maximum number of images to include
    """
    from PIL import Image

    n = min(len(originals), max_images)
    if n == 0:
        return

    # Convert to PIL images and compute diffs
    rows = []
    for i in range(n):
        # Convert to float32 for PIL compatibility
        orig = originals[i].float().clamp(0, 1)
        recon = reconstructions[i].float().clamp(0, 1)

        # Compute diff (amplified for visibility)
        diff = (orig - recon).abs() * 5  # Amplify by 5x
        diff = diff.clamp(0, 1)

        # Convert to PIL
        orig_pil = TF.to_pil_image(orig)
        recon_pil = TF.to_pil_image(recon)
        diff_pil = TF.to_pil_image(diff)

        rows.append((orig_pil, recon_pil, diff_pil))

    # Create grid
    w, h = rows[0][0].size
    grid = Image.new('RGB', (w * 3, h * n))

    for i, (orig, recon, diff) in enumerate(rows):
        grid.paste(orig, (0, i * h))
        grid.paste(recon, (w, i * h))
        grid.paste(diff, (w * 2, i * h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, quality=95)


def save_individual_samples(
    originals: list[torch.Tensor],
    reconstructions: list[torch.Tensor],
    output_dir: Path,
    prefix: str = "sample",
) -> None:
    """Save individual original/reconstruction pairs.

    Args:
        originals: List of original images [C, H, W] in [0, 1]
        reconstructions: List of reconstructed images [C, H, W] in [0, 1]
        output_dir: Directory to save images
        prefix: Filename prefix
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (orig, recon) in enumerate(zip(originals, reconstructions)):
        # Convert to float32 for PIL compatibility
        orig = orig.float().clamp(0, 1)
        recon = recon.float().clamp(0, 1)

        TF.to_pil_image(orig).save(output_dir / f"{prefix}_{i:03d}_orig.png")
        TF.to_pil_image(recon).save(output_dir / f"{prefix}_{i:03d}_recon.png")


def evaluate(
    checkpoint: str | None = None,
    model_name: str | None = None,
    variant: str | None = None,
    data: str = "",
    max_size: int = 512,
    batch_size: int = 16,
    num_samples: int = 5000,
    crop_style: str = "native",
    swa_window: int | None = None,
    metrics: tuple[str, ...] = ("fid", "fdd", "ssim", "psnr"),
    compile: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    verbose: bool = True,
    save_visuals: int = 0,
    output_dir: str | Path | None = None,
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
        crop_style: Crop style - "native" (preserve aspect ratio) or "adm_square" (center crop)
        swa_window: Sliding window attention radius (None=full attention)
        metrics: Tuple of metrics to compute ("fid", "fdd", "ssim", "psnr")
        compile: Whether to use torch.compile
        device: Device to use (default: auto-detect)
        dtype: Data type (default: bf16 on CUDA, fp32 on CPU)
        verbose: Print progress
        save_visuals: Number of sample images to save (0=none)
        output_dir: Directory to save visuals and results (required if save_visuals > 0)

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
        print(f"Crop style: {crop_style}, SWA window: {swa_window}")
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

    # Set SWA window if specified (override the model's default)
    if swa_window is not None:
        encoder.sw = swa_window
        decoder.sw = swa_window

    patch_size = encoder.spatial_stride
    max_tokens = (max_size // patch_size) ** 2

    # Compile for performance
    if compile and device.type == "cuda":
        if verbose:
            print("  Compiling model...")
        encoder = torch.compile(encoder, fullgraph=True)
        decoder = torch.compile(decoder, fullgraph=True)

        # Warmup with dummy batch to trigger compilation
        # This avoids the "flex_attention called without torch.compile()" warning
        grid_size = max_size // patch_size
        row_idx = torch.arange(grid_size, device=device).repeat_interleave(grid_size).unsqueeze(0)
        col_idx = torch.arange(grid_size, device=device).repeat(grid_size).unsqueeze(0)
        dummy_batch = {
            "patches": torch.randn(1, max_tokens, patch_size * patch_size * 3, device=device, dtype=dtype),
            "patch_sizes": torch.tensor([[grid_size, grid_size]], device=device),
            "row_idx": row_idx,
            "col_idx": col_idx,
        }
        with torch.no_grad():
            dummy_encoded = encoder.encode(dummy_batch)
            _ = decoder.decode(dummy_encoded)
        del dummy_batch, dummy_encoded, row_idx, col_idx
        torch.cuda.empty_cache()
        if verbose:
            print("  Compilation complete.")

    if verbose:
        print(f"  Patch size: {patch_size}")
        print(f"  Max tokens: {max_tokens}")

    # Create dataloader with appropriate preprocessing
    if crop_style == "adm_square":
        # ADM-style: center crop to square
        pp = f"center_crop({max_size})|to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"
    else:
        # Native: preserve aspect ratio, resize longest side
        pp = f"resize_longest_side({max_size})|to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"
    loader = create_dataloader(data, pp, batch_size=batch_size)

    # Initialize metrics
    if verbose:
        print(f"\nEvaluating on {num_samples} samples...")
    metric_calc = MetricCalculator(metrics=metrics)
    metric_calc.move_model_to_device(device, dtype=dtype)

    # Collect samples for visualization
    visual_originals = []
    visual_recons = []

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

        ref = postprocess(batch, do_unpack=True, patch=patch_size, output_format="zero_to_one")
        recon = postprocess(output, do_unpack=True, patch=patch_size, output_format="zero_to_one")
        metric_calc.update(ref, recon)

        # Collect samples for visualization (first N images)
        if save_visuals > 0 and len(visual_originals) < save_visuals:
            for i in range(min(len(ref), save_visuals - len(visual_originals))):
                visual_originals.append(ref[i].cpu())
                visual_recons.append(recon[i].cpu())

        samples_seen += len(batch["patches"])
        if verbose and samples_seen % (batch_size * 25) == 0:
            print(f"  Processed {samples_seen}/{num_samples} samples")

    # Gather results
    stats = metric_calc.gather()
    stats["samples"] = samples_seen
    stats["model"] = model_name or checkpoint
    stats["variant"] = variant
    stats["crop_style"] = crop_style
    stats["swa_window"] = swa_window
    stats["max_size"] = max_size

    if verbose:
        print(f"\nResults ({samples_seen} samples):")
        for k, v in sorted(stats.items()):
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")

    # Save visuals
    if save_visuals > 0 and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison grid
        grid_path = output_dir / "comparison_grid.jpg"
        save_comparison_grid(visual_originals, visual_recons, grid_path, max_images=save_visuals)
        if verbose:
            print(f"\nSaved comparison grid to: {grid_path}")

        # Save individual samples
        save_individual_samples(visual_originals, visual_recons, output_dir / "samples")
        if verbose:
            print(f"Saved {len(visual_originals)} individual samples to: {output_dir / 'samples'}")

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
