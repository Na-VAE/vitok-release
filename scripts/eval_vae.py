#!/usr/bin/env python
"""Evaluate ViTok VAE reconstruction quality.

Usage:
    # Local evaluation with checkpoint path
    python scripts/eval_vae.py --checkpoint model.safetensors --data /path/to/images

    # With pretrained model name (downloads from HuggingFace)
    python scripts/eval_vae.py --model L-64 --data /path/to/images

    # Full options
    python scripts/eval_vae.py --model L-64 --data ./data/coco/val2017 --max-size 512 --num-samples 5000

    # Run on Modal cloud GPU (recommended - no local GPU needed!)
    python scripts/eval_vae.py --modal --model L-64 --num-samples 5000

    # Modal with dataset preset (coco-val, imagenet-val, div8k)
    python scripts/eval_vae.py --modal --model L-64 --dataset coco-val
"""
import argparse
import time
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

import torch.distributed as dist

from vitok import AE, decode_variant
from safetensors.torch import load_file
from vitok.utils import setup_distributed
from vitok.data import create_dataloader
from vitok.pp.io import postprocess
from vitok.metrics import MetricCalculator
from vitok.pretrained import load_pretrained
from torch.utils.flop_counter import FlopCounterMode


def measure_flops(encoder, decoder, batch, device, dtype):
    """Measure FLOPs on a sample batch."""
    if FlopCounterMode is None:
        return 0, 0
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            with FlopCounterMode(display=False) as fc:
                _ = encoder.encode(batch)
            encoder_flops = fc.get_total_flops()

            encoded = encoder.encode(batch)
            with FlopCounterMode(display=False) as fc:
                _ = decoder.decode(encoded)
            decoder_flops = fc.get_total_flops()
        return encoder_flops, decoder_flops
    except Exception:
        return 0, 0


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
    float8_mode: str | None = None,
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
        float8_mode: Quantization mode - "inference" for FP8/INT8, None for bf16
        device: Device to use (default: auto-detect)
        dtype: Data type (default: bf16 on CUDA, fp32 on CPU)
        verbose: Print progress
        save_visuals: Number of sample images to save (0=none)
        output_dir: Directory to save visuals and results (required if save_visuals > 0)

    Returns:
        Dictionary with computed metrics
    """
    # Check if distributed is initialized
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_distributed = world_size > 1

    # Resolve model and variant
    if model_name is not None:
        pretrained = load_pretrained(model_name)
        variant = pretrained['variant']
        encoder_weights = pretrained['encoder']
        decoder_weights = pretrained['decoder']
    elif checkpoint is None:
        raise ValueError("Either checkpoint or model_name must be provided")
    else:
        if variant is None:
            raise ValueError("variant must be provided when using checkpoint path")
        weights = {}
        for key, value in load_file(checkpoint).items():
            weights[key.replace("_orig_mod.", "")] = value
        encoder_weights = weights
        decoder_weights = weights

    # Setup device and dtype
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Print model info
    if verbose:
        print(f"Model: {model_name or checkpoint} ({variant})")
        print(f"Evaluating: {data} at {max_size}px ({crop_style})")

    config = decode_variant(variant)
    if swa_window is not None:
        config["sw"] = swa_window

    # Create models (float8 auto-applied on load_state_dict)
    encoder = AE(**config, decoder=False, float8_mode=float8_mode).to(device=device, dtype=dtype)
    encoder.load_state_dict(encoder_weights, strict=False)
    encoder.eval()

    decoder = AE(**config, encoder=False, float8_mode=float8_mode).to(device=device, dtype=dtype)
    decoder.load_state_dict(decoder_weights, strict=False)
    decoder.eval()

    patch_size = encoder.spatial_stride
    max_tokens = (max_size // patch_size) ** 2

    # Compile for performance (disable for multi-GPU to avoid flex_attention issues)
    do_compile = compile and device.type == "cuda" and not is_distributed
    if do_compile:
        encoder = torch.compile(encoder)
        decoder = torch.compile(decoder)

        # Warmup to trigger compilation
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

    # Reset memory stats before eval
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Create dataloader with appropriate preprocessing
    if crop_style == "adm_square":
        # ADM-style: center crop to square
        pp = f"center_crop({max_size})|to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"
    else:
        # Native: preserve aspect ratio, resize longest side
        pp = f"resize_longest_side({max_size})|to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"
    # Use drop_last=True for distributed to ensure consistent batch sizes across GPUs
    loader = create_dataloader(data, pp, batch_size=batch_size, drop_last=is_distributed)

    # Initialize metrics
    metric_calc = MetricCalculator(metrics=metrics)
    metric_calc.move_model_to_device(device, dtype=dtype)

    # Measure FLOPs on first batch (outside main loop)
    first_batch = next(iter(loader))
    first_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in first_batch.items()}
    encoder_flops, decoder_flops = measure_flops(encoder, decoder, first_batch, device, dtype)

    # Collect samples for visualization
    visual_originals = []
    visual_recons = []

    # Timing tracking
    inference_times = []
    samples_seen = 0
    eval_start_time = time.perf_counter()

    for batch in tqdm(loader, disable=not verbose):
        if samples_seen >= num_samples:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_start = time.perf_counter()

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            encoded = encoder.encode(batch)
            output = decoder.decode(encoded)

        batch_time = time.perf_counter() - batch_start
        inference_times.append(batch_time)
        grid_size = max_size // patch_size
        ref = postprocess(batch, do_unpack=True, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
        recon = postprocess(output, do_unpack=True, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
        metric_calc.update(ref, recon)
        samples_seen += len(batch["patches"])

        # Collect samples for visualization (convert to [0, 1] for saving)
        if save_visuals > 0 and len(visual_originals) < save_visuals:
            for i in range(min(len(ref), save_visuals - len(visual_originals))):
                visual_originals.append(((ref[i] + 1.0) / 2.0).clamp(0, 1).cpu())
                visual_recons.append(((recon[i] + 1.0) / 2.0).clamp(0, 1).cpu())

    eval_end_time = time.perf_counter()
    total_eval_time = eval_end_time - eval_start_time

    # Gather results
    stats = metric_calc.gather()
    stats["samples"] = samples_seen
    stats["model"] = model_name or checkpoint
    stats["variant"] = variant
    stats["crop_style"] = crop_style
    stats["swa_window"] = swa_window
    stats["max_size"] = max_size
    stats["batch_size"] = batch_size
    stats["compiled"] = do_compile
    stats["float8_mode"] = float8_mode

    # Add timing stats
    stats["total_time_sec"] = total_eval_time
    stats["throughput_img_per_sec"] = samples_seen / total_eval_time if total_eval_time > 0 else 0
    if inference_times:
        # Skip first batch (warmup) for latency stats
        latency_times = inference_times[1:] if len(inference_times) > 1 else inference_times
        stats["avg_batch_latency_ms"] = sum(latency_times) / len(latency_times) * 1000
        stats["avg_img_latency_ms"] = stats["avg_batch_latency_ms"] / batch_size

    # Add memory stats
    if device.type == "cuda":
        stats["memory_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024**3)
        stats["memory_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024**3)
        stats["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)

    # Add FLOPs stats
    if encoder_flops > 0 or decoder_flops > 0:
        total_flops = encoder_flops + decoder_flops
        stats["encoder_gflops"] = encoder_flops / 1e9
        stats["decoder_gflops"] = decoder_flops / 1e9
        stats["total_gflops_per_img"] = total_flops / 1e9

    # Save visuals
    if save_visuals > 0 and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison grid and individual samples
        save_comparison_grid(visual_originals, visual_recons, output_dir / "comparison_grid.jpg", max_images=save_visuals)
        save_individual_samples(visual_originals, visual_recons, output_dir / "samples")
        if verbose:
            print(f"Saved visuals to: {output_dir}")

    return stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate ViTok VAE")

    # Modal flag
    parser.add_argument("--modal", action="store_true", help="Run on Modal cloud GPU")
    parser.add_argument("--dataset", choices=["coco-val", "imagenet-val", "div8k"],
                        help="Dataset preset (for --modal, provides data path)")

    # Model selection (mutually exclusive unless using --modal with checkpoint on volume)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", help="Path to checkpoint file")
    group.add_argument("--model", help="Pretrained model name (e.g., L-64)")
    parser.add_argument("--variant", default=None, help="Model variant (required if using --checkpoint)")
    parser.add_argument("--data", help="Path to evaluation data")
    parser.add_argument("--max-size", type=int, default=256, help="Max image size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--crop-style", default="native", choices=["native", "adm_square"], help="Crop style")
    parser.add_argument("--swa-window", type=int, default=None, help="Sliding window attention radius")
    parser.add_argument("--metrics", nargs="+", default=["fid", "fdd", "ssim", "psnr"], help="Metrics to compute")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--float8", choices=["inference", "training"], default=None,
                        help="Float8 mode: 'inference' (quantization API) or 'training' (float8 training API)")
    parser.add_argument("--save-visuals", type=int, default=0, help="Number of sample images to save")
    parser.add_argument("--output-dir", default=None, help="Directory to save visuals")
    parser.add_argument("--output-json", default=None, help="Save results to JSON file")
    parser.add_argument("--output-csv", default=None, help="Save results to CSV file (one row per run)")
    args = parser.parse_args()

    # Validate args
    if not args.modal and not args.model and not args.checkpoint:
        parser.error("Either --model or --checkpoint is required (unless using --modal)")
    if not args.modal and not args.data:
        parser.error("--data is required (unless using --modal with --dataset)")

    # Run on Modal if requested
    if args.modal:
        from scripts.modal.modal_config import multi_gpu_modal, DATASET_PATHS

        data_path = DATASET_PATHS.get(args.dataset, args.data) if args.dataset else args.data
        if not data_path:
            data_path = DATASET_PATHS["coco-val"]

        eval_kwargs = {
            "model_name": args.model,
            "checkpoint": args.checkpoint,
            "variant": args.variant,
            "data": data_path,
            "max_size": args.max_size,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "crop_style": args.crop_style,
            "swa_window": args.swa_window,
            "metrics": tuple(args.metrics),
            "compile": not args.no_compile,
            "float8_mode": args.float8,
            "save_visuals": args.save_visuals,
            "output_dir": "/tmp/eval_output" if args.save_visuals > 0 else None,
        }

        @multi_gpu_modal("vitok-eval", gpu="H100", timeout=3600)
        def run():
            return evaluate(**eval_kwargs)

        stats = run()
        if args.output_json:
            import json
            with open(args.output_json, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\nResults saved to: {args.output_json}")
        return

    # Setup distributed (for multi-GPU)
    rank, world_size, _, device, _ = setup_distributed()
    stats = evaluate(
        checkpoint=args.checkpoint,
        model_name=args.model,
        variant=args.variant,
        data=args.data,
        max_size=args.max_size,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        crop_style=args.crop_style,
        swa_window=args.swa_window,
        metrics=tuple(args.metrics),
        compile=not args.no_compile,
        float8_mode=args.float8,
        device=device,
        verbose=(rank == 0),
        save_visuals=args.save_visuals if rank == 0 else 0,
        output_dir=args.output_dir,
    )

    # Save results to JSON (only rank 0)
    if rank == 0 and args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

if __name__ == "__main__":
    main()
