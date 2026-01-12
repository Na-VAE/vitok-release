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
import os
import time
from pathlib import Path
import torch
import torchvision.transforms.functional as TF

import torch.distributed as dist

from vitok import AE, decode_variant
from safetensors.torch import load_file

from vitok.utils import setup_distributed, cleanup_distributed
from vitok.data import create_dataloader
from vitok.pp.io import postprocess
from vitok.metrics import MetricCalculator
from vitok.pretrained import download_pretrained, get_pretrained_info

# FlopCounter for measuring compute (optional, may not be available in all PyTorch versions)
try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:
    FlopCounterMode = None


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
        _, _, variant = get_pretrained_info(model_name)
        checkpoint_paths = download_pretrained(model_name)
        if verbose:
            print(f"Model: {model_name}")
            print(f"  Variant: {variant}")
            print(f"  Weights: {checkpoint_paths}")
    elif checkpoint is None:
        raise ValueError("Either checkpoint or model_name must be provided")
    else:
        checkpoint_paths = [checkpoint]

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
        if float8_mode:
            print(f"Quantization: {float8_mode} (FP8 on H100+, INT8 on A100)")
        if is_distributed:
            rank = dist.get_rank()
            print(f"Distributed: world_size={world_size}, rank={rank}")

    # Load weights (merge if split encoder/decoder files)
    weights = {}
    for path in checkpoint_paths:
        weights.update(load_file(path))

    config = decode_variant(variant)
    # Create models WITHOUT float8_mode - we apply quantization AFTER loading weights
    encoder = AE(**config, decoder=False, float8_mode=None).to(device=device, dtype=dtype)
    encoder.load_state_dict(weights, strict=False)
    encoder.eval()

    decoder = AE(**config, encoder=False, float8_mode=None).to(device=device, dtype=dtype)
    decoder.load_state_dict(weights, strict=False)
    decoder.eval()

    # Apply quantization AFTER loading weights (if requested)
    if float8_mode:
        from vitok.models.ae import _apply_float8
        if verbose:
            print(f"  Applying {float8_mode} quantization...")
        for block in encoder.encoder_blocks:
            _apply_float8(block, float8_mode)
        for block in decoder.decoder_blocks:
            _apply_float8(block, float8_mode)

    # Set SWA window if specified (override the model's default)
    if swa_window is not None:
        encoder.sw = swa_window
        decoder.sw = swa_window

    patch_size = encoder.spatial_stride
    max_tokens = (max_size // patch_size) ** 2

    # Compile for performance (before DDP wrapping)
    # Note: Disable compile for multi-GPU to avoid shape mismatches with flex_attention
    do_compile = compile and device.type == "cuda" and not is_distributed
    if do_compile:
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
    elif is_distributed and verbose:
        print("  Skipping compilation for multi-GPU")

    if verbose:
        print(f"  Patch size: {patch_size}")
        print(f"  Max tokens: {max_tokens}")

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
    if verbose:
        print(f"\nEvaluating on {num_samples} samples...")
    metric_calc = MetricCalculator(metrics=metrics)
    metric_calc.move_model_to_device(device, dtype=dtype)

    # Collect samples for visualization
    visual_originals = []
    visual_recons = []

    # Timing and FLOPs tracking
    inference_times = []
    encoder_flops = 0
    decoder_flops = 0

    samples_seen = 0
    eval_start_time = time.perf_counter()

    for batch in loader:
        if samples_seen >= num_samples:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if "patches" in batch:
            batch["patches"] = batch["patches"].to(dtype)

        # Time inference with proper CUDA synchronization
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_start = time.perf_counter()

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            encoded = encoder.encode(batch)
            output = decoder.decode(encoded)

        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_time = time.perf_counter() - batch_start
        inference_times.append(batch_time)

        # Estimate FLOPs on first batch only (skip for float8 training mode - incompatible with FlopCounter)
        # Also skip if flex_attention (SWA) is used - FlopCounterMode doesn't support it
        if samples_seen == 0 and FlopCounterMode is not None and float8_mode != "training":
            try:
                with FlopCounterMode(display=False) as fc:
                    _ = encoder.encode(batch)
                encoder_flops = fc.get_total_flops()

                with FlopCounterMode(display=False) as fc:
                    _ = decoder.decode(encoded)
                decoder_flops = fc.get_total_flops()
            except NotImplementedError:
                # FlopCounterMode doesn't support flex_attention (used with SWA)
                pass

        # Use [-1, 1] range for metrics (SSIM/PSNR expect data_range=2.0, FID/FDD convert internally)
        # For square crops, images are uniform size so no unpacking needed
        do_unpack = crop_style != "adm_square"
        grid_size = max_size // patch_size
        ref = postprocess(batch, do_unpack=do_unpack, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
        recon = postprocess(output, do_unpack=do_unpack, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
        metric_calc.update(ref, recon)

        # Collect samples for visualization (convert to [0, 1] for saving)
        if save_visuals > 0 and len(visual_originals) < save_visuals:
            # Handle both list (from unpack) and tensor (from no unpack) cases
            ref_list = ref if isinstance(ref, list) else [ref[i] for i in range(ref.shape[0])]
            recon_list = recon if isinstance(recon, list) else [recon[i] for i in range(recon.shape[0])]
            for i in range(min(len(ref_list), save_visuals - len(visual_originals))):
                # Convert from [-1, 1] to [0, 1] for visualization
                visual_originals.append(((ref_list[i] + 1.0) / 2.0).clamp(0, 1).cpu())
                visual_recons.append(((recon_list[i] + 1.0) / 2.0).clamp(0, 1).cpu())

        samples_seen += len(batch["patches"])
        if verbose and samples_seen % (batch_size * 25) == 0:
            elapsed = time.perf_counter() - eval_start_time
            throughput = samples_seen / elapsed
            print(f"  Processed {samples_seen}/{num_samples} samples ({throughput:.1f} img/s)")

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

    if verbose:
        print(f"\nResults ({samples_seen} samples):")
        # Print quality metrics first
        print("  Quality metrics:")
        for k in ["fid", "fdd", "ssim", "psnr"]:
            if k in stats:
                print(f"    {k.upper()}: {stats[k]:.4f}")

        # Print performance stats
        print("  Performance:")
        print(f"    Total time: {total_eval_time:.1f}s")
        print(f"    Throughput: {stats['throughput_img_per_sec']:.1f} img/s")
        if "avg_img_latency_ms" in stats:
            print(f"    Latency: {stats['avg_img_latency_ms']:.2f} ms/img")

        # Print memory stats
        if "max_memory_allocated_gb" in stats:
            print("  Memory:")
            print(f"    Peak allocated: {stats['max_memory_allocated_gb']:.2f} GB")
            print(f"    Reserved: {stats['memory_reserved_gb']:.2f} GB")

        # Print FLOPs stats
        if "total_gflops_per_img" in stats:
            print("  FLOPs (per image):")
            print(f"    Encoder: {stats['encoder_gflops']:.2f} GFLOPs")
            print(f"    Decoder: {stats['decoder_gflops']:.2f} GFLOPs")
            print(f"    Total: {stats['total_gflops_per_img']:.2f} GFLOPs")

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


def _run_on_modal(args):
    """Run evaluation on Modal cloud GPU."""
    import modal
    from scripts.modal_config import (
        EVAL_CONFIG,
        DATASET_PATHS,
        eval_image,
        weights_vol,
        data_vol,
        hf_secret,
    )

    # Resolve data path from dataset preset or args
    if args.dataset:
        data_path = DATASET_PATHS.get(args.dataset, f"/data/{args.dataset}")
    elif args.data:
        data_path = args.data
    else:
        # Default to COCO
        data_path = DATASET_PATHS["coco-val"]

    # Print config
    print(f"Running on Modal cloud GPU")
    print(f"  Model: {args.model or args.checkpoint}")
    print(f"  Data: {data_path}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Max size: {args.max_size}")
    print()

    # Build image with vitok code
    image = (
        eval_image
        .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
        .add_local_file("scripts/eval_vae.py", remote_path="/root/vitok-release/scripts/eval_vae.py")
    )

    app = modal.App("vitok-eval")

    # Capture args for closure
    _model = args.model
    _checkpoint = args.checkpoint
    _variant = args.variant
    _data_path = data_path
    _max_size = args.max_size
    _batch_size = args.batch_size
    _num_samples = args.num_samples
    _crop_style = args.crop_style
    _swa_window = args.swa_window
    _metrics = tuple(args.metrics)
    _compile = not args.no_compile
    _float8 = args.float8
    _save_visuals = args.save_visuals

    @app.function(image=image, serialized=True, **EVAL_CONFIG)
    def remote_evaluate():
        import sys
        import os

        sys.path.insert(0, "/root/vitok-release")
        os.environ["HF_HOME"] = "/cache/huggingface"

        # Check if data exists, download COCO if needed
        from pathlib import Path
        resolved_data = _data_path
        if resolved_data.startswith("/data") and not Path(resolved_data).exists():
            if "coco" in resolved_data:
                print(f"Dataset not cached at {resolved_data}, downloading COCO val2017...")
                coco_dir = Path("/data/coco/val2017")
                coco_dir.parent.mkdir(parents=True, exist_ok=True)
                zip_path = coco_dir.parent / "val2017.zip"
                os.system(f"wget -q --show-progress -O {zip_path} http://images.cocodataset.org/zips/val2017.zip")
                os.system(f"unzip -q {zip_path} -d {coco_dir.parent}")
                os.remove(zip_path)
                resolved_data = str(coco_dir)
                # Note: Volume commit happens automatically on function exit
            else:
                raise FileNotFoundError(
                    f"Dataset not found at {resolved_data}. "
                    f"Run 'modal run scripts/modal/setup_data.py' to cache datasets."
                )

        from scripts.eval_vae import evaluate
        return evaluate(
            model_name=_model,
            checkpoint=_checkpoint,
            variant=_variant,
            data=resolved_data,
            max_size=_max_size,
            batch_size=_batch_size,
            num_samples=_num_samples,
            crop_style=_crop_style,
            swa_window=_swa_window,
            metrics=_metrics,
            compile=_compile,
            float8_mode=_float8,
            save_visuals=_save_visuals,
            output_dir="/tmp/eval_output" if _save_visuals > 0 else None,
        )

    # Run on Modal
    with app.run():
        stats = remote_evaluate.remote()

    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results: {stats.get('model', args.model)}")
    print(f"{'='*50}")
    print(f"Variant: {stats.get('variant', 'N/A')}")
    print(f"Samples: {stats.get('samples', 'N/A')}")
    print()
    print("Metrics:")
    for k in ["fid", "fdd", "ssim", "psnr"]:
        if k in stats:
            print(f"  {k.upper():6s}: {stats[k]:.4f}")

    if "throughput_img_per_sec" in stats:
        print(f"\nThroughput: {stats['throughput_img_per_sec']:.1f} img/s")

    # Save results locally if requested
    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


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
        _run_on_modal(args)
        return

    # Setup distributed (for multi-GPU)
    rank, world_size, _, device, _ = setup_distributed()

    try:
        stats = evaluate(
            checkpoint=args.checkpoint,
            model_name=args.model,
            variant=args.variant,
            data=args.data,
            max_size=args.max_size,
            batch_size=args.batch_size,  # Per-GPU batch size, not divided
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

        # Save results to CSV (only rank 0)
        if rank == 0 and args.output_csv:
            import csv
            from datetime import datetime

            # Define CSV columns (flattened metrics)
            csv_columns = [
                "timestamp", "model", "variant", "data", "max_size", "crop_style",
                "batch_size", "samples", "compiled", "float8_mode",
                "fid", "fdd", "ssim", "psnr",
                "throughput_img_per_sec", "avg_img_latency_ms",
                "max_memory_allocated_gb",
                "encoder_gflops", "decoder_gflops", "total_gflops_per_img",
            ]

            csv_path = Path(args.output_csv)
            write_header = not csv_path.exists()

            # Add timestamp and data path to stats
            row = {"timestamp": datetime.now().isoformat(), "data": args.data}
            row.update({k: stats.get(k, "") for k in csv_columns if k not in row})

            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(f"Results appended to: {args.output_csv}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
