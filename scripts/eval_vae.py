#!/usr/bin/env python
"""Evaluate ViTok VAE reconstruction quality.

See README.md for full usage examples.

Quick start:
    modal run scripts/eval_vae.py --model 350M-f16x64 --data coco
    modal run scripts/eval_vae.py --baseline flux --data coco
"""
import argparse
import json
import time
from pathlib import Path
import torch
from tqdm import tqdm

import torch.distributed as dist
import modal

from vitok import AE, decode_variant
from safetensors.torch import load_file
from vitok.utils import setup_distributed
from vitok.data import create_dataloader
from vitok.pp.io import postprocess
from vitok.metrics import MetricCalculator
from vitok.pretrained import load_pretrained
from torch.utils.flop_counter import FlopCounterMode
from scripts.modal.modal_config import image, gpu, DATASET_PATHS


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


def evaluate(
    checkpoint: str | None = None,
    model_name: str | None = None,
    baseline: str | None = None,
    variant: str | None = None,
    data: str = "coco",
    max_size: int = 512,
    batch_size: int = 16,
    num_samples: int = 5000,
    crop_style: str = "native",
    swa_window: int | None = None,
    metrics: tuple[str, ...] = ("fid", "fdd", "ssim", "psnr"),
    compile: bool = True,
    float8_mode: str | None = "inference",
    attn_backend: str = "flash",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    verbose: bool = True,
    save_visuals: int = 0,
    output_dir: str | Path | None = None,
) -> dict:
    """Evaluate VAE reconstruction quality.

    Args:
        checkpoint: Path to checkpoint file (mutually exclusive with model_name/baseline)
        model_name: Pretrained model name, e.g. "350M-f16x64" (mutually exclusive with checkpoint/baseline)
        baseline: Baseline VAE name - "flux", "sd", or "qwen" (mutually exclusive with model_name/checkpoint)
        variant: Model variant string. Required if using checkpoint, inferred if using model_name
        data: Data source - dataset name ("coco", "div8k"), local path, or hf:// URL
        max_size: Maximum image size
        batch_size: Batch size for evaluation
        num_samples: Number of samples to evaluate
        crop_style: Crop style - "native" (preserve aspect ratio) or "adm_square" (center crop)
        swa_window: Sliding window attention radius (None=full attention, try 1024 if OOM)
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
    is_baseline = baseline is not None

    # Setup device and dtype
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        # Baselines work better with fp16, ViTok with bf16
        dtype = torch.float16 if is_baseline else (torch.bfloat16 if device.type == "cuda" else torch.float32)

    # Check if distributed is initialized
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_distributed = world_size > 1

    # ==========================================================================
    # Load model (ViTok or Baseline)
    # ==========================================================================
    if is_baseline:
        from scripts.eval.baselines import BaselineVAE
        do_compile = compile and device.type == "cuda" and not is_distributed
        vae = BaselineVAE(baseline, device=device, dtype=dtype, compile=do_compile)
        patch_size = 8  # Baselines use 8x compression
        model_label = f"Baseline: {baseline}" + (" (compiled)" if do_compile else "")
        encoder = decoder = None
    else:
        # Resolve ViTok model and variant
        if model_name is not None:
            pretrained = load_pretrained(model_name)
            variant = pretrained['variant']
            encoder_weights = pretrained['encoder']
            decoder_weights = pretrained['decoder']
        elif checkpoint is None:
            raise ValueError("Either checkpoint, model_name, or baseline must be provided")
        else:
            if variant is None:
                raise ValueError("variant must be provided when using checkpoint path")
            weights = {}
            for key, value in load_file(checkpoint).items():
                weights[key.replace("_orig_mod.", "")] = value
            encoder_weights = weights
            decoder_weights = weights

        config = decode_variant(variant)
        if swa_window is not None:
            config["sw"] = swa_window

        # Create models (float8 auto-applied on load_state_dict)
        encoder = AE(**config, decoder=False, float8_mode=float8_mode, attn_backend=attn_backend).to(device=device, dtype=dtype)
        encoder.load_state_dict(encoder_weights)
        encoder.eval()

        decoder = AE(**config, encoder=False, float8_mode=float8_mode, attn_backend=attn_backend).to(device=device, dtype=dtype)
        decoder.load_state_dict(decoder_weights)
        decoder.eval()

        patch_size = encoder.spatial_stride
        model_label = f"Model: {model_name or checkpoint} ({variant})"
        vae = None

    max_tokens = (max_size // patch_size) ** 2

    # Print model info
    if verbose:
        print(model_label)
        print(f"Evaluating: {data} at {max_size}px ({crop_style})")

    # ==========================================================================
    # Compile and Warmup
    # ==========================================================================
    do_compile = compile and device.type == "cuda" and not is_distributed
    encoder_flops = decoder_flops = 0

    if is_baseline:
        # Baseline warmup (compilation done in BaselineVAE.__init__)
        if do_compile:
            vae.warmup(size=max_size)
    else:
        # Verify SWA is set correctly
        if verbose and swa_window is not None:
            print(f"SWA window: {swa_window} (encoder.sw={encoder.sw}, decoder.sw={decoder.sw})")

        # ViTok compile and warmup
        if do_compile:
            encoder = torch.compile(encoder, fullgraph=True, mode="max-autotune")
            decoder = torch.compile(decoder, fullgraph=True, mode="max-autotune")

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

    # ==========================================================================
    # Create dataloader
    # ==========================================================================
    if is_baseline:
        # Simple preprocessing (images in [0, 1])
        # Native: resize only, let BaselineVAE handle padding to 8
        # ADM: center crop to square
        if crop_style == "native":
            pp = f"resize_longest_side({max_size})|to_tensor"
        else:
            pp = f"resize_longest_side({max_size})|center_crop({max_size})|to_tensor"
        loader = create_dataloader(data, pp, batch_size=batch_size, num_samples=num_samples, drop_last=is_distributed)
    else:
        # ViTok with local data (patchified)
        if crop_style == "adm_square":
            pp = f"center_crop({max_size})|to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"
        else:
            pp = f"resize_longest_side({max_size})|to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"
        loader = create_dataloader(data, pp, batch_size=batch_size, drop_last=is_distributed)

    # Initialize metrics
    metric_calc = MetricCalculator(metrics=metrics)
    metric_calc.move_model_to_device(device, dtype=dtype)

    # Measure FLOPs on first batch (ViTok only, outside main loop)
    if not is_baseline:
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

    # ==========================================================================
    # Main evaluation loop
    # ==========================================================================
    grid_size = max_size // patch_size

    for batch in tqdm(loader, disable=not verbose):
        if samples_seen >= num_samples:
            break

        batch_start = time.perf_counter()

        # Forward pass: get ref/recon in [0, 1] for visuals, [-1, 1] for metrics
        if is_baseline:
            images = batch["image"].to(device, dtype=dtype) if isinstance(batch, dict) else batch.to(device, dtype=dtype)
            with torch.no_grad():
                recon = vae.encode_decode(images)
            # Baseline outputs [0,1], postprocess converts to [-1,1] for metrics
            ref = postprocess(images, current_format="zero_to_one", output_format="minus_one_to_one")
            recon = postprocess(recon, current_format="zero_to_one", output_format="minus_one_to_one")
            batch_size_actual = len(images)
        else:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
                encoded = encoder.encode(batch)
                output = decoder.decode(encoded)
            # ViTok patch dicts are [-1,1], postprocess unpatchifies and keeps [-1,1]
            ref = postprocess(batch, do_unpack=True, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
            recon = postprocess(output, do_unpack=True, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
            batch_size_actual = len(batch["patches"])

        # Update metrics (expects [-1, 1] range)
        metric_calc.update(ref, recon)

        # Collect visuals (convert to [0, 1] float32 for saving)
        if save_visuals > 0 and len(visual_originals) < save_visuals:
            for i in range(min(len(ref), save_visuals - len(visual_originals))):
                visual_originals.append(((ref[i].float() + 1.0) / 2.0).clamp(0, 1).cpu())
                visual_recons.append(((recon[i].float() + 1.0) / 2.0).clamp(0, 1).cpu())

        samples_seen += batch_size_actual

        batch_time = time.perf_counter() - batch_start
        inference_times.append(batch_time)

    eval_end_time = time.perf_counter()
    total_eval_time = eval_end_time - eval_start_time

    # ==========================================================================
    # Gather results
    # ==========================================================================
    stats = metric_calc.gather()
    stats["samples"] = samples_seen
    stats["max_size"] = max_size
    stats["batch_size"] = batch_size
    stats["total_time_sec"] = total_eval_time
    stats["throughput_img_per_sec"] = samples_seen / total_eval_time if total_eval_time > 0 else 0

    if is_baseline:
        stats["baseline"] = baseline
        stats["compiled"] = do_compile
    else:
        stats["model"] = model_name or checkpoint
        stats["variant"] = variant
        stats["crop_style"] = crop_style
        stats["swa_window"] = swa_window
        stats["compiled"] = do_compile
        stats["float8_mode"] = float8_mode

    stats["data"] = data

    # Add timing stats
    if inference_times:
        latency_times = inference_times[1:] if len(inference_times) > 1 else inference_times
        stats["avg_batch_latency_ms"] = sum(latency_times) / len(latency_times) * 1000
        stats["avg_img_latency_ms"] = stats["avg_batch_latency_ms"] / batch_size

    # Add memory stats
    if device.type == "cuda":
        stats["memory_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024**3)
        stats["memory_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024**3)
        stats["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)

    # Add FLOPs stats (ViTok only)
    if encoder_flops > 0 or decoder_flops > 0:
        total_flops = encoder_flops + decoder_flops
        stats["encoder_gflops"] = encoder_flops / 1e9
        stats["decoder_gflops"] = decoder_flops / 1e9
        stats["total_gflops_per_img"] = total_flops / 1e9

    # Save visuals as tensors (can generate comparison grids later)
    if save_visuals > 0 and output_dir is not None:
        # Create model-specific subdirectory
        model_subdir = baseline if is_baseline else model_name
        visuals_dir = Path(output_dir) / model_subdir
        visuals_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"ref": visual_originals, "recon": visual_recons}, visuals_dir / "visuals.pt")
        if verbose:
            print(f"Saved {len(visual_originals)} visual samples to: {visuals_dir / 'visuals.pt'}")

    return stats


def _print_results(result: dict, model: str, max_size: int, crop_style: str, output_json: str = None):
    """Print evaluation results and optionally save to JSON."""
    print(f"\n{'='*60}")
    print(f"Model: {model} @ {max_size}px ({crop_style})")
    print(f"{'='*60}")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_json}")


def main():
    """Local execution: python scripts/eval_vae.py --model X --data /path/to/images"""
    parser = argparse.ArgumentParser(description="Evaluate ViTok VAE")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Path to checkpoint file")
    group.add_argument("--model", help="Pretrained model name (e.g., 350M-f16x64)")
    parser.add_argument("--variant", default=None, help="Model variant (required if using --checkpoint)")
    parser.add_argument("--data", required=True, help="Path to evaluation data")
    parser.add_argument("--max-size", type=int, default=256, help="Max image size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--crop-style", default="native", choices=["native", "adm_square"], help="Crop style")
    parser.add_argument("--swa-window", type=int, default=None, help="Sliding window attention radius (try 1024 if OOM)")
    parser.add_argument("--metrics", nargs="+", default=["fid", "fdd", "ssim", "psnr"], help="Metrics to compute")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--float8", choices=["inference", "training", "none"], default="inference")
    parser.add_argument("--attn-backend", choices=["flex", "flash", "sdpa"], default="flash", help="Attention backend")
    parser.add_argument("--save-visuals", type=int, default=0, help="Number of sample images to save")
    parser.add_argument("--output-dir", default=None, help="Directory to save visuals")
    parser.add_argument("--output-json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

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
        float8_mode=None if args.float8 == "none" else args.float8,
        attn_backend=args.attn_backend,
        device=device,
        verbose=(rank == 0),
        save_visuals=args.save_visuals if rank == 0 else 0,
        output_dir=args.output_dir,
    )

    if rank == 0:
        _print_results(stats, args.model or args.checkpoint, args.max_size, args.crop_style, args.output_json)


# =============================================================================
# Modal support: modal run scripts/eval_vae.py --model X --data coco
# =============================================================================
app = modal.App("vitok-eval")


@app.function(image=image, **gpu("H100"))
def run_eval_remote(
    model: str | None = None,
    baseline: str | None = None,
    data: str = "coco",
    max_size: int = 256,
    crop_style: str = "native",
    num_samples: int = 5000,
    batch_size: int = 64,
    swa_window: int | None = None,
    metrics: list[str] | None = None,
    no_compile: bool = False,
    float8: str | None = "inference",
    attn_backend: str = "flash",
    save_visuals: int = 0,
    output_dir: str | None = None,
) -> dict:
    """Run evaluation on Modal cloud GPU."""
    from scripts.eval_vae import evaluate

    # Map volume aliases to paths (e.g., "coco-val" -> "/data/coco/val2017")
    if data in DATASET_PATHS:
        data = DATASET_PATHS[data]

    return evaluate(
        model_name=model,
        baseline=baseline,
        data=data,
        max_size=max_size,
        crop_style=crop_style,
        num_samples=num_samples,
        batch_size=batch_size,
        swa_window=swa_window,
        metrics=tuple(metrics or ["fid", "fdd", "ssim", "psnr"]),
        compile=not no_compile,
        float8_mode=float8,
        attn_backend=attn_backend,
        save_visuals=save_visuals,
        output_dir=output_dir or ("/output/eval" if save_visuals > 0 else None),
    )


@app.local_entrypoint()
def modal_main(
    model: str = None,
    baseline: str = None,
    data: str = "coco",
    max_size: int = 256,
    crop_style: str = "native",
    num_samples: int = 5000,
    batch_size: int = 64,
    swa_window: int = None,
    metrics: str = "fid,fdd,ssim,psnr",
    no_compile: bool = False,
    float8: str = "inference",
    attn_backend: str = "flash",
    save_visuals: int = 0,
    output_dir: str = None,
    output_json: str = None,
):
    """Modal entrypoint for VAE evaluation. See README.md for examples."""
    if model is None and baseline is None:
        raise ValueError("Either --model or --baseline must be provided")

    result = run_eval_remote.remote(
        model=model,
        baseline=baseline,
        data=data,
        max_size=max_size,
        crop_style=crop_style,
        num_samples=num_samples,
        batch_size=batch_size,
        swa_window=swa_window,
        metrics=metrics.split(","),
        no_compile=no_compile,
        float8=float8,
        attn_backend=attn_backend,
        save_visuals=save_visuals,
        output_dir=output_dir,
    )

    model_label = model or f"baseline:{baseline}"
    _print_results(result, model_label, max_size, crop_style, output_json)


if __name__ == "__main__":
    main()
