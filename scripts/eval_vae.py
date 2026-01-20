#!/usr/bin/env python
"""Evaluate ViTok VAE reconstruction quality.

Quick start:
    modal run scripts/eval_vae.py --model 350M-f16x64 --data coco
    modal run scripts/eval_vae.py --model flux --data coco
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
from vitok.data import create_dataloader
from vitok.pp.io import postprocess
from vitok.metrics import MetricCalculator
from vitok.pretrained import load_pretrained
from scripts.eval.baselines import BaselineVAE, list_baselines
from scripts.modal.modal_config import image, gpu, DATASET_PATHS


def _is_baseline(model_name: str) -> bool:
    """Check if model_name refers to a baseline VAE."""
    return model_name in list_baselines()


def evaluate(
    model_name: str,
    data: str = "coco",
    max_size: int = 512,
    batch_size: int = 16,
    num_samples: int = 5000,
    crop_style: str = "native",
    swa_window: int | None = None,
    metrics: tuple[str, ...] = ("fid", "fdd", "ssim", "psnr"),
    save_visuals: int = 0,
    output_dir: str | Path | None = None,
) -> dict:
    """Evaluate VAE reconstruction quality.

    Args:
        model_name: Model to evaluate - ViTok ("350M-f16x64") or baseline ("flux", "sdxl", etc.)
        data: Data source - dataset name ("coco", "div8k"), local path, or hf:// URL
        max_size: Maximum image size
        batch_size: Batch size for evaluation
        num_samples: Number of samples to evaluate
        crop_style: Crop style - "native" (preserve aspect ratio) or "adm_center" (center crop)
        swa_window: Sliding window attention radius for ViTok (None=full attention)
        metrics: Tuple of metrics to compute ("fid", "fdd", "ssim", "psnr")
        save_visuals: Number of sample images to save (0=none)
        output_dir: Directory to save visuals and results

    Returns:
        Dictionary with computed metrics
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_distributed = world_size > 1
    is_baseline = _is_baseline(model_name)

    # ==========================================================================
    # Load model
    # ==========================================================================
    if is_baseline:
        vae = BaselineVAE(model_name, device="cuda", dtype=torch.float16, compile=True)
        patch_size = vae.spatial_stride
        variant = None
    else:
        # Load ViTok model
        pretrained = load_pretrained(model_name)
        variant = pretrained["variant"]
        config = decode_variant(variant)
        if swa_window is not None:
            config["sw"] = swa_window
        patch_size = config["spatial_stride"]

        encoder = AE(**config, decoder=False).to(device="cuda", dtype=torch.bfloat16)
        encoder.load_state_dict(pretrained["encoder"])
        encoder.eval()
        encoder.quantize()
        encoder = torch.compile(encoder, fullgraph=True)

        decoder = AE(**config, encoder=False).to(device="cuda", dtype=torch.bfloat16)
        decoder.load_state_dict(pretrained["decoder"])
        decoder.eval()
        decoder.quantize()
        decoder = torch.compile(decoder, fullgraph=True)

    # ==========================================================================
    # Setup data loading
    # ==========================================================================
    max_tokens = (max_size // patch_size) ** 2

    if crop_style == "native":
        pp = f"resize_longest_side({max_size})"
    else:
        pp = f"center_crop({max_size})"
    pp += "|to_tensor"
    if not is_baseline:
        pp += f"|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"

    loader = create_dataloader(data, pp, batch_size=batch_size, num_samples=num_samples, drop_last=is_distributed)

    # Initialize metrics
    metric_calc = MetricCalculator(metrics=metrics)
    metric_calc.move_model_to_device("cuda", dtype=torch.float32)

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

    for batch in tqdm(loader, desc=f"Evaluating {model_name}"):
        if samples_seen >= num_samples:
            break

        if is_baseline:
            images = batch["image"] if isinstance(batch, dict) else batch
            images = images.to("cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                recon = vae.encode_decode(images)
            torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - t0)
            ref = images * 2 - 1
            recon = recon * 2 - 1
            batch_size_actual = len(images)
        else:
            batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoded = encoder(batch)
                output = decoder(encoded)
            torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - t0)
            ref = postprocess(batch, do_unpack=True, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
            recon = postprocess(output, do_unpack=True, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
            batch_size_actual = len(batch["patches"])

        metric_calc.update(ref, recon)

        if save_visuals > 0 and len(visual_originals) < save_visuals:
            for i in range(min(len(ref), save_visuals - len(visual_originals))):
                visual_originals.append(((ref[i].float() + 1.0) / 2.0).clamp(0, 1).cpu())
                visual_recons.append(((recon[i].float() + 1.0) / 2.0).clamp(0, 1).cpu())

        samples_seen += batch_size_actual

    total_eval_time = time.perf_counter() - eval_start_time

    # ==========================================================================
    # Gather results
    # ==========================================================================
    stats = metric_calc.gather()
    stats["model"] = model_name
    stats["samples"] = samples_seen
    stats["max_size"] = max_size
    stats["crop_style"] = crop_style
    stats["data"] = data
    stats["total_time_sec"] = total_eval_time
    stats["throughput_img_per_sec"] = samples_seen / total_eval_time if total_eval_time > 0 else 0

    if not is_baseline:
        stats["variant"] = variant
        stats["swa_window"] = swa_window

    # Add latency stats (skip first batch for warmup)
    if len(inference_times) > 1:
        latency_times = inference_times[1:]
        stats["avg_batch_latency_ms"] = sum(latency_times) / len(latency_times) * 1000
        stats["avg_img_latency_ms"] = stats["avg_batch_latency_ms"] / batch_size

    # Save visuals
    if save_visuals > 0 and output_dir is not None:
        visuals_dir = Path(output_dir) / model_name
        visuals_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"ref": visual_originals, "recon": visual_recons}, visuals_dir / "visuals.pt")

    return stats


def main():
    """Local execution: python scripts/eval_vae.py --model X --data /path/to/images"""
    parser = argparse.ArgumentParser(description="Evaluate ViTok VAE")
    parser.add_argument("--model", required=True, help="Model name (e.g., 350M-f16x64, flux, sdxl)")
    parser.add_argument("--data", required=True, help="Path to evaluation data")
    parser.add_argument("--max-size", type=int, default=256, help="Max image size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--crop-style", default="native", choices=["native", "adm_center"], help="Crop style")
    parser.add_argument("--swa-window", type=int, default=None, help="Sliding window attention radius")
    parser.add_argument("--metrics", nargs="+", default=["fid", "fdd", "ssim", "psnr"], help="Metrics to compute")
    parser.add_argument("--save-visuals", type=int, default=0, help="Number of sample images to save")
    parser.add_argument("--output-dir", default=None, help="Directory to save visuals")
    parser.add_argument("--output-json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    stats = evaluate(
        model_name=args.model,
        data=args.data,
        max_size=args.max_size,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        crop_style=args.crop_style,
        swa_window=args.swa_window,
        metrics=tuple(args.metrics),
        save_visuals=args.save_visuals,
        output_dir=args.output_dir,
    )

    print(json.dumps(stats, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(stats, f, indent=2)


# =============================================================================
# Modal support: modal run scripts/eval_vae.py --model X --data coco
# =============================================================================
app = modal.App("vitok-eval")


@app.function(image=image, **gpu("H100"))
def run_eval_remote(
    model: str,
    data: str = "coco",
    max_size: int = 256,
    crop_style: str = "native",
    num_samples: int = 5000,
    batch_size: int = 64,
    swa_window: int | None = None,
    metrics: list[str] | None = None,
    save_visuals: int = 0,
    output_dir: str | None = None,
) -> dict:
    """Run evaluation on Modal cloud GPU."""
    from scripts.eval_vae import evaluate

    # Map dataset aliases to paths
    if data in DATASET_PATHS:
        data = DATASET_PATHS[data]

    return evaluate(
        model_name=model,
        data=data,
        max_size=max_size,
        crop_style=crop_style,
        num_samples=num_samples,
        batch_size=batch_size,
        swa_window=swa_window,
        metrics=tuple(metrics or ["fid", "fdd", "ssim", "psnr"]),
        save_visuals=save_visuals,
        output_dir=output_dir or ("/output/eval" if save_visuals > 0 else None),
    )


@app.local_entrypoint()
def modal_main(
    model: str = None,
    data: str = "coco",
    max_size: int = 256,
    crop_style: str = "native",
    num_samples: int = 5000,
    batch_size: int = 64,
    swa_window: int = None,
    metrics: str = "fid,fdd,ssim,psnr",
    save_visuals: int = 0,
    output_dir: str = None,
    output_json: str = None,
):
    """Modal entrypoint for VAE evaluation."""
    if model is None:
        raise ValueError("--model is required")

    stats = run_eval_remote.remote(
        model=model,
        data=data,
        max_size=max_size,
        crop_style=crop_style,
        num_samples=num_samples,
        batch_size=batch_size,
        swa_window=swa_window,
        metrics=metrics.split(","),
        save_visuals=save_visuals,
        output_dir=output_dir,
    )

    print(json.dumps(stats, indent=2))
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
