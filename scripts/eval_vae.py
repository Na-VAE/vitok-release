#!/usr/bin/env python
"""Evaluate ViTok VAE reconstruction quality.

See README.md for full usage examples.

Quick start:
    modal run scripts/eval_vae.py --model 350M-f16x64 --dataset coco --stream
    modal run scripts/eval_vae.py --baseline flux --dataset coco --stream
"""
import argparse
import json
import time
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
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


# =============================================================================
# HuggingFace Streaming Datasets
# =============================================================================

HF_DATASETS = {
    "coco": ("detection-datasets/coco", "val", "image"),
    "div8k": ("Iceclear/DIV8K_TrainingSet", "train", "image"),
    "nature": ("eugenesiow/Div2k", "validation", "hr"),
    "portraits": ("jlbaker361/celebrity-100k", "train", "image"),
    "text": ("nielsr/funsd", "train", "image"),
    "architecture": ("GATE-engine/mini-Unsplash", "train", "image"),
    "animals": ("cats_vs_dogs", "train", "image"),
}


def create_streaming_dataloader(
    dataset_name: str,
    max_size: int,
    batch_size: int,
    num_samples: int,
):
    """Create a dataloader that streams from HuggingFace.

    Args:
        dataset_name: Key from HF_DATASETS
        max_size: Image size (will center crop to square)
        batch_size: Batch size
        num_samples: Max number of samples to stream

    Returns:
        Iterator yielding batches of {"image": tensor [B, C, H, W] in [0, 1]}
    """
    from datasets import load_dataset
    import torchvision.transforms as T

    repo, split, image_key = HF_DATASETS[dataset_name]
    ds = load_dataset(repo, split=split, streaming=True, trust_remote_code=True)

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize(max_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(max_size),
        T.ToTensor(),
    ])

    def batch_iterator():
        batch = []
        count = 0
        for example in ds:
            if count >= num_samples:
                break
            img = example[image_key]
            batch.append(transform(img))
            count += 1
            if len(batch) == batch_size:
                yield {"image": torch.stack(batch)}
                batch = []
        if batch:
            yield {"image": torch.stack(batch)}

    return batch_iterator()


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
    baseline: str | None = None,
    variant: str | None = None,
    data: str = "",
    stream: bool = False,
    dataset: str = "coco",
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
        checkpoint: Path to checkpoint file (mutually exclusive with model_name/baseline)
        model_name: Pretrained model name, e.g. "350M-f16x64" (mutually exclusive with checkpoint/baseline)
        baseline: Baseline VAE name - "flux", "sd", or "qwen" (mutually exclusive with model_name/checkpoint)
        variant: Model variant string. Required if using checkpoint, inferred if using model_name
        data: Path to evaluation data (image folder or WebDataset)
        stream: If True, stream from HuggingFace instead of using data path
        dataset: Dataset name for streaming (key from HF_DATASETS)
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
        vae = BaselineVAE(baseline, device=device, dtype=dtype)
        patch_size = 8  # Baselines use 8x compression
        model_label = f"Baseline: {baseline}"
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
        encoder = AE(**config, decoder=False, float8_mode=float8_mode).to(device=device, dtype=dtype)
        encoder.load_state_dict(encoder_weights)
        encoder.eval()

        decoder = AE(**config, encoder=False, float8_mode=float8_mode).to(device=device, dtype=dtype)
        decoder.load_state_dict(decoder_weights)
        decoder.eval()

        patch_size = encoder.spatial_stride
        model_label = f"Model: {model_name or checkpoint} ({variant})"
        vae = None

    max_tokens = (max_size // patch_size) ** 2

    # Print model info
    if verbose:
        print(model_label)
        source = dataset if stream else data
        print(f"Evaluating: {source} at {max_size}px ({crop_style})")

    # ==========================================================================
    # Compile (ViTok only)
    # ==========================================================================
    do_compile = False
    encoder_flops = decoder_flops = 0

    if not is_baseline:
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

    # ==========================================================================
    # Create dataloader
    # ==========================================================================
    if stream:
        # Stream from HuggingFace
        loader = create_streaming_dataloader(dataset, max_size, batch_size, num_samples)
    elif is_baseline:
        # Baseline with local data: simple preprocessing (outputs [0, 1] tensors)
        pp = f"resize_longest_side({max_size})|center_crop({max_size})|to_tensor"
        loader = create_dataloader(data, pp, batch_size=batch_size, drop_last=is_distributed)
    else:
        # ViTok with local data: patchified preprocessing
        if crop_style == "adm_square":
            pp = f"center_crop({max_size})|to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"
        else:
            pp = f"resize_longest_side({max_size})|to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"
        loader = create_dataloader(data, pp, batch_size=batch_size, drop_last=is_distributed)

    # Initialize metrics
    metric_calc = MetricCalculator(metrics=metrics)
    metric_calc.move_model_to_device(device, dtype=dtype)

    # Measure FLOPs on first batch (ViTok only, outside main loop)
    if not is_baseline and not stream:
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
    for batch in tqdm(loader, disable=not verbose):
        if samples_seen >= num_samples:
            break

        batch_start = time.perf_counter()

        if is_baseline:
            # Baseline: batch is {"image": tensor} with images in [0, 1]
            if isinstance(batch, dict):
                images = batch["image"].to(device, dtype=dtype)
            else:
                images = batch.to(device, dtype=dtype)

            with torch.no_grad():
                recon = vae.encode_decode(images)

            # Metrics expect [-1, 1] range
            ref = images * 2 - 1
            recon_norm = recon * 2 - 1
            metric_calc.update(ref, recon_norm)

            # Collect visuals (keep in [0, 1] for saving)
            if save_visuals > 0 and len(visual_originals) < save_visuals:
                for i in range(min(len(images), save_visuals - len(visual_originals))):
                    visual_originals.append(images[i].cpu())
                    visual_recons.append(recon[i].cpu())

            samples_seen += len(images)

        elif stream:
            # ViTok with streaming: images in [0, 1], need to normalize and patchify
            from vitok.pp.ops import patchify as make_patchify

            if isinstance(batch, dict):
                images = batch["image"].to(device, dtype=dtype)
            else:
                images = batch.to(device, dtype=dtype)

            # Normalize to [-1, 1] and patchify each image
            images_norm = images * 2 - 1  # [0,1] -> [-1,1]
            patchify_fn = make_patchify(patch_size, max_tokens)

            # Patchify each image and collate
            patchified_list = [patchify_fn(img) for img in images_norm]
            patchified = {
                "patches": torch.stack([p["patches"] for p in patchified_list]).to(device),
                "patch_sizes": torch.stack([torch.tensor([p["grid_rows"], p["grid_cols"]]) for p in patchified_list]).to(device),
                "patch_mask": torch.stack([p["patch_mask"] for p in patchified_list]).to(device),
                "row_idx": torch.stack([p["row_idx"] for p in patchified_list]).to(device),
                "col_idx": torch.stack([p["col_idx"] for p in patchified_list]).to(device),
                "orig_height": torch.stack([p["orig_height"] for p in patchified_list]).to(device),
                "orig_width": torch.stack([p["orig_width"] for p in patchified_list]).to(device),
            }

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
                encoded = encoder.encode(patchified)
                output = decoder.decode(encoded)

            grid_size = max_size // patch_size
            ref = postprocess(patchified, do_unpack=True, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
            recon = postprocess(output, do_unpack=True, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
            metric_calc.update(ref, recon)

            # Collect visuals (convert from [-1, 1] to [0, 1] for saving)
            if save_visuals > 0 and len(visual_originals) < save_visuals:
                for i in range(min(len(ref), save_visuals - len(visual_originals))):
                    visual_originals.append(((ref[i] + 1.0) / 2.0).clamp(0, 1).cpu())
                    visual_recons.append(((recon[i] + 1.0) / 2.0).clamp(0, 1).cpu())

            samples_seen += len(images)

        else:
            # ViTok with local data: batch is already patchified dict
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
                encoded = encoder.encode(batch)
                output = decoder.decode(encoded)

            grid_size = max_size // patch_size
            ref = postprocess(batch, do_unpack=True, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
            recon = postprocess(output, do_unpack=True, patch=patch_size, max_grid_size=grid_size, output_format="minus_one_to_one")
            metric_calc.update(ref, recon)

            # Collect visuals (convert from [-1, 1] to [0, 1] for saving)
            if save_visuals > 0 and len(visual_originals) < save_visuals:
                for i in range(min(len(ref), save_visuals - len(visual_originals))):
                    visual_originals.append(((ref[i] + 1.0) / 2.0).clamp(0, 1).cpu())
                    visual_recons.append(((recon[i] + 1.0) / 2.0).clamp(0, 1).cpu())

            samples_seen += len(batch["patches"])

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
    else:
        stats["model"] = model_name or checkpoint
        stats["variant"] = variant
        stats["crop_style"] = crop_style
        stats["swa_window"] = swa_window
        stats["compiled"] = do_compile
        stats["float8_mode"] = float8_mode

    if stream:
        stats["dataset"] = dataset
        stats["stream"] = True

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

    # Save visuals
    if save_visuals > 0 and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_comparison_grid(visual_originals, visual_recons, output_dir / "comparison_grid.jpg", max_images=save_visuals)
        save_individual_samples(visual_originals, visual_recons, output_dir / "samples")
        if verbose:
            print(f"Saved visuals to: {output_dir}")

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
    parser.add_argument("--swa-window", type=int, default=None, help="Sliding window attention radius")
    parser.add_argument("--metrics", nargs="+", default=["fid", "fdd", "ssim", "psnr"], help="Metrics to compute")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--float8", choices=["inference", "training"], default=None)
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
        float8_mode=args.float8,
        device=device,
        verbose=(rank == 0),
        save_visuals=args.save_visuals if rank == 0 else 0,
        output_dir=args.output_dir,
    )

    if rank == 0:
        _print_results(stats, args.model or args.checkpoint, args.max_size, args.crop_style, args.output_json)


# =============================================================================
# Modal support: modal run scripts/eval_vae.py --model X --dataset coco
# =============================================================================
app = modal.App("vitok-eval")


@app.function(image=image, **gpu("H100"))
def run_eval_remote(
    model: str | None = None,
    baseline: str | None = None,
    dataset: str = "coco",
    stream: bool = False,
    max_size: int = 256,
    crop_style: str = "native",
    num_samples: int = 5000,
    batch_size: int = 64,
    swa_window: int | None = None,
    metrics: list[str] | None = None,
    no_compile: bool = False,
    float8: str | None = None,
    save_visuals: int = 0,
    output_dir: str | None = None,
) -> dict:
    """Run evaluation on Modal cloud GPU."""
    import torch
    from scripts.eval_vae import evaluate, HF_DATASETS

    # Determine data path
    if stream:
        data = ""  # Not used when streaming
    elif dataset in DATASET_PATHS:
        data = DATASET_PATHS[dataset]
    elif dataset in HF_DATASETS:
        # Dataset exists in HF_DATASETS but not on volume - must stream
        stream = True
        data = ""
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use --stream for HF datasets or ensure data is on volume.")

    return evaluate(
        model_name=model,
        baseline=baseline,
        data=data,
        stream=stream,
        dataset=dataset,
        max_size=max_size,
        crop_style=crop_style,
        num_samples=num_samples,
        batch_size=batch_size,
        swa_window=swa_window,
        metrics=tuple(metrics or ["fid", "fdd", "ssim", "psnr"]),
        compile=not no_compile,
        float8_mode=float8,
        save_visuals=save_visuals,
        output_dir=output_dir or ("/output/eval" if save_visuals > 0 else None),
    )


@app.local_entrypoint()
def modal_main(
    model: str = None,
    baseline: str = None,
    dataset: str = "coco",
    stream: bool = False,
    max_size: int = 256,
    crop_style: str = "native",
    num_samples: int = 5000,
    batch_size: int = 64,
    swa_window: int = None,
    metrics: str = "fid,fdd,ssim,psnr",
    no_compile: bool = False,
    float8: str = None,
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
        dataset=dataset,
        stream=stream,
        max_size=max_size,
        crop_style=crop_style,
        num_samples=num_samples,
        batch_size=batch_size,
        swa_window=swa_window,
        metrics=metrics.split(","),
        no_compile=no_compile,
        float8=float8,
        save_visuals=save_visuals,
        output_dir=output_dir,
    )

    model_label = model or f"baseline:{baseline}"
    _print_results(result, model_label, max_size, crop_style, output_json)


if __name__ == "__main__":
    main()
