#!/usr/bin/env python
"""Benchmark quantization modes: bf16 vs float8-all vs float8-mlp_only.

Compares inference latency and memory usage across different quantization strategies.

Usage:
    # Run on Modal H100
    modal run scripts/benchmark_quant.py --model 5B-f16x64 --max-size 1024

    # Quick test with fewer samples
    modal run scripts/benchmark_quant.py --model 5B-f16x64 --max-size 512 --num-samples 100
"""
import json
import time
from pathlib import Path

import modal
import torch

from scripts.modal.modal_config import image, gpu, DATASET_PATHS

app = modal.App("vitok-benchmark-quant")


def run_single_benchmark(
    model_name: str,
    data: str,
    max_size: int,
    batch_size: int,
    num_samples: int,
    float8_mode: str | None,
    float8_scope: str,
    compile: bool = True,
) -> dict:
    """Run a single benchmark configuration."""
    from vitok import AE, decode_variant
    from vitok.pretrained import load_pretrained
    from vitok.data import create_dataloader
    from vitok.pp.io import postprocess

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load model
    pretrained = load_pretrained(model_name)
    variant = pretrained['variant']
    config = decode_variant(variant)

    encoder = AE(
        **config,
        decoder=False,
        float8_mode=float8_mode,
        float8_scope=float8_scope,
        attn_backend="flash"
    ).to(device=device, dtype=dtype)
    encoder.load_state_dict(pretrained['encoder'])
    encoder.eval()

    decoder = AE(
        **config,
        encoder=False,
        float8_mode=float8_mode,
        float8_scope=float8_scope,
        attn_backend="flash"
    ).to(device=device, dtype=dtype)
    decoder.load_state_dict(pretrained['decoder'])
    decoder.eval()

    patch_size = encoder.spatial_stride
    max_tokens = (max_size // patch_size) ** 2

    # Compile if requested
    if compile:
        encoder = torch.compile(encoder, fullgraph=True, mode="max-autotune")
        decoder = torch.compile(decoder, fullgraph=True, mode="max-autotune")

        # Warmup
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
        del dummy_batch, dummy_encoded
        torch.cuda.empty_cache()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)

    # Create dataloader
    pp = f"resize_longest_side({max_size})|to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})"
    loader = create_dataloader(data, pp, batch_size=batch_size, drop_last=False)

    # Benchmark loop
    inference_times = []
    samples_seen = 0
    grid_size = max_size // patch_size

    for batch in loader:
        if samples_seen >= num_samples:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            encoded = encoder.encode(batch)
            output = decoder.decode(encoded)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        inference_times.append(elapsed)
        samples_seen += len(batch["patches"])

    # Compute stats
    # Skip first batch (warmup)
    latency_times = inference_times[1:] if len(inference_times) > 1 else inference_times

    return {
        "float8_mode": float8_mode,
        "float8_scope": float8_scope,
        "compiled": compile,
        "samples": samples_seen,
        "max_size": max_size,
        "batch_size": batch_size,
        "avg_batch_latency_ms": sum(latency_times) / len(latency_times) * 1000 if latency_times else 0,
        "avg_img_latency_ms": (sum(latency_times) / len(latency_times) * 1000 / batch_size) if latency_times else 0,
        "memory_allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "memory_reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }


@app.function(image=image, **gpu("H100", timeout=1800))
def run_benchmark(
    model: str = "5B-f16x64",
    data: str = "div8k",
    max_size: int = 1024,
    batch_size: int = 64,
    num_samples: int = 500,
) -> dict:
    """Run all benchmark configurations and return comparison."""
    # Map volume aliases to paths
    if data in DATASET_PATHS:
        data = DATASET_PATHS[data]

    configs = [
        {"float8_mode": None, "float8_scope": "all", "label": "bf16 (no quant)"},
        {"float8_mode": "inference", "float8_scope": "all", "label": "float8-all"},
        {"float8_mode": "inference", "float8_scope": "mlp_only", "label": "float8-mlp_only"},
    ]

    results = {}
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Running: {cfg['label']}")
        print(f"{'='*60}")

        try:
            result = run_single_benchmark(
                model_name=model,
                data=data,
                max_size=max_size,
                batch_size=batch_size,
                num_samples=num_samples,
                float8_mode=cfg["float8_mode"],
                float8_scope=cfg["float8_scope"],
            )
            result["label"] = cfg["label"]
            results[cfg["label"]] = result

            print(f"  Latency: {result['avg_img_latency_ms']:.2f} ms/img")
            print(f"  Peak Memory: {result['max_memory_allocated_gb']:.2f} GB")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[cfg["label"]] = {"error": str(e), "label": cfg["label"]}

        # Clear CUDA cache between runs
        torch.cuda.empty_cache()

    return {
        "model": model,
        "max_size": max_size,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "results": results,
    }


def print_comparison(data: dict):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"Quantization Benchmark: {data['model']} @ {data['max_size']}px")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'Latency (ms/img)':<20} {'Peak Memory (GB)':<20}")
    print(f"{'-'*60}")

    baseline_latency = None
    for label, result in data["results"].items():
        if "error" in result:
            print(f"{label:<20} ERROR: {result['error']}")
            continue

        latency = result["avg_img_latency_ms"]
        memory = result["max_memory_allocated_gb"]

        if baseline_latency is None:
            baseline_latency = latency
            speedup = ""
        else:
            speedup = f" ({latency/baseline_latency:.2f}x)"

        print(f"{label:<20} {latency:>8.2f}{speedup:<12} {memory:>8.2f}")

    print(f"{'='*80}")


@app.local_entrypoint()
def main(
    model: str = "5B-f16x64",
    data: str = "div8k",
    max_size: int = 1024,
    batch_size: int = 64,
    num_samples: int = 500,
    output_json: str = None,
):
    """Run quantization benchmark on Modal.

    Compares:
    - bf16 (no quantization)
    - float8-all (quantize all linear layers including attention)
    - float8-mlp_only (quantize only MLP layers, keep attention in bf16)
    """
    result = run_benchmark.remote(
        model=model,
        data=data,
        max_size=max_size,
        batch_size=batch_size,
        num_samples=num_samples,
    )

    print_comparison(result)

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="5B-f16x64")
    parser.add_argument("--data", default="div8k")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    main(
        model=args.model,
        data=args.data,
        max_size=args.max_size,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_json=args.output_json,
    )
