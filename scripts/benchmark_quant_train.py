#!/usr/bin/env python
"""Benchmark float8 training vs bf16 training.

Training has 3x the FLOPs of inference (forward + 2x backward), so the
relative overhead of quantization is lower and memory savings matter more.

Usage:
    modal run scripts/benchmark_quant_train.py --model 5B-f16x64 --max-size 512
"""
import json
import time
from pathlib import Path

import modal
import torch

from scripts.modal.modal_config import image, gpu, DATASET_PATHS

app = modal.App("vitok-benchmark-quant-train")


def run_training_benchmark(
    model_name: str,
    max_size: int,
    batch_size: int,
    num_steps: int,
    float8_mode: str | None,
    compile: bool = True,
) -> dict:
    """Run a training benchmark configuration."""
    from vitok import AE, decode_variant
    from vitok.pretrained import load_pretrained

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load model config
    pretrained = load_pretrained(model_name)
    variant = pretrained['variant']
    config = decode_variant(variant)

    # Create full model (encoder + decoder) for training
    # Note: float8 training uses convert_to_float8_training, not the inference path
    model = AE(
        **config,
        encoder=True,
        decoder=True,
        float8_mode=float8_mode if float8_mode == "training" else None,
        attn_backend="flash"
    ).to(device=device, dtype=dtype)

    # Load weights
    for key, value in pretrained['encoder'].items():
        if key in model.state_dict():
            model.state_dict()[key].copy_(value)
    for key, value in pretrained['decoder'].items():
        if key in model.state_dict():
            model.state_dict()[key].copy_(value)

    model.train()

    patch_size = model.spatial_stride
    max_tokens = (max_size // patch_size) ** 2
    grid_size = max_size // patch_size

    # Compile if requested
    if compile:
        model = torch.compile(model, fullgraph=True, mode="max-autotune")

    # Create dummy training data
    row_idx = torch.arange(grid_size, device=device).repeat_interleave(grid_size).unsqueeze(0).expand(batch_size, -1)
    col_idx = torch.arange(grid_size, device=device).repeat(grid_size).unsqueeze(0).expand(batch_size, -1)

    def make_batch():
        return {
            "patches": torch.randn(batch_size, max_tokens, patch_size * patch_size * 3, device=device, dtype=dtype),
            "patch_sizes": torch.tensor([[grid_size, grid_size]], device=device).expand(batch_size, -1),
            "row_idx": row_idx,
            "col_idx": col_idx,
        }

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(3):
        batch = make_batch()
        with torch.autocast(device_type="cuda", dtype=dtype):
            encoded = model.encode(batch)
            output = model.decode(encoded)
            loss = (output["patches"] - batch["patches"]).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    # Benchmark loop
    step_times = []

    for step in range(num_steps):
        batch = make_batch()

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.autocast(device_type="cuda", dtype=dtype):
            encoded = model.encode(batch)
            output = model.decode(encoded)
            loss = (output["patches"] - batch["patches"]).pow(2).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        step_times.append(elapsed)

    # Compute stats
    avg_step_time = sum(step_times) / len(step_times)
    throughput = batch_size / avg_step_time

    return {
        "float8_mode": float8_mode,
        "compiled": compile,
        "num_steps": num_steps,
        "max_size": max_size,
        "batch_size": batch_size,
        "avg_step_time_ms": avg_step_time * 1000,
        "throughput_img_per_sec": throughput,
        "memory_allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "memory_reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }


@app.function(image=image, **gpu("H100", timeout=1800))
def run_benchmark(
    model: str = "5B-f16x64",
    max_size: int = 512,
    batch_size: int = 8,
    num_steps: int = 20,
) -> dict:
    """Run training benchmark configurations."""

    configs = [
        {"float8_mode": None, "label": "bf16 training"},
        {"float8_mode": "training", "label": "float8 training"},
    ]

    results = {}
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Running: {cfg['label']}")
        print(f"{'='*60}")

        try:
            result = run_training_benchmark(
                model_name=model,
                max_size=max_size,
                batch_size=batch_size,
                num_steps=num_steps,
                float8_mode=cfg["float8_mode"],
            )
            result["label"] = cfg["label"]
            results[cfg["label"]] = result

            print(f"  Step time: {result['avg_step_time_ms']:.1f} ms")
            print(f"  Throughput: {result['throughput_img_per_sec']:.2f} img/s")
            print(f"  Peak Memory: {result['max_memory_allocated_gb']:.2f} GB")
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results[cfg["label"]] = {"error": str(e), "label": cfg["label"]}

        # Clear CUDA cache between runs
        torch.cuda.empty_cache()

    return {
        "model": model,
        "max_size": max_size,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "results": results,
    }


def print_comparison(data: dict):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"Training Benchmark: {data['model']} @ {data['max_size']}px, batch={data['batch_size']}")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'Step (ms)':<15} {'Throughput':<15} {'Peak Mem (GB)':<15}")
    print(f"{'-'*65}")

    baseline_time = None
    for label, result in data["results"].items():
        if "error" in result:
            print(f"{label:<20} ERROR: {result['error'][:40]}")
            continue

        step_time = result["avg_step_time_ms"]
        throughput = result["throughput_img_per_sec"]
        memory = result["max_memory_allocated_gb"]

        if baseline_time is None:
            baseline_time = step_time
            speedup = ""
        else:
            ratio = baseline_time / step_time
            speedup = f" ({ratio:.2f}x)"

        print(f"{label:<20} {step_time:>8.1f}{speedup:<7} {throughput:>8.2f} img/s   {memory:>8.2f}")

    print(f"{'='*80}")


@app.local_entrypoint()
def main(
    model: str = "5B-f16x64",
    max_size: int = 512,
    batch_size: int = 8,
    num_steps: int = 20,
    output_json: str = None,
):
    """Run training quantization benchmark on Modal."""
    result = run_benchmark.remote(
        model=model,
        max_size=max_size,
        batch_size=batch_size,
        num_steps=num_steps,
    )

    print_comparison(result)

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to: {output_json}")
