#!/usr/bin/env python
"""Benchmark to isolate MFU bottleneck: data loading vs compute.

Usage:
    # First sync code
    modal run scripts/modal_train_vae.py --sync-only

    # Run benchmark
    modal run modal_tests/benchmark_mfu.py

    # With options
    modal run modal_tests/benchmark_mfu.py --batch-size 256 --n-steps 50
"""

import modal
from pathlib import Path

app = modal.App("vitok-mfu-benchmark")

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .pip_install(
        "torch>=2.5.0",
        "torchvision",
        "numpy",
        "scipy",
        "tqdm",
        "safetensors",
        "webdataset",
        "Pillow",
        "huggingface_hub",
    )
)

code_volume = modal.Volume.from_name("vitok-code", create_if_missing=True)


@app.function(
    image=base_image,
    gpu="A100-80GB",
    timeout=1800,
    volumes={"/code": code_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def benchmark_mfu(
    batch_size: int = 256,
    max_tokens: int = 256,
    n_steps: int = 50,
    n_warmup: int = 5,
):
    """Benchmark MFU with synthetic data vs real data loading."""
    import sys
    import os
    import time

    os.environ["PYTHONPATH"] = "/code/vitok-release:/code/dino_perceptual"
    sys.path.insert(0, "/code/vitok-release")
    sys.path.insert(0, "/code/dino_perceptual")
    os.chdir("/code/vitok-release")

    import torch
    from vitok import AEConfig, create_ae
    from vitok.data import create_dataloader

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Create model
    print("=" * 60)
    print("MFU Benchmark")
    print("=" * 60)

    config = AEConfig(variant="Ld2-Ld22/1x16x64")
    model = create_ae(config)
    model.to(device=device, dtype=dtype)
    model = torch.compile(model, fullgraph=True)
    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Ld2-Ld22/1x16x64 ({n_params/1e6:.1f}M params)")
    print(f"Batch size: {batch_size}")
    print(f"Max tokens: {max_tokens}")
    print(f"Steps: {n_steps} (warmup: {n_warmup})")
    print()

    # A100-80GB peak bf16 TFLOPS
    GPU_TFLOPS = 312e12

    # ------------------------------------------------------------------
    # Test 1: Pure compute with synthetic data
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Test 1: Pure compute (synthetic data, no data loading)")
    print("-" * 60)

    # Create synthetic batch
    max_grid = 512 // 16  # 32x32 grid
    patches = torch.randn(batch_size, max_tokens, 3 * 16 * 16, device=device, dtype=dtype)
    ptype = torch.ones(batch_size, max_tokens, device=device, dtype=torch.bool)
    yidx = torch.randint(0, max_grid, (batch_size, max_tokens), device=device)
    xidx = torch.randint(0, max_grid, (batch_size, max_tokens), device=device)
    # Required by decode()
    original_height = torch.full((batch_size,), 512, device=device, dtype=torch.long)
    original_width = torch.full((batch_size,), 512, device=device, dtype=torch.long)

    synthetic_batch = {
        "patches": patches,
        "ptype": ptype,
        "yidx": yidx,
        "xidx": xidx,
        "original_height": original_height,
        "original_width": original_width,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

    # Warmup
    print(f"Warming up ({n_warmup} steps)...")
    for _ in range(n_warmup):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(synthetic_batch)
        diff = out["patches"] - synthetic_batch["patches"]
        loss = diff.pow(2).mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Timed run
    print(f"Running {n_steps} steps...")
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    for i in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(synthetic_batch)
        diff = out["patches"] - synthetic_batch["patches"]
        loss = diff.pow(2).mean()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t_compute = time.perf_counter() - t_start

    tokens_per_step = batch_size * max_tokens
    flops_per_step = 6 * n_params * tokens_per_step
    flops_per_sec = (flops_per_step * n_steps) / t_compute
    mfu_compute = flops_per_sec / GPU_TFLOPS * 100

    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nResults (pure compute):")
    print(f"  Time: {t_compute:.2f}s ({t_compute/n_steps*1000:.1f}ms/step)")
    print(f"  Throughput: {n_steps * batch_size / t_compute:.1f} samples/sec")
    print(f"  MFU: {mfu_compute:.1f}%")
    print(f"  Peak memory: {mem_gb:.1f} GB")

    # ------------------------------------------------------------------
    # Test 2: Data loading throughput only
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print("Test 2: Data loading throughput (no compute)")
    print("-" * 60)

    data_source = "hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0049}.tar"
    pp_string = (
        f"random_resized_crop(512)|"
        f"flip|"
        f"to_tensor|"
        f"normalize(minus_one_to_one)|"
        f"patchify(512, 16, {max_tokens})"
    )

    print(f"Loading data from: {data_source}")
    loader = create_dataloader(
        source=data_source,
        pp=pp_string,
        batch_size=batch_size,
        num_workers=4,
        seed=42,
    )

    loader_iter = iter(loader)

    # Warmup data loading
    print(f"Warming up data loader ({n_warmup} batches)...")
    for _ in range(n_warmup):
        batch, _ = next(loader_iter)

    # Timed run
    print(f"Loading {n_steps} batches...")
    t_start = time.perf_counter()

    for i in range(n_steps):
        batch, _ = next(loader_iter)

    t_data = time.perf_counter() - t_start

    print(f"\nResults (data loading only):")
    print(f"  Time: {t_data:.2f}s ({t_data/n_steps*1000:.1f}ms/batch)")
    print(f"  Throughput: {n_steps * batch_size / t_data:.1f} samples/sec")

    # ------------------------------------------------------------------
    # Test 3: Full training loop (data + compute)
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print("Test 3: Full training loop (data loading + compute)")
    print("-" * 60)

    # Reset iterator
    loader_iter = iter(loader)

    # Warmup
    print(f"Warming up ({n_warmup} steps)...")
    for _ in range(n_warmup):
        batch, _ = next(loader_iter)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if "patches" in batch:
            batch["patches"] = batch["patches"].to(dtype)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(batch)
        diff = out["patches"] - batch["patches"]
        loss = diff.pow(2).mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Timed run with detailed breakdown
    print(f"Running {n_steps} steps with timing breakdown...")
    torch.cuda.reset_peak_memory_stats()

    t_data_total = 0
    t_move_total = 0
    t_fwd_total = 0
    t_bwd_total = 0
    t_opt_total = 0

    t_start = time.perf_counter()

    for i in range(n_steps):
        # Data loading
        t0 = time.perf_counter()
        batch, _ = next(loader_iter)
        t_data_total += time.perf_counter() - t0

        # Move to device
        t0 = time.perf_counter()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if "patches" in batch:
            batch["patches"] = batch["patches"].to(dtype)
        torch.cuda.synchronize()
        t_move_total += time.perf_counter() - t0

        optimizer.zero_grad(set_to_none=True)

        # Forward
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(batch)
        diff = out["patches"] - batch["patches"]
        loss = diff.pow(2).mean()
        torch.cuda.synchronize()
        t_fwd_total += time.perf_counter() - t0

        # Backward
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd_total += time.perf_counter() - t0

        # Optimizer step
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t_opt_total += time.perf_counter() - t0

    torch.cuda.synchronize()
    t_full = time.perf_counter() - t_start

    flops_per_sec_full = (flops_per_step * n_steps) / t_full
    mfu_full = flops_per_sec_full / GPU_TFLOPS * 100

    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nResults (full training loop):")
    print(f"  Total time: {t_full:.2f}s ({t_full/n_steps*1000:.1f}ms/step)")
    print(f"  Throughput: {n_steps * batch_size / t_full:.1f} samples/sec")
    print(f"  MFU: {mfu_full:.1f}%")
    print(f"  Peak memory: {mem_gb:.1f} GB")
    print()
    print(f"  Breakdown per step:")
    print(f"    Data loading: {t_data_total/n_steps*1000:.1f}ms ({t_data_total/t_full*100:.1f}%)")
    print(f"    Move to GPU:  {t_move_total/n_steps*1000:.1f}ms ({t_move_total/t_full*100:.1f}%)")
    print(f"    Forward:      {t_fwd_total/n_steps*1000:.1f}ms ({t_fwd_total/t_full*100:.1f}%)")
    print(f"    Backward:     {t_bwd_total/n_steps*1000:.1f}ms ({t_bwd_total/t_full*100:.1f}%)")
    print(f"    Optimizer:    {t_opt_total/n_steps*1000:.1f}ms ({t_opt_total/t_full*100:.1f}%)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Pure compute MFU:     {mfu_compute:.1f}%")
    print(f"  Full training MFU:    {mfu_full:.1f}%")
    print(f"  MFU drop:             {mfu_compute - mfu_full:.1f}%")
    print()

    if t_data_total / t_full > 0.2:
        print("  BOTTLENECK: Data loading is >20% of time!")
        print("  Recommendations:")
        print("    - Increase num_workers")
        print("    - Check network/HF streaming speed")
        print("    - Consider local data caching")
    elif t_move_total / t_full > 0.2:
        print("  BOTTLENECK: Data transfer to GPU is >20% of time!")
        print("  Recommendations:")
        print("    - Use pin_memory=True (already enabled)")
        print("    - Consider async data loading")
    else:
        print("  No obvious data bottleneck detected.")
        print("  Low MFU may be due to:")
        print("    - Model architecture (not compute-bound)")
        print("    - CUDA kernel overhead")
        print("    - Memory bandwidth limits")

    return {
        "mfu_compute": mfu_compute,
        "mfu_full": mfu_full,
        "data_pct": t_data_total / t_full * 100,
        "fwd_pct": t_fwd_total / t_full * 100,
        "bwd_pct": t_bwd_total / t_full * 100,
    }


@app.local_entrypoint()
def main(
    batch_size: int = 256,
    max_tokens: int = 256,
    n_steps: int = 50,
    n_warmup: int = 5,
):
    """Run MFU benchmark."""
    result = benchmark_mfu.remote(
        batch_size=batch_size,
        max_tokens=max_tokens,
        n_steps=n_steps,
        n_warmup=n_warmup,
    )
    print(f"\nBenchmark complete!")
    print(f"Results: {result}")
