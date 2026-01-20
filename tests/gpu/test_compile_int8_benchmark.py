"""Benchmark torch.compile and quantization for ViTok VAE.

Tests 6 configurations on H100 IN PARALLEL:
1. bf16 baseline (no compile, no quantization)
2. bf16 + torch.compile
3. FP8 quantization (no compile) - H100 native
4. FP8 + torch.compile
5. INT8 weight-only (no compile)
6. INT8 weight-only + torch.compile

Reports throughput, latency, memory, and quality metrics.

Usage:
    modal run tests/gpu/test_compile_int8_benchmark.py
    modal run tests/gpu/test_compile_int8_benchmark.py --model 5B-64
"""

import modal
from pathlib import Path

app = modal.App("vitok-quant-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.23.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "einops",
        "torchao>=0.5.0",
        "torchmetrics>=1.0.0",
        "tabulate",
        "webdataset>=0.2.90",
    )
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
)

weights_vol = modal.Volume.from_name("vitok-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/cache": weights_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def run_single_config(
    model_name: str,
    config_name: str,
    image_size: int = 256,
    warmup: int = 5,
    iterations: int = 20,
) -> dict:
    """Run a single quantization config benchmark."""
    import os
    import sys
    import time
    import torch
    import numpy as np

    sys.path.insert(0, "/root/vitok-release")
    os.environ["HF_HOME"] = "/cache/huggingface"

    from vitok import AE, decode_variant
    from vitok.pretrained import download_pretrained, get_pretrained_info
    from safetensors.torch import load_file

    device = torch.device("cuda")
    dtype = torch.bfloat16

    gpu_name = torch.cuda.get_device_name(device)
    cc = torch.cuda.get_device_capability()

    print(f"[{config_name}] Starting on {gpu_name}")

    # Download and load model
    _, _, variant = get_pretrained_info(model_name)
    weights_paths = download_pretrained(model_name)

    weights = {}
    for p in weights_paths:
        weights.update(load_file(p))

    config = decode_variant(variant)
    patch_size = config["spatial_stride"]

    # Create test data
    torch.manual_seed(42)
    n_patches = (image_size // patch_size) ** 2
    grid_size = image_size // patch_size
    row_idx = torch.arange(grid_size, device=device).repeat_interleave(grid_size)
    col_idx = torch.arange(grid_size, device=device).repeat(grid_size)

    test_data = []
    for _ in range(max(warmup, iterations)):
        patches = torch.randn(1, n_patches, 3 * patch_size * patch_size, device=device, dtype=dtype)
        test_data.append({
            "patches": patches,
            "row_idx": row_idx.unsqueeze(0),
            "col_idx": col_idx.unsqueeze(0),
            "patch_mask": torch.ones(1, n_patches, dtype=torch.bool, device=device),
        })

    # Create encoder/decoder based on config
    encoder = AE(**config, decoder=False, float8_mode=None).to(device, dtype)
    encoder.load_state_dict(weights, strict=False)
    encoder.eval()
    decoder = AE(**config, encoder=False, float8_mode=None).to(device, dtype)
    decoder.load_state_dict(weights, strict=False)
    decoder.eval()

    # Apply quantization based on config
    if "fp8" in config_name:
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
        print(f"  [{config_name}] Applying FP8 quantization...")
        for block in encoder.encoder_blocks:
            quantize_(block, Float8DynamicActivationFloat8WeightConfig())
        for block in decoder.decoder_blocks:
            quantize_(block, Float8DynamicActivationFloat8WeightConfig())

    elif "int8" in config_name:
        from torchao.quantization import quantize_, int8_weight_only
        print(f"  [{config_name}] Applying INT8 weight-only quantization...")
        for block in encoder.encoder_blocks:
            quantize_(block, int8_weight_only())
        for block in decoder.decoder_blocks:
            quantize_(block, int8_weight_only())

    # Apply compile if needed
    if "compile" in config_name:
        print(f"  [{config_name}] Compiling encoder/decoder...")
        encoder = torch.compile(encoder, fullgraph=True, mode="reduce-overhead")
        decoder = torch.compile(decoder, fullgraph=True, mode="reduce-overhead")

    # Warmup
    print(f"  [{config_name}] Warmup ({warmup} iterations)...")
    with torch.no_grad():
        for i in range(warmup):
            encoded = encoder.encode(test_data[i % len(test_data)])
            _ = decoder.decode(encoded)
            torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    outputs = []
    print(f"  [{config_name}] Benchmarking ({iterations} iterations)...")
    with torch.no_grad():
        for i in range(iterations):
            batch = test_data[i % len(test_data)]
            torch.cuda.synchronize()
            start = time.perf_counter()

            encoded = encoder.encode(batch)
            decoded = decoder.decode(encoded)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

            if i < 5:
                outputs.append(decoded["patches"].cpu().clone())

    times_arr = np.array(times)
    result = {
        "config": config_name,
        "mean_latency_ms": float(np.mean(times_arr)),
        "std_latency_ms": float(np.std(times_arr)),
        "throughput_img_per_sec": 1000.0 / np.mean(times_arr),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "outputs": outputs,  # For SSIM comparison
    }

    print(f"  [{config_name}] Done: {result['mean_latency_ms']:.2f}ms, {result['throughput_img_per_sec']:.1f} img/s")
    return result


@app.function(
    image=image,
    gpu="H100",
    volumes={"/cache": weights_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def benchmark_parallel(
    model_name: str = "350M-64",
    image_size: int = 256,
    warmup: int = 5,
    iterations: int = 20,
) -> dict:
    """Run all configs in parallel and aggregate results."""
    import os
    import sys
    sys.path.insert(0, "/root/vitok-release")
    os.environ["HF_HOME"] = "/cache/huggingface"

    import torch
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from tabulate import tabulate

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    cc = torch.cuda.get_device_capability()
    supports_fp8 = cc[0] >= 9

    print("=" * 70)
    print("PARALLEL QUANTIZATION BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"GPU: {gpu_name}")
    print(f"FP8 Support: {supports_fp8}")
    print()

    # Define configs to run
    configs = ["bf16_baseline", "bf16_compile"]
    if supports_fp8:
        configs.extend(["fp8_only", "fp8_compile"])
    configs.extend(["int8_weight_only", "int8_weight_compile"])

    print(f"Running {len(configs)} configs in parallel: {configs}")
    print()

    # Launch all configs in parallel using starmap
    results = list(run_single_config.starmap([
        (model_name, cfg, image_size, warmup, iterations)
        for cfg in configs
    ]))

    # Find baseline for comparison
    baseline_result = next(r for r in results if r["config"] == "bf16_baseline")
    baseline_latency = baseline_result["mean_latency_ms"]
    baseline_outputs = baseline_result["outputs"]

    # Compute SSIM for each config
    print("\n" + "=" * 70)
    print("QUALITY COMPARISON (SSIM vs baseline)")
    print("=" * 70)

    from vitok import decode_variant
    from vitok.pretrained import get_pretrained_info
    _, _, variant = get_pretrained_info(model_name)
    config = decode_variant(variant)
    patch_size = config["spatial_stride"]
    grid_size = image_size // patch_size

    def patches_to_images(patches_list, patch_size, grid_size):
        patches = torch.cat([p.to(device) for p in patches_list], dim=0).float()
        B, N, D = patches.shape
        patches = patches.view(B, N, 3, patch_size, patch_size)
        patches = patches.view(B, grid_size, grid_size, 3, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(B, 3, grid_size * patch_size, grid_size * patch_size)

    ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    baseline_images = patches_to_images(baseline_outputs, patch_size, grid_size)

    for r in results:
        if r["config"] == "bf16_baseline":
            continue
        try:
            out_images = patches_to_images(r["outputs"], patch_size, grid_size)
            n = min(baseline_images.shape[0], out_images.shape[0])
            score = ssim(out_images[:n], baseline_images[:n]).item()
            r["ssim_vs_baseline"] = score
            print(f"  {r['config']}: SSIM = {score:.6f}")
        except Exception as e:
            print(f"  {r['config']}: Error - {e}")

    # Results table
    print("\n" + "=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)

    headers = ["Config", "Latency (ms)", "Std", "Throughput", "Memory (GB)", "Speedup"]
    rows = []

    for r in results:
        latency = r["mean_latency_ms"]
        speedup = baseline_latency / latency if latency > 0 else 0
        rows.append([
            r["config"],
            f"{latency:.2f}",
            f"{r['std_latency_ms']:.2f}",
            f"{r['throughput_img_per_sec']:.1f} img/s",
            f"{r['peak_memory_gb']:.2f}",
            f"{speedup:.2f}x",
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    best = min(results, key=lambda x: x["mean_latency_ms"])
    print(f"  Best config: {best['config']}")
    print(f"  Latency: {best['mean_latency_ms']:.2f}ms")
    print(f"  Throughput: {best['throughput_img_per_sec']:.1f} img/s")

    # Clean up outputs before returning (can't serialize tensors)
    for r in results:
        del r["outputs"]

    return {"model": model_name, "gpu": gpu_name, "results": results}


@app.local_entrypoint()
def main(
    model: str = "350M-64",
    image_size: int = 256,
    warmup: int = 5,
    iterations: int = 20,
):
    """Run quantization benchmark with parallel config execution."""
    print(f"Running parallel benchmark: model={model}")
    print()

    results = benchmark_parallel.remote(
        model_name=model,
        image_size=image_size,
        warmup=warmup,
        iterations=iterations,
    )

    print("\nDone!")
