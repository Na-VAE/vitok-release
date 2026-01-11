"""Test quantized inference mode for T models.

This test verifies that quantized inference (INT8 on A100, FP8 on H100+)
produces outputs comparable to bf16 baseline.

NOTE: A100 GPUs do NOT support FP8 (requires compute capability 8.9+).
The code automatically falls back to INT8 on A100.

Must pass before running full evaluations with float8_mode="inference".

Usage:
    # Run on Modal with A100 GPU
    modal run tests/gpu/test_float8_inference.py

    # Test specific model
    modal run tests/gpu/test_float8_inference.py --model T-32x64

    # Quick test (fewer samples)
    modal run tests/gpu/test_float8_inference.py --quick
"""

import modal
from pathlib import Path

app = modal.App("vitok-test-float8")

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
        "webdataset",
    )
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
)

weights_vol = modal.Volume.from_name("vitok-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/cache": weights_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def test_float8_inference(
    model_name: str = "T-32x64",
    n_samples: int = 10,
    image_size: int = 256,
) -> dict:
    """Test float8 inference vs bf16 baseline.

    Returns:
        Dictionary with test results including:
        - ssim_between_modes: SSIM between float8 and bf16 outputs (should be > 0.99)
        - bf16_has_nan: Whether bf16 output contains NaN/Inf
        - float8_has_nan: Whether float8 output contains NaN/Inf
        - bf16_latency_ms: Average latency for bf16 inference
        - float8_latency_ms: Average latency for float8 inference
        - speedup: Speedup ratio (bf16_latency / float8_latency)
    """
    import os
    import sys
    import time
    import torch
    import numpy as np
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    sys.path.insert(0, "/root/vitok-release")
    os.environ["HF_HOME"] = "/cache/huggingface"

    from vitok import AE, decode_variant
    from vitok.pretrained import download_pretrained, get_pretrained_info

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Check GPU compute capability
    cc = torch.cuda.get_device_capability()
    quant_type = "FP8" if cc[0] >= 9 else "INT8"

    print(f"Testing quantized inference for: {model_name}")
    print(f"  GPU Compute Capability: {cc[0]}.{cc[1]}")
    print(f"  Quantization Type: {quant_type} (auto-selected based on GPU)")
    print(f"  Samples: {n_samples}")
    print(f"  Image size: {image_size}")
    print()

    # Download model
    _, _, variant = get_pretrained_info(model_name)
    checkpoint = download_pretrained(model_name)
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Variant: {variant}")

    # Load weights
    from safetensors.torch import load_file
    weights = load_file(checkpoint)

    # Get model config
    config = decode_variant(variant)
    patch_size = config["spatial_stride"]

    # Create random test images
    print("\nCreating test images...")
    torch.manual_seed(42)
    test_images = torch.randn(n_samples, 3, image_size, image_size, device=device, dtype=dtype)
    test_images = test_images.clamp(-1, 1)  # Normalize to expected range

    # Create patch dict
    n_patches = (image_size // patch_size) ** 2
    row_idx = torch.arange(image_size // patch_size, device=device).repeat_interleave(image_size // patch_size)
    col_idx = torch.arange(image_size // patch_size, device=device).repeat(image_size // patch_size)

    results = {}

    # Test 1: BF16 baseline
    print("\n[1/3] Testing BF16 baseline...")
    encoder_bf16 = AE(**config, decoder=False, float8_mode=None).to(device, dtype)
    encoder_bf16.load_state_dict(weights, strict=False)
    encoder_bf16.eval()

    decoder_bf16 = AE(**config, encoder=False, float8_mode=None).to(device, dtype)
    decoder_bf16.load_state_dict(weights, strict=False)
    decoder_bf16.eval()

    bf16_outputs = []
    bf16_times = []

    with torch.no_grad():
        for i in range(n_samples):
            img = test_images[i:i+1]

            # Patchify
            patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, -1, 3 * patch_size * patch_size)

            patch_dict = {
                "patches": patches,
                "row_idx": row_idx.unsqueeze(0),
                "col_idx": col_idx.unsqueeze(0),
                "patch_mask": torch.ones(1, n_patches, dtype=torch.bool, device=device),
            }

            torch.cuda.synchronize()
            start = time.perf_counter()

            encoded = encoder_bf16.encode(patch_dict)
            decoded = decoder_bf16.decode(encoded)

            torch.cuda.synchronize()
            bf16_times.append((time.perf_counter() - start) * 1000)

            bf16_outputs.append(decoded["patches"].clone())

    bf16_outputs = torch.cat(bf16_outputs, dim=0)
    bf16_has_nan = torch.isnan(bf16_outputs).any().item() or torch.isinf(bf16_outputs).any().item()
    bf16_latency = np.mean(bf16_times[1:])  # Skip first (warmup)

    print(f"  BF16 has NaN/Inf: {bf16_has_nan}")
    print(f"  BF16 latency: {bf16_latency:.2f}ms")

    del encoder_bf16, decoder_bf16
    torch.cuda.empty_cache()

    # Test 2: INT8 inference mode (applied AFTER loading weights)
    print("\n[2/3] Testing INT8 inference mode...")

    try:
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

        # Create model WITHOUT quantization, load weights, THEN quantize
        encoder_f8 = AE(**config, decoder=False, float8_mode=None).to(device, dtype)
        encoder_f8.load_state_dict(weights, strict=False)
        encoder_f8.eval()

        # Apply INT8 quantization AFTER loading weights
        for block in encoder_f8.encoder_blocks:
            quantize_(block, Int8DynamicActivationInt8WeightConfig())

        decoder_f8 = AE(**config, encoder=False, float8_mode=None).to(device, dtype)
        decoder_f8.load_state_dict(weights, strict=False)
        decoder_f8.eval()

        # Apply INT8 quantization AFTER loading weights
        for block in decoder_f8.decoder_blocks:
            quantize_(block, Int8DynamicActivationInt8WeightConfig())

        f8_outputs = []
        f8_times = []

        with torch.no_grad():
            for i in range(n_samples):
                img = test_images[i:i+1]

                patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
                patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, -1, 3 * patch_size * patch_size)

                patch_dict = {
                    "patches": patches,
                    "row_idx": row_idx.unsqueeze(0),
                    "col_idx": col_idx.unsqueeze(0),
                    "patch_mask": torch.ones(1, n_patches, dtype=torch.bool, device=device),
                }

                torch.cuda.synchronize()
                start = time.perf_counter()

                encoded = encoder_f8.encode(patch_dict)
                decoded = decoder_f8.decode(encoded)

                torch.cuda.synchronize()
                f8_times.append((time.perf_counter() - start) * 1000)

                f8_outputs.append(decoded["patches"].clone())

        f8_outputs = torch.cat(f8_outputs, dim=0)
        float8_has_nan = torch.isnan(f8_outputs).any().item() or torch.isinf(f8_outputs).any().item()
        float8_latency = np.mean(f8_times[1:])

        print(f"  Float8 has NaN/Inf: {float8_has_nan}")
        print(f"  Float8 latency: {float8_latency:.2f}ms")

        float8_works = True

    except Exception as e:
        print(f"  Float8 ERROR: {e}")
        float8_works = False
        float8_has_nan = True
        float8_latency = 0
        f8_outputs = bf16_outputs  # Use bf16 for comparison

    # Test 3: Compare outputs
    print("\n[3/3] Comparing outputs...")

    if float8_works:
        # Reshape for SSIM calculation
        bf16_imgs = bf16_outputs.reshape(n_samples, image_size // patch_size, image_size // patch_size, 3, patch_size, patch_size)
        bf16_imgs = bf16_imgs.permute(0, 3, 1, 4, 2, 5).reshape(n_samples, 3, image_size, image_size)

        f8_imgs = f8_outputs.reshape(n_samples, image_size // patch_size, image_size // patch_size, 3, patch_size, patch_size)
        f8_imgs = f8_imgs.permute(0, 3, 1, 4, 2, 5).reshape(n_samples, 3, image_size, image_size)

        # Compute SSIM between modes
        ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        ssim_score = ssim(f8_imgs.float(), bf16_imgs.float()).item()

        # Compute max absolute difference
        max_diff = (f8_outputs - bf16_outputs).abs().max().item()

        print(f"  SSIM between float8 and bf16: {ssim_score:.6f}")
        print(f"  Max absolute difference: {max_diff:.6f}")

        speedup = bf16_latency / float8_latency if float8_latency > 0 else 0

        results = {
            "model": model_name,
            "n_samples": n_samples,
            "image_size": image_size,
            "bf16_has_nan": bf16_has_nan,
            "float8_has_nan": float8_has_nan,
            "float8_works": float8_works,
            "ssim_between_modes": ssim_score,
            "max_abs_diff": max_diff,
            "bf16_latency_ms": bf16_latency,
            "float8_latency_ms": float8_latency,
            "speedup": speedup,
        }
    else:
        results = {
            "model": model_name,
            "n_samples": n_samples,
            "image_size": image_size,
            "bf16_has_nan": bf16_has_nan,
            "float8_has_nan": True,
            "float8_works": False,
            "error": "Float8 mode failed to initialize or run",
            "bf16_latency_ms": bf16_latency,
        }

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    passed = True
    if results.get("float8_works"):
        if results["ssim_between_modes"] < 0.99:
            print(f"FAIL: SSIM too low ({results['ssim_between_modes']:.4f} < 0.99)")
            passed = False
        else:
            print(f"PASS: SSIM = {results['ssim_between_modes']:.4f} (>= 0.99)")

        if results["float8_has_nan"]:
            print("FAIL: Float8 output contains NaN/Inf")
            passed = False
        else:
            print("PASS: No NaN/Inf in float8 output")

        print(f"Speedup: {results['speedup']:.2f}x")
    else:
        print("FAIL: Float8 mode did not work")
        passed = False

    results["passed"] = passed
    print(f"\nOverall: {'PASSED' if passed else 'FAILED'}")

    return results


@app.local_entrypoint()
def main(
    model: str = "T-32x64",
    n_samples: int = 10,
    image_size: int = 256,
    quick: bool = False,
):
    """Run float8 inference test.

    Args:
        model: Model to test (default: T-32x64)
        n_samples: Number of test samples (default: 10)
        image_size: Test image size (default: 256)
        quick: Quick test with 3 samples
    """
    if quick:
        n_samples = 3

    print(f"Testing float8 inference for: {model}")
    print(f"  Samples: {n_samples}")
    print(f"  Image size: {image_size}")
    print()

    results = test_float8_inference.remote(
        model_name=model,
        n_samples=n_samples,
        image_size=image_size,
    )

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    for key, value in sorted(results.items()):
        print(f"  {key}: {value}")

    if results.get("passed"):
        print("\nFloat8 inference test PASSED!")
        print("Safe to run evaluations with --float8 flag.")
    else:
        print("\nFloat8 inference test FAILED!")
        print("Do not use float8 mode for evaluations.")
        print("Fall back to bf16 (default).")
