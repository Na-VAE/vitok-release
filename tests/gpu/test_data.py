#!/usr/bin/env python
"""Quick Modal test for data loading pipeline.

Tests HuggingFace streaming without full training setup.

Usage:
    modal run tests/gpu/test_data.py
    modal run tests/gpu/test_data.py --source "hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0004}.tar"
"""

import modal

app = modal.App("vitok-test-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install(
        "torch>=2.5.0",
        "torchvision",
        "numpy",
        "pillow",
        "webdataset",
        "huggingface_hub",
    )
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
)


@app.function(
    image=image,
    gpu="T4",  # Cheapest GPU
    timeout=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_data_loading(source: str, num_batches: int = 5, batch_size: int = 4):
    """Test data loading from HuggingFace."""
    import sys
    import time

    sys.path.insert(0, "/root/vitok-release")

    print(f"Testing data loading from: {source}")
    print(f"Batch size: {batch_size}, Num batches: {num_batches}")
    print("=" * 60)

    # Test 1: URL generation
    print("\n[1] Testing URL generation...")
    t0 = time.perf_counter()

    from vitok.data import _get_hf_shard_urls

    try:
        urls = _get_hf_shard_urls(source)
        print(f"    Generated {len(urls)} URLs in {time.perf_counter() - t0:.2f}s")
        print(f"    First URL: {urls[0][:80]}...")
        if len(urls) > 1:
            print(f"    Last URL:  {urls[-1][:80]}...")
    except Exception as e:
        print(f"    FAILED: {e}")
        return {"success": False, "error": str(e)}

    # Test 2: Create dataloader
    print("\n[2] Testing dataloader creation...")
    t0 = time.perf_counter()

    from vitok.data import create_dataloader

    pp_string = "random_resized_crop(256)|flip|to_tensor|normalize(minus_one_to_one)|patchify(256, 16, 64)"

    try:
        loader = create_dataloader(
            source=source,
            pp=pp_string,
            batch_size=batch_size,
            num_workers=2,
            seed=42,
            return_labels=True,
        )
        print(f"    Dataloader created in {time.perf_counter() - t0:.2f}s")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}

    # Test 3: Fetch batches
    print(f"\n[3] Testing batch fetching ({num_batches} batches)...")
    loader_iter = iter(loader)

    batch_times = []
    for i in range(num_batches):
        t0 = time.perf_counter()
        try:
            batch, labels = next(loader_iter)
            elapsed = time.perf_counter() - t0
            batch_times.append(elapsed)

            if i == 0:
                print(f"    Batch {i+1}: {elapsed:.2f}s (first batch, includes warmup)")
                print(f"    Batch keys: {list(batch.keys())}")
                if "patches" in batch:
                    print(f"    patches shape: {batch['patches'].shape}")
                if "ptype" in batch:
                    print(f"    ptype shape: {batch['ptype'].shape}")
            else:
                print(f"    Batch {i+1}: {elapsed:.2f}s")
        except StopIteration:
            print(f"    Batch {i+1}: Iterator exhausted!")
            break
        except Exception as e:
            print(f"    Batch {i+1}: FAILED - {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_time = sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else batch_times[0]
    print(f"Average batch time (excluding first): {avg_time:.2f}s")
    print(f"Throughput: {batch_size / avg_time:.1f} samples/sec")
    print("SUCCESS!")

    return {
        "success": True,
        "num_urls": len(urls),
        "batch_times": batch_times,
        "avg_time": avg_time,
        "throughput": batch_size / avg_time,
    }


@app.local_entrypoint()
def main(
    source: str = "hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0004}.tar",
    num_batches: int = 5,
    batch_size: int = 4,
):
    """Test data loading on Modal."""
    print("=" * 60)
    print("ViTok Data Loading Test")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"GPU: T4 (cheapest)")
    print()

    result = test_data_loading.remote(
        source=source,
        num_batches=num_batches,
        batch_size=batch_size,
    )

    if result["success"]:
        print(f"\nTest passed! Throughput: {result['throughput']:.1f} samples/sec")
    else:
        print(f"\nTest failed: {result['error']}")
