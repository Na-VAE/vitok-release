#!/usr/bin/env python
"""Benchmark full training loop including perceptual losses.

Usage:
    modal run scripts/modal_train_vae.py --sync-only  # First sync code
    modal run modal_tests/benchmark_full_train.py
"""

import modal
from pathlib import Path

app = modal.App("vitok-full-train-benchmark")

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
        "torchmetrics",
        "transformers",
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
def benchmark_full_train(
    batch_size: int = 64,
    max_tokens: int = 256,
    n_steps: int = 30,
    n_warmup: int = 5,
    tile_size: int = 256,
    n_tiles: int = 1,
):
    """Benchmark full training including perceptual losses."""
    import sys
    import os
    import time

    os.environ["PYTHONPATH"] = "/code/vitok-release:/code/dino_perceptual"
    sys.path.insert(0, "/code/vitok-release")
    sys.path.insert(0, "/code/dino_perceptual")
    os.chdir("/code/vitok-release")

    import torch
    from vitok import AE, decode_variant
    from vitok.pp.io import postprocess_images
    from vitok.pp import sample_tiles
    from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
    from dino_perceptual import DINOPerceptual

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 60)
    print("Full Training Benchmark (with perceptual losses)")
    print("=" * 60)

    # Create model
    model = AE(**decode_variant("Ld2-Ld22/1x16x64"))
    model.to(device=device, dtype=dtype)
    model = torch.compile(model, fullgraph=True)
    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AE Model: Ld2-Ld22/1x16x64 ({n_params/1e6:.1f}M params)")

    # Perceptual losses
    print("Loading DINO perceptual model (ViT-S)...")
    dino_loss_fn = DINOPerceptual(model_size='S', target_size=tile_size)
    dino_loss_fn.to(device)
    dino_params = sum(p.numel() for p in dino_loss_fn.parameters())
    print(f"DINO model: {dino_params/1e6:.1f}M params")

    print(f"Batch size: {batch_size}")
    print(f"Max tokens: {max_tokens}")
    print(f"Steps: {n_steps} (warmup: {n_warmup})")
    print()

    # A100-80GB peak bf16 TFLOPS
    GPU_TFLOPS = 312e12

    # Create synthetic batch
    max_grid = 512 // 16  # 32x32 grid
    patches = torch.randn(batch_size, max_tokens, 3 * 16 * 16, device=device, dtype=dtype)
    ptype = torch.ones(batch_size, max_tokens, device=device, dtype=torch.bool)
    yidx = torch.arange(max_tokens, device=device).unsqueeze(0).expand(batch_size, -1) // 16
    xidx = torch.arange(max_tokens, device=device).unsqueeze(0).expand(batch_size, -1) % 16
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

    # Loss weights (same as training script defaults)
    charbonnier_weight = 1.0
    charbonnier_eps = 1e-3
    ssim_weight = 0.1
    dino_weight = 250.0

    def train_step(batch):
        """Full training step with all losses."""
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=dtype):
            decode_dict = model(batch)

        ptype = batch["ptype"]
        diff = decode_dict["patches"] - batch["patches"]

        # Charbonnier loss
        diff_f32 = diff.float()
        charb_per_token = (diff_f32.pow(2) + charbonnier_eps**2).sqrt().mean(dim=2)
        charb_per_token = charb_per_token * ptype.float()
        actual_tokens = ptype.sum(dim=1).clamp_min(1).float()
        charb_loss = (charb_per_token.sum(dim=1) / actual_tokens).mean()

        loss = charbonnier_weight * charb_loss

        # Perceptual losses on tiles
        with torch.no_grad():
            recon_images = postprocess_images(
                decode_dict, output_format="minus_one_to_one",
                current_format="minus_one_to_one", unpack=False,
                patch=16, max_grid_size=max_grid,
            )
            ref_images = postprocess_images(
                batch, output_format="minus_one_to_one",
                current_format="minus_one_to_one", unpack=False,
                patch=16, max_grid_size=max_grid,
            )

        # Sample tiles
        orig_h = batch['original_height']
        orig_w = batch['original_width']
        tiles_ref, tile_indices = sample_tiles(
            ref_images, orig_h, orig_w,
            n_tiles=n_tiles, tile_size=(tile_size, tile_size)
        )
        tiles_pred, _ = sample_tiles(
            recon_images, orig_h, orig_w,
            n_tiles=n_tiles, tile_size=(tile_size, tile_size),
            indices=tile_indices
        )

        B = tiles_ref.shape[0]
        tiles_ref = tiles_ref.reshape(B * n_tiles, 3, tile_size, tile_size)
        tiles_pred = tiles_pred.reshape(B * n_tiles, 3, tile_size, tile_size)

        with torch.autocast(device_type="cuda", dtype=dtype):
            # SSIM
            ssim_val = SSIM(preds=tiles_pred, target=tiles_ref, data_range=2.0)
            ssim_loss = (1.0 - ssim_val).float()
            loss = loss + ssim_weight * ssim_loss

            # DINO perceptual
            dino_loss = dino_loss_fn(tiles_pred, tiles_ref).mean().float()
            loss = loss + dino_weight * dino_loss

        loss.backward()
        optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Test 1: AE forward/backward only (no perceptual losses)
    # ------------------------------------------------------------------
    print("-" * 60)
    print("Test 1: AE only (no perceptual losses)")
    print("-" * 60)

    def ae_only_step(batch):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(batch)
        diff = out["patches"] - batch["patches"]
        loss = diff.pow(2).mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    # Warmup
    for _ in range(n_warmup):
        ae_only_step(synthetic_batch)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    for _ in range(n_steps):
        ae_only_step(synthetic_batch)
    torch.cuda.synchronize()
    t_ae = time.perf_counter() - t_start

    tokens_per_step = batch_size * max_tokens
    flops_per_step = 6 * n_params * tokens_per_step
    mfu_ae = (flops_per_step * n_steps / t_ae) / GPU_TFLOPS * 100
    mem_ae = torch.cuda.max_memory_allocated() / 1e9

    print(f"  Time: {t_ae:.2f}s ({t_ae/n_steps*1000:.1f}ms/step)")
    print(f"  Throughput: {n_steps * batch_size / t_ae:.1f} samples/sec")
    print(f"  MFU (AE only): {mfu_ae:.1f}%")
    print(f"  Memory: {mem_ae:.1f} GB")

    # ------------------------------------------------------------------
    # Test 2: Full training with perceptual losses
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print("Test 2: Full training (AE + SSIM + DINO)")
    print("-" * 60)

    # Warmup
    for _ in range(n_warmup):
        train_step(synthetic_batch)
    torch.cuda.synchronize()

    # Timed run with breakdown
    torch.cuda.reset_peak_memory_stats()

    t_ae_total = 0
    t_postprocess_total = 0
    t_tile_total = 0
    t_ssim_total = 0
    t_dino_total = 0
    t_backward_total = 0

    t_start = time.perf_counter()

    for i in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        # AE forward
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=dtype):
            decode_dict = model(synthetic_batch)
        torch.cuda.synchronize()
        t_ae_total += time.perf_counter() - t0

        ptype = synthetic_batch["ptype"]
        diff = decode_dict["patches"] - synthetic_batch["patches"]
        diff_f32 = diff.float()
        charb_per_token = (diff_f32.pow(2) + charbonnier_eps**2).sqrt().mean(dim=2)
        charb_per_token = charb_per_token * ptype.float()
        actual_tokens = ptype.sum(dim=1).clamp_min(1).float()
        charb_loss = (charb_per_token.sum(dim=1) / actual_tokens).mean()
        loss = charbonnier_weight * charb_loss

        # Postprocess
        t0 = time.perf_counter()
        with torch.no_grad():
            recon_images = postprocess_images(
                decode_dict, output_format="minus_one_to_one",
                current_format="minus_one_to_one", unpack=False,
                patch=16, max_grid_size=max_grid,
            )
            ref_images = postprocess_images(
                synthetic_batch, output_format="minus_one_to_one",
                current_format="minus_one_to_one", unpack=False,
                patch=16, max_grid_size=max_grid,
            )
        torch.cuda.synchronize()
        t_postprocess_total += time.perf_counter() - t0

        # Sample tiles
        t0 = time.perf_counter()
        orig_h = synthetic_batch['original_height']
        orig_w = synthetic_batch['original_width']
        tiles_ref, tile_indices = sample_tiles(
            ref_images, orig_h, orig_w,
            n_tiles=n_tiles, tile_size=(tile_size, tile_size)
        )
        tiles_pred, _ = sample_tiles(
            recon_images, orig_h, orig_w,
            n_tiles=n_tiles, tile_size=(tile_size, tile_size),
            indices=tile_indices
        )
        B = tiles_ref.shape[0]
        tiles_ref = tiles_ref.reshape(B * n_tiles, 3, tile_size, tile_size)
        tiles_pred = tiles_pred.reshape(B * n_tiles, 3, tile_size, tile_size)
        torch.cuda.synchronize()
        t_tile_total += time.perf_counter() - t0

        with torch.autocast(device_type="cuda", dtype=dtype):
            # SSIM
            t0 = time.perf_counter()
            ssim_val = SSIM(preds=tiles_pred, target=tiles_ref, data_range=2.0)
            ssim_loss = (1.0 - ssim_val).float()
            loss = loss + ssim_weight * ssim_loss
            torch.cuda.synchronize()
            t_ssim_total += time.perf_counter() - t0

            # DINO
            t0 = time.perf_counter()
            dino_loss = dino_loss_fn(tiles_pred, tiles_ref).mean().float()
            loss = loss + dino_weight * dino_loss
            torch.cuda.synchronize()
            t_dino_total += time.perf_counter() - t0

        # Backward + optimizer
        t0 = time.perf_counter()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t_backward_total += time.perf_counter() - t0

    torch.cuda.synchronize()
    t_full = time.perf_counter() - t_start

    mem_full = torch.cuda.max_memory_allocated() / 1e9
    mfu_full = (flops_per_step * n_steps / t_full) / GPU_TFLOPS * 100

    print(f"\nResults:")
    print(f"  Total time: {t_full:.2f}s ({t_full/n_steps*1000:.1f}ms/step)")
    print(f"  Throughput: {n_steps * batch_size / t_full:.1f} samples/sec")
    print(f"  MFU (AE params only): {mfu_full:.1f}%")
    print(f"  Memory: {mem_full:.1f} GB")
    print()
    print(f"  Breakdown per step:")
    print(f"    AE forward:     {t_ae_total/n_steps*1000:6.1f}ms ({t_ae_total/t_full*100:5.1f}%)")
    print(f"    Postprocess:    {t_postprocess_total/n_steps*1000:6.1f}ms ({t_postprocess_total/t_full*100:5.1f}%)")
    print(f"    Tile sampling:  {t_tile_total/n_steps*1000:6.1f}ms ({t_tile_total/t_full*100:5.1f}%)")
    print(f"    SSIM:           {t_ssim_total/n_steps*1000:6.1f}ms ({t_ssim_total/t_full*100:5.1f}%)")
    print(f"    DINO forward:   {t_dino_total/n_steps*1000:6.1f}ms ({t_dino_total/t_full*100:5.1f}%)")
    print(f"    Backward+Opt:   {t_backward_total/n_steps*1000:6.1f}ms ({t_backward_total/t_full*100:5.1f}%)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  AE-only MFU:        {mfu_ae:.1f}%")
    print(f"  Full training MFU:  {mfu_full:.1f}%")
    print(f"  Slowdown factor:    {t_full/t_ae:.2f}x")
    print()

    overhead_pct = (t_postprocess_total + t_tile_total + t_ssim_total + t_dino_total) / t_full * 100
    print(f"  Perceptual loss overhead: {overhead_pct:.1f}% of step time")

    if t_dino_total / t_full > 0.2:
        print("  BOTTLENECK: DINO forward pass is >20% of time")
    if t_postprocess_total / t_full > 0.1:
        print("  BOTTLENECK: Postprocess (unpacking patches) is >10% of time")

    return {
        "mfu_ae_only": mfu_ae,
        "mfu_full": mfu_full,
        "slowdown": t_full / t_ae,
        "dino_pct": t_dino_total / t_full * 100,
        "postprocess_pct": t_postprocess_total / t_full * 100,
        "ssim_pct": t_ssim_total / t_full * 100,
    }


@app.local_entrypoint()
def main(
    batch_size: int = 64,
    max_tokens: int = 256,
    n_steps: int = 30,
    n_warmup: int = 5,
):
    """Run full training benchmark."""
    result = benchmark_full_train.remote(
        batch_size=batch_size,
        max_tokens=max_tokens,
        n_steps=n_steps,
        n_warmup=n_warmup,
    )
    print(f"\nBenchmark complete!")
    print(f"Results: {result}")
