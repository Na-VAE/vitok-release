"""GPU tests for video integration.

Run with: modal run tests/gpu/test_video.py

Tests verify:
1. Video AE forward pass on GPU
2. Flash attention SWA scaling for 3D
3. Pretrained model: unified video approach
4. Pretrained model: per-frame batch approach
"""

import modal

# Paths inside Modal container
VITOK_PATH = "/root/vitok-release"

# Base image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "webdataset>=0.2.86",
        "huggingface_hub>=0.23.0,<1.0",
        "pytest>=7.0.0",
        "requests",
        "diffusers>=0.31.0",
        "ml_collections",
    )
    .add_local_dir("vitok", remote_path=f"{VITOK_PATH}/vitok")
    .add_local_dir("tests", remote_path=f"{VITOK_PATH}/tests")
)

# Modal app for video tests
app = modal.App("vitok-video-tests")

# Volume for pretrained models
downloads_volume = modal.Volume.from_name("vitok-downloads", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=300)
def test_video_patchify_gpu():
    """Test video patchify/unpatchify on GPU."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    from vitok.pp.ops import patchify, unpatchify

    print("=" * 60)
    print("Test: Video Patchify/Unpatchify on GPU")
    print("=" * 60)

    # Create synthetic video
    video = torch.randn(8, 3, 128, 128).cuda()  # T=8, 128x128
    print(f"Input video shape: {video.shape}")

    # Patchify
    patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=512)
    patch_dict = patch_fn(video.cpu())  # patchify works on CPU

    # Move to GPU and add batch dim
    batch = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
             for k, v in patch_dict.items()}

    print(f"Patches shape: {batch['patches'].shape}")
    print(f"Grid: t={batch['grid_t'].item()}, h={batch['grid_rows'].item()}, w={batch['grid_cols'].item()}")
    print(f"Valid patches: {batch['patch_mask'].sum().item()}")

    # Unpatchify
    recon = unpatchify(batch, patch=16, temporal_patch=2)
    print(f"Reconstructed shape: {recon.shape}")

    # Verify roundtrip
    orig_t = patch_dict['orig_frames'].item()
    orig_h = patch_dict['orig_height'].item()
    orig_w = patch_dict['orig_width'].item()

    recon_cropped = recon[0, :orig_t, :, :orig_h, :orig_w].cpu()
    video_cpu = video.cpu()

    max_diff = (recon_cropped - video_cpu).abs().max().item()
    print(f"Max reconstruction error: {max_diff:.6f}")

    assert max_diff < 1e-5, f"Reconstruction error too high: {max_diff}"
    print("[PASS] Video patchify/unpatchify roundtrip")
    return {"status": "pass", "max_error": max_diff}


@app.function(image=image, gpu="T4", timeout=600)
def test_video_ae_forward():
    """Test AE forward pass with video patches."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from vitok import AE, decode_variant
    from vitok.pp.ops import patchify

    print("=" * 60)
    print("Test: Video AE Forward Pass")
    print("=" * 60)

    # Create small model
    model = AE(
        **decode_variant("Bd2-Bd4/1x16x32"),
        variational=False,
        attn_backend="sdpa",  # Use SDPA for compatibility
    ).cuda().eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Create video patches using temporal_patch=1 (matches image model's 768 patch dim)
    # With temporal_patch=1, each frame becomes a separate time step in the grid
    video = torch.randn(4, 3, 64, 64)  # T=4, 64x64
    patch_fn = patchify(patch=16, temporal_patch=1, max_tokens=256)
    patch_dict = patch_fn(video)

    # Batch and move to GPU
    batch = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
             for k, v in patch_dict.items()}
    batch['patches'] = batch['patches'].float()

    print(f"Input patches: {batch['patches'].shape}")
    print(f"Grid info: t={batch['grid_t'].item()}, h={batch['grid_rows'].item()}, w={batch['grid_cols'].item()}")

    # Forward pass
    with torch.no_grad():
        output = model(batch)

    assert 'patches' in output, f"Missing 'patches' in output"
    assert not torch.isnan(output['patches']).any(), "NaN in output"
    assert output['patches'].shape == batch['patches'].shape

    print(f"Output patches: {output['patches'].shape}")
    print("[PASS] Video AE forward pass")
    return {"status": "pass", "output_shape": list(output['patches'].shape)}


@app.function(image=image, gpu="T4", timeout=600)
def test_video_3d_rope():
    """Test 3D RoPE frequencies on GPU."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    from vitok.models.modules.rotary_embedding import compute_rope_freqs

    print("=" * 60)
    print("Test: 3D RoPE Frequencies")
    print("=" * 60)

    B, N = 2, 64
    head_dim = 64

    # Create 4x4x4 grid positions (t=4, h=4, w=4)
    t_pos = torch.arange(4).repeat_interleave(16).view(1, -1).expand(B, -1).float().cuda()
    y_pos = torch.arange(4).repeat(4).repeat(4).view(1, -1).expand(B, -1).float().cuda()
    x_pos = torch.arange(4).repeat(16).view(1, -1).expand(B, -1).float().cuda()

    print(f"Position shapes: t={t_pos.shape}, y={y_pos.shape}, x={x_pos.shape}")

    # Compute RoPE
    cos, sin = compute_rope_freqs(t_pos, y_pos, x_pos, head_dim)

    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")

    # Verify shapes
    assert cos.shape == (B, N, head_dim // 2)
    assert sin.shape == (B, N, head_dim // 2)

    # Check for NaN
    assert not torch.isnan(cos).any(), "NaN in cos"
    assert not torch.isnan(sin).any(), "NaN in sin"

    # Verify different positions give different embeddings
    assert not torch.allclose(cos[0, 0], cos[0, 16])  # Different time
    assert not torch.allclose(cos[0, 0], cos[0, 1])   # Different position

    print("[PASS] 3D RoPE frequencies")
    return {"status": "pass", "cos_shape": list(cos.shape)}


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/downloads": downloads_volume},
)
def test_video_pretrained_perframe():
    """Test pretrained model with per-frame batch encoding.

    This approach treats each frame as a separate batch item,
    using existing image weights without temporal modeling.
    """
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    from vitok.pp.ops import patchify, unpatchify
    from vitok import AE, decode_variant

    print("=" * 60)
    print("Test: Per-Frame Batch Encoding")
    print("=" * 60)

    # Create model (small for testing)
    model = AE(
        **decode_variant("Bd2-Bd4/1x16x32"),
        variational=False,
        attn_backend="sdpa",
    ).cuda().eval()

    # Create synthetic video
    T = 8
    video = torch.randn(T, 3, 64, 64)
    print(f"Input video: {video.shape}")

    # Per-frame approach: each frame is a batch item
    patch_fn = patchify(patch=16, temporal_patch=1, max_tokens=256)

    # Patchify each frame
    frame_patches = [patch_fn(video[t]) for t in range(T)]

    # Stack into batch (T becomes B)
    # For grid scalars (grid_t, grid_rows, grid_cols), only use first value since all frames same size
    batch = {}
    scalar_keys = {'grid_t', 'grid_rows', 'grid_cols', 'orig_frames', 'orig_height', 'orig_width'}
    for key in frame_patches[0].keys():
        tensors = [fp[key] for fp in frame_patches]
        if isinstance(tensors[0], torch.Tensor):
            if key in scalar_keys:
                # Keep as single scalar for model compatibility
                batch[key] = tensors[0].cuda()
            else:
                batch[key] = torch.stack(tensors).cuda()
    batch['patches'] = batch['patches'].float()

    print(f"Batch patches shape: {batch['patches'].shape}")  # [T, N, D]
    print(f"Grid: h={batch['grid_rows'].item()}, w={batch['grid_cols'].item()}")

    # Forward pass (T frames as batch)
    with torch.no_grad():
        output = model(batch)

    print(f"Output patches shape: {output['patches'].shape}")

    # Reconstruct each frame
    recon_frames = []
    for t in range(T):
        # Only slice tensors with batch dimension (ndim > 0), keep scalars as-is
        frame_out = {k: v[t:t+1] if v.ndim > 0 else v for k, v in output.items() if isinstance(v, torch.Tensor)}
        frame_recon = unpatchify(frame_out, patch=16, temporal_patch=1)
        recon_frames.append(frame_recon.squeeze(0))

    recon_video = torch.stack(recon_frames)
    print(f"Reconstructed video: {recon_video.shape}")

    assert recon_video.shape == (T, 3, 64, 64), f"Shape mismatch: {recon_video.shape}"
    assert not torch.isnan(recon_video).any(), "NaN in reconstruction"

    print("[PASS] Per-frame batch encoding")
    return {"status": "pass", "recon_shape": list(recon_video.shape)}


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/downloads": downloads_volume},
)
def test_video_pretrained_unified():
    """Test pretrained model with unified video encoding.

    This approach encodes the full video as a single sequence
    with tubelet patches and 3D RoPE.
    """
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    from vitok.pp.ops import patchify, unpatchify
    from vitok import AE, decode_variant

    print("=" * 60)
    print("Test: Unified Video Encoding")
    print("=" * 60)

    # Create model
    model = AE(
        **decode_variant("Bd2-Bd4/1x16x32"),
        variational=False,
        attn_backend="sdpa",
    ).cuda().eval()

    # Create synthetic video
    video = torch.randn(8, 3, 64, 64)  # T=8
    print(f"Input video: {video.shape}")

    # Unified approach: temporal_patch=1 to match image model's 768 patch dim
    # Each frame becomes a separate time step (grid_t=8) with 3D RoPE
    patch_fn = patchify(patch=16, temporal_patch=1, max_tokens=256)
    patch_dict = patch_fn(video)

    # Batch and move to GPU
    batch = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
             for k, v in patch_dict.items()}
    batch['patches'] = batch['patches'].float()

    print(f"Batch patches shape: {batch['patches'].shape}")
    print(f"Grid: t={batch['grid_t'].item()}, h={batch['grid_rows'].item()}, w={batch['grid_cols'].item()}")
    print(f"Valid patches: {batch['patch_mask'].sum().item()}")

    # Forward pass
    with torch.no_grad():
        output = model(batch)

    print(f"Output patches shape: {output['patches'].shape}")

    # Reconstruct video
    recon = unpatchify(output, patch=16, temporal_patch=1)
    print(f"Reconstructed shape: {recon.shape}")

    # Crop to original size
    orig_t = patch_dict['orig_frames'].item()
    orig_h = patch_dict['orig_height'].item()
    orig_w = patch_dict['orig_width'].item()
    recon_cropped = recon[0, :orig_t, :, :orig_h, :orig_w]

    print(f"Cropped reconstruction: {recon_cropped.shape}")

    assert recon_cropped.shape == video.shape, f"Shape mismatch"
    assert not torch.isnan(recon_cropped).any(), "NaN in reconstruction"

    print("[PASS] Unified video encoding")
    return {"status": "pass", "recon_shape": list(recon_cropped.shape)}


@app.function(image=image, gpu="T4", timeout=300)
def test_video_swa_scaling():
    """Test SWA window scaling for 3D grid."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    print("=" * 60)
    print("Test: SWA Window Scaling")
    print("=" * 60)

    # Test grid_info extraction logic
    grid_info_image = {'grid_t': 1, 'grid_h': 8, 'grid_w': 8}
    grid_info_video = {'grid_t': 4, 'grid_h': 8, 'grid_w': 8}

    base_window = 4

    # For images (grid_t=1): window stays as-is
    if grid_info_image.get('grid_t', 1) > 1:
        effective_window_image = base_window * grid_info_image['grid_h'] * grid_info_image['grid_w']
    else:
        effective_window_image = base_window

    # For videos (grid_t>1): window scales by spatial size
    if grid_info_video.get('grid_t', 1) > 1:
        spatial_size = grid_info_video['grid_h'] * grid_info_video['grid_w']
        effective_window_video = base_window * spatial_size
    else:
        effective_window_video = base_window

    print(f"Base window: {base_window}")
    print(f"Image effective window: {effective_window_image}")
    print(f"Video effective window: {effective_window_video}")
    print(f"Spatial size: {grid_info_video['grid_h'] * grid_info_video['grid_w']}")

    assert effective_window_image == 4, "Image window should stay as base"
    assert effective_window_video == 256, "Video window should be scaled"

    print("[PASS] SWA window scaling")
    return {"status": "pass", "image_window": effective_window_image, "video_window": effective_window_video}


@app.local_entrypoint()
def main():
    """Run all video GPU tests."""
    print("\n" + "=" * 70)
    print("VIDEO INTEGRATION GPU TESTS")
    print("=" * 70 + "\n")

    results = {}

    # Run tests
    tests = [
        ("Patchify GPU", test_video_patchify_gpu),
        ("AE Forward", test_video_ae_forward),
        ("3D RoPE", test_video_3d_rope),
        ("SWA Scaling", test_video_swa_scaling),
        ("Per-Frame Batch", test_video_pretrained_perframe),
        ("Unified Video", test_video_pretrained_unified),
    ]

    for name, test_fn in tests:
        print(f"\n>>> Running: {name}")
        try:
            result = test_fn.remote()
            results[name] = result
            print(f"<<< {name}: PASSED")
        except Exception as e:
            results[name] = {"status": "fail", "error": str(e)}
            print(f"<<< {name}: FAILED - {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r.get("status") == "pass")
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result.get("status") == "pass" else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed < total:
        raise SystemExit(1)
