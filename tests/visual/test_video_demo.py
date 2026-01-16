"""Visual demo for video encoding/decoding.

Creates synthetic video, encodes with two approaches, decodes, and saves comparison.

Run with: python tests/visual/test_video_demo.py

Output: tests/visual/test_outputs/video_*.png
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from vitok.pp.ops import patchify, unpatchify

OUTPUT_DIR = Path(__file__).parent / "test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_synthetic_video(frames: int = 8, size: int = 256) -> torch.Tensor:
    """Create a moving gradient video for testing.

    Returns:
        Tensor of shape (T, C, H, W) with values in [-1, 1]
    """
    video = torch.zeros(frames, 3, size, size)

    for t in range(frames):
        # Moving diagonal gradient
        offset = (t / frames) * size
        for y in range(size):
            for x in range(size):
                # Create moving diagonal pattern
                val = ((x + y + offset) % size) / size
                video[t, 0, y, x] = val  # R channel
                video[t, 1, y, x] = 1 - val  # G channel
                video[t, 2, y, x] = abs(2 * val - 1)  # B channel

    # Normalize to [-1, 1]
    return video * 2 - 1


def create_bouncing_ball_video(frames: int = 8, size: int = 256) -> torch.Tensor:
    """Create a bouncing ball video for testing.

    Returns:
        Tensor of shape (T, C, H, W) with values in [-1, 1]
    """
    video = torch.zeros(frames, 3, size, size)
    ball_radius = size // 8

    for t in range(frames):
        # Ball position (bouncing)
        progress = t / (frames - 1)
        ball_x = int(ball_radius + progress * (size - 2 * ball_radius))
        ball_y = int(size // 2 + np.sin(progress * 2 * np.pi) * (size // 4))

        # Draw background gradient
        for y in range(size):
            for x in range(size):
                video[t, 0, y, x] = y / size * 0.3
                video[t, 1, y, x] = x / size * 0.3
                video[t, 2, y, x] = 0.2

        # Draw ball
        for dy in range(-ball_radius, ball_radius + 1):
            for dx in range(-ball_radius, ball_radius + 1):
                if dx * dx + dy * dy <= ball_radius * ball_radius:
                    y, x = ball_y + dy, ball_x + dx
                    if 0 <= y < size and 0 <= x < size:
                        # Ball color (yellow-ish)
                        video[t, 0, y, x] = 1.0
                        video[t, 1, y, x] = 0.8
                        video[t, 2, y, x] = 0.2

    # Normalize to [-1, 1]
    return video * 2 - 1


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor (C, H, W) in [-1, 1] to PIL Image."""
    # Denormalize
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)

    # Convert to numpy
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_video_frames(video: torch.Tensor, prefix: str, label: str = ""):
    """Save video frames as individual images."""
    T = video.shape[0]
    for t in range(T):
        img = tensor_to_pil(video[t])
        draw = ImageDraw.Draw(img)
        if label:
            draw.text((5, 5), f"{label} t={t}", fill=(255, 255, 255))
        img.save(OUTPUT_DIR / f"{prefix}_frame{t:02d}.png")


def save_video_grid(video: torch.Tensor, filename: str, label: str = ""):
    """Save video as a grid of frames."""
    T, C, H, W = video.shape
    cols = min(4, T)
    rows = (T + cols - 1) // cols

    grid_w = cols * W + (cols - 1) * 5
    grid_h = rows * H + (rows - 1) * 5 + 30

    grid = Image.new("RGB", (grid_w, grid_h), (50, 50, 50))
    draw = ImageDraw.Draw(grid)

    if label:
        draw.text((5, 5), label, fill=(255, 255, 255))

    for t in range(T):
        row, col = t // cols, t % cols
        x = col * (W + 5)
        y = row * (H + 5) + 30

        frame = tensor_to_pil(video[t])
        grid.paste(frame, (x, y))

        # Frame number
        draw.text((x + 5, y + 5), f"t={t}", fill=(255, 255, 0))

    grid.save(OUTPUT_DIR / filename)
    print(f"Saved: {OUTPUT_DIR / filename}")


def test_patchify_roundtrip_visual():
    """Visual test: patchify/unpatchify roundtrip without model."""
    print("\n" + "=" * 60)
    print("Visual Test: Patchify/Unpatchify Roundtrip")
    print("=" * 60)

    # Create video
    video = create_bouncing_ball_video(frames=8, size=128)
    print(f"Video shape: {video.shape}")

    # Save original
    save_video_grid(video, "video_01_original.png", "Original Video (8 frames)")

    # Test unified approach (temporal_patch=2)
    print("\nUnified approach (temporal_patch=2):")
    patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=256)
    patch_dict = patch_fn(video)

    print(f"  Patches: {patch_dict['patches'].shape}")
    print(f"  Grid: t={patch_dict['grid_t'].item()}, h={patch_dict['grid_rows'].item()}, w={patch_dict['grid_cols'].item()}")
    print(f"  Valid: {patch_dict['patch_mask'].sum().item()}")

    # Unpatchify
    batch = {k: v.unsqueeze(0) for k, v in patch_dict.items()}
    recon = unpatchify(batch, patch=16, temporal_patch=2)

    # Crop to original size
    orig_t = patch_dict['orig_frames'].item()
    orig_h = patch_dict['orig_height'].item()
    orig_w = patch_dict['orig_width'].item()
    recon_cropped = recon[0, :orig_t, :, :orig_h, :orig_w]

    save_video_grid(recon_cropped, "video_02_unified_recon.png", "Unified Reconstruction (temporal_patch=2)")

    # Compute error
    error = (video - recon_cropped).abs().mean().item()
    print(f"  Mean reconstruction error: {error:.6f}")

    # Test per-frame approach (temporal_patch=1)
    print("\nPer-frame approach (temporal_patch=1):")
    patch_fn_2d = patchify(patch=16, temporal_patch=1, max_tokens=256)

    recon_frames = []
    for t in range(video.shape[0]):
        frame = video[t]
        patch_dict_2d = patch_fn_2d(frame)
        batch_2d = {k: v.unsqueeze(0) for k, v in patch_dict_2d.items()}
        frame_recon = unpatchify(batch_2d, patch=16, temporal_patch=1)
        recon_frames.append(frame_recon[0])

    recon_perframe = torch.stack(recon_frames)
    print(f"  Reconstructed: {recon_perframe.shape}")

    save_video_grid(recon_perframe, "video_03_perframe_recon.png", "Per-Frame Reconstruction (temporal_patch=1)")

    error_perframe = (video - recon_perframe).abs().mean().item()
    print(f"  Mean reconstruction error: {error_perframe:.6f}")

    # Create comparison
    create_comparison_image(video, recon_cropped, recon_perframe)


def create_comparison_image(original: torch.Tensor, unified: torch.Tensor, perframe: torch.Tensor):
    """Create side-by-side comparison of approaches."""
    T = original.shape[0]

    # Select 4 frames to compare
    frames_to_show = [0, T // 4, T // 2, T - 1]

    H, W = original.shape[2], original.shape[3]
    comp_w = W * len(frames_to_show) + (len(frames_to_show) - 1) * 5
    comp_h = H * 3 + 2 * 5 + 80

    comp = Image.new("RGB", (comp_w, comp_h), (30, 30, 30))
    draw = ImageDraw.Draw(comp)

    # Labels
    draw.text((5, 5), "Video Encoding Comparison", fill=(255, 255, 255))
    draw.text((5, 25), "Row 1: Original | Row 2: Unified (3D tubelets) | Row 3: Per-Frame (2D patches)", fill=(200, 200, 200))

    row_labels = ["Original", "Unified", "Per-Frame"]

    for row, (video_data, label) in enumerate([(original, "Original"), (unified, "Unified"), (perframe, "Per-Frame")]):
        y_offset = 50 + row * (H + 5)

        for col, t in enumerate(frames_to_show):
            x_offset = col * (W + 5)

            frame = tensor_to_pil(video_data[t])
            comp.paste(frame, (x_offset, y_offset))

            # Frame label
            if row == 0:
                draw.text((x_offset + 5, y_offset + 5), f"t={t}", fill=(255, 255, 0))

    # Row labels on the right
    for row, label in enumerate(row_labels):
        y_offset = 50 + row * (H + 5) + H // 2
        draw.text((comp_w - 80, y_offset), label, fill=(255, 255, 255))

    comp.save(OUTPUT_DIR / "video_04_comparison.png")
    print(f"Saved: {OUTPUT_DIR / 'video_04_comparison.png'}")


def test_gradient_video_visual():
    """Visual test with gradient video."""
    print("\n" + "=" * 60)
    print("Visual Test: Gradient Video")
    print("=" * 60)

    video = create_synthetic_video(frames=8, size=128)
    save_video_grid(video, "video_05_gradient_original.png", "Gradient Video Original")

    # Patchify with temporal_patch=2
    patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=256)
    patch_dict = patch_fn(video)

    batch = {k: v.unsqueeze(0) for k, v in patch_dict.items()}
    recon = unpatchify(batch, patch=16, temporal_patch=2)

    orig_t = patch_dict['orig_frames'].item()
    orig_h = patch_dict['orig_height'].item()
    orig_w = patch_dict['orig_width'].item()
    recon_cropped = recon[0, :orig_t, :, :orig_h, :orig_w]

    save_video_grid(recon_cropped, "video_06_gradient_recon.png", "Gradient Video Reconstructed")

    print(f"Reconstruction error: {(video - recon_cropped).abs().mean().item():.6f}")


def main():
    """Run all visual tests."""
    print("=" * 70)
    print("VIDEO INTEGRATION VISUAL DEMO")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")

    test_patchify_roundtrip_visual()
    test_gradient_video_visual()

    print("\n" + "=" * 70)
    print("All visual tests complete!")
    print(f"Check outputs in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
