"""Visual tests that save images for inspection.

Run with: pytest tests/test_pp_visual.py -v
Images saved to: tests/test_outputs/
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from vitok.pp import build_transform
from vitok.data import patch_collate_fn
from vitok.pp.io import preprocess_images, postprocess_images, unpatchify


OUTPUT_DIR = Path(__file__).parent / "test_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_image(img, name: str):
    """Save image (PIL, tensor, or numpy) to test_outputs."""
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img[0]
        # Assume CHW format, convert to HWC
        img = img.permute(1, 2, 0).cpu().numpy()
        # Handle normalization
        if img.min() < 0:
            img = (img + 1) / 2
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img)
    elif isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img)

    path = OUTPUT_DIR / f"{name}.png"
    img.save(path)
    print(f"Saved: {path}")
    return path


def create_test_images():
    """Create labeled test images."""
    images = {}

    # Landscape with gradient and text
    h, w = 480, 640
    r = np.linspace(50, 200, w, dtype=np.uint8).reshape(1, -1).repeat(h, axis=0)
    g = np.linspace(50, 200, h, dtype=np.uint8).reshape(-1, 1).repeat(w, axis=1)
    b = np.full((h, w), 100, dtype=np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=-1))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"Landscape {w}x{h}", fill=(255, 255, 255))
    draw.rectangle([50, 50, 150, 150], outline=(255, 0, 0), width=3)
    draw.ellipse([200, 100, 300, 200], outline=(0, 255, 0), width=3)
    images["01_landscape"] = img

    # Portrait
    h, w = 640, 480
    r = np.linspace(200, 50, w, dtype=np.uint8).reshape(1, -1).repeat(h, axis=0)
    g = np.full((h, w), 100, dtype=np.uint8)
    b = np.linspace(50, 200, h, dtype=np.uint8).reshape(-1, 1).repeat(w, axis=1)
    img = Image.fromarray(np.stack([r, g, b], axis=-1))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"Portrait {w}x{h}", fill=(255, 255, 255))
    images["02_portrait"] = img

    # Square with pattern
    h, w = 512, 512
    x = np.linspace(-1, 1, w).reshape(1, -1).repeat(h, axis=0)
    y = np.linspace(-1, 1, h).reshape(-1, 1).repeat(w, axis=1)
    r = ((np.sin(x * 15) + 1) * 100 + 50).astype(np.uint8)
    g = ((np.cos(y * 15) + 1) * 100 + 50).astype(np.uint8)
    b = ((np.sin((x + y) * 10) + 1) * 100 + 50).astype(np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=-1))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"Square {w}x{h}", fill=(255, 255, 255))
    images["03_square"] = img

    # Small image
    h, w = 128, 192
    img = Image.fromarray(np.random.randint(50, 200, (h, w, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), f"Small {w}x{h}", fill=(255, 255, 255))
    images["04_small"] = img

    # Large image with checkerboard
    h, w = 1080, 1920
    checker_size = 60
    x_idx = np.arange(w) // checker_size
    y_idx = np.arange(h) // checker_size
    checker = ((x_idx.reshape(1, -1) + y_idx.reshape(-1, 1)) % 2).astype(np.uint8)
    r = checker * 150 + 50
    g = checker * 100 + 80
    b = (1 - checker) * 150 + 50
    img = Image.fromarray(np.stack([r, g, b], axis=-1).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"Large {w}x{h}", fill=(255, 0, 0))
    images["05_large"] = img

    return images


class TestVisualTransforms:
    """Save visual outputs for each transform."""

    def test_save_input_images(self):
        """Save the input test images."""
        images = create_test_images()
        for name, img in images.items():
            save_image(img, f"input_{name}")

    def test_center_crop(self):
        """Visualize center crop."""
        images = create_test_images()
        transform = build_transform("center_crop(256)")

        for name, img in images.items():
            result = transform(img)
            save_image(result, f"center_crop_256_{name}")

    def test_random_resized_crop(self):
        """Visualize random resized crop (multiple samples)."""
        images = create_test_images()
        transform = build_transform("random_resized_crop(256, scale=(0.5, 1.0))")

        for name, img in list(images.items())[:2]:
            for i in range(3):
                result = transform(img)
                save_image(result, f"random_resized_crop_256_{name}_sample{i}")

    def test_flip(self):
        """Visualize horizontal flip."""
        images = create_test_images()
        transform_flip = build_transform("flip(1.0)")  # Always flip

        img = images["01_landscape"]
        save_image(img, "flip_original")
        save_image(transform_flip(img), "flip_flipped")

    def test_normalize_visualization(self):
        """Visualize normalization by converting back."""
        images = create_test_images()
        img = images["01_landscape"]

        # Original as tensor
        to_tensor = build_transform("to_tensor")
        tensor = to_tensor(img)
        save_image(tensor, "normalize_01_original_tensor")

        # Minus one to one (need to convert back for viz)
        transform = build_transform("to_tensor|normalize(minus_one_to_one)")
        normalized = transform(img)
        # Convert back: [-1, 1] -> [0, 1]
        viz = (normalized + 1) / 2
        save_image(viz, "normalize_02_minus_one_to_one")

        # ImageNet (harder to visualize, just save raw)
        transform_in = build_transform("to_tensor|normalize(imagenet)")
        normalized_in = transform_in(img)
        # Rough denormalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denorm = normalized_in * std + mean
        save_image(denorm.clamp(0, 1), "normalize_03_imagenet_denorm")


class TestVisualPatchify:
    """Visualize patchification."""

    def test_patchify_grid_overlay(self):
        """Show patch grid overlay on images."""
        images = create_test_images()
        patch_size = 16

        for name, img in images.items():
            transform = build_transform(f"to_tensor|normalize(minus_one_to_one)|patchify(512, {patch_size}, 256)")
            result = transform(img)

            # Get dimensions
            orig_h = result["original_height"].item()
            orig_w = result["original_width"].item()
            grid_h = result["grid_h"].item()
            grid_w = result["grid_w"].item()
            n_valid = result["ptype"].sum().item()

            # Create visualization
            # First resize original to match patchified size
            img_resized = img.resize((orig_w, orig_h), Image.LANCZOS)
            draw = ImageDraw.Draw(img_resized)

            # Draw grid
            for i in range(grid_h + 1):
                y = i * patch_size
                if y <= orig_h:
                    draw.line([(0, y), (orig_w, y)], fill=(255, 255, 0), width=1)
            for j in range(grid_w + 1):
                x = j * patch_size
                if x <= orig_w:
                    draw.line([(x, 0), (x, orig_h)], fill=(255, 255, 0), width=1)

            # Add info
            draw.text((5, 5), f"Grid: {grid_w}x{grid_h} = {n_valid} patches", fill=(255, 0, 0))
            draw.text((5, 20), f"Size: {orig_w}x{orig_h}", fill=(255, 0, 0))

            save_image(img_resized, f"patchify_grid_{name}")

    def test_patchify_token_budget(self):
        """Show how token budget affects large images."""
        images = create_test_images()
        large_img = images["05_large"]  # 1920x1080

        for max_tokens in [64, 128, 256]:
            transform = build_transform(f"to_tensor|normalize(minus_one_to_one)|patchify(512, 16, {max_tokens})")
            result = transform(large_img)

            orig_h = result["original_height"].item()
            orig_w = result["original_width"].item()
            n_valid = result["ptype"].sum().item()

            # Resize for visualization
            scale = min(800 / orig_w, 600 / orig_h)
            viz_w, viz_h = int(orig_w * scale), int(orig_h * scale)

            # Reconstruct and resize
            batch = {k: v.unsqueeze(0) for k, v in result.items()}
            recon = unpatchify(batch, patch=16)
            recon = (recon[0] + 1) / 2  # Denormalize

            # Resize for saving
            recon_pil = Image.fromarray((recon.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            recon_pil = recon_pil.resize((viz_w, viz_h), Image.LANCZOS)

            draw = ImageDraw.Draw(recon_pil)
            draw.text((5, 5), f"max_tokens={max_tokens}, used={n_valid}", fill=(255, 0, 0))
            draw.text((5, 20), f"Size: {orig_w}x{orig_h}", fill=(255, 0, 0))

            save_image(recon_pil, f"patchify_budget_{max_tokens}_large")


class TestVisualRoundtrip:
    """Visualize pack/unpack roundtrip."""

    def test_roundtrip_comparison(self):
        """Side-by-side comparison of original and reconstructed."""
        images = create_test_images()

        for name, img in list(images.items())[:3]:
            # Original
            save_image(img, f"roundtrip_{name}_1_original")

            # Patchify
            transform = build_transform("to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)")
            patch_dict = transform(img)

            # Info about patchification
            orig_h = patch_dict["original_height"].item()
            orig_w = patch_dict["original_width"].item()
            n_valid = patch_dict["ptype"].sum().item()

            # Reconstruct
            batch = {k: v.unsqueeze(0) for k, v in patch_dict.items()}
            recon = unpatchify(batch, patch=16)

            # Crop to original size
            recon_cropped = recon[0, :, :orig_h, :orig_w]

            # Denormalize and save
            recon_viz = (recon_cropped + 1) / 2
            save_image(recon_viz, f"roundtrip_{name}_2_reconstructed")

            # Create comparison image
            orig_resized = img.resize((orig_w, orig_h), Image.LANCZOS)
            recon_pil = Image.fromarray((recon_viz.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

            # Side by side
            comparison = Image.new("RGB", (orig_w * 2 + 10, orig_h + 40), (50, 50, 50))
            comparison.paste(orig_resized, (0, 30))
            comparison.paste(recon_pil, (orig_w + 10, 30))

            draw = ImageDraw.Draw(comparison)
            draw.text((10, 5), "Original", fill=(255, 255, 255))
            draw.text((orig_w + 20, 5), f"Reconstructed ({n_valid} patches)", fill=(255, 255, 255))

            save_image(comparison, f"roundtrip_{name}_3_comparison")


class TestVisualBatching:
    """Visualize batching of multiple images."""

    def test_batch_variable_resolution(self):
        """Show batch with different resolutions."""
        images = create_test_images()
        selected = [images["01_landscape"], images["02_portrait"], images["04_small"]]

        # Process batch
        batch = preprocess_images(
            selected,
            pp="to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)",
            device="cpu"
        )

        # Reconstruct each
        for i in range(3):
            single = {k: v[i:i+1] for k, v in batch.items()}
            recon = unpatchify(single, patch=16)

            orig_h = batch["original_height"][i].item()
            orig_w = batch["original_width"][i].item()
            n_valid = batch["ptype"][i].sum().item()

            recon_cropped = recon[0, :, :orig_h, :orig_w]
            recon_viz = (recon_cropped + 1) / 2

            # Add label
            recon_pil = Image.fromarray((recon_viz.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            draw = ImageDraw.Draw(recon_pil)
            draw.text((5, 5), f"Batch item {i}: {orig_w}x{orig_h}, {n_valid} patches", fill=(255, 0, 0))

            save_image(recon_pil, f"batch_item_{i}")

        # Create grid of all
        max_h = max(batch["original_height"]).item()
        total_w = sum(batch["original_width"]).item() + 20

        grid = Image.new("RGB", (total_w, max_h + 30), (30, 30, 30))
        x_offset = 0

        for i in range(3):
            single = {k: v[i:i+1] for k, v in batch.items()}
            recon = unpatchify(single, patch=16)

            orig_h = batch["original_height"][i].item()
            orig_w = batch["original_width"][i].item()

            recon_cropped = recon[0, :, :orig_h, :orig_w]
            recon_viz = (recon_cropped + 1) / 2
            recon_pil = Image.fromarray((recon_viz.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

            grid.paste(recon_pil, (x_offset, 30))
            x_offset += orig_w + 10

        draw = ImageDraw.Draw(grid)
        draw.text((10, 5), "Batched variable-resolution images (NaFlex)", fill=(255, 255, 255))

        save_image(grid, "batch_grid_all")


class TestVisualPipelines:
    """Visualize complete pipelines."""

    def test_training_pipeline(self):
        """Visualize training augmentation pipeline."""
        images = create_test_images()
        img = images["01_landscape"]

        # Save original
        save_image(img, "pipeline_train_0_original")

        # Multiple augmented versions
        transform = build_transform(
            "random_resized_crop(256, scale=(0.5, 1.0))|flip|to_tensor|normalize(minus_one_to_one)|patchify(256, 16, 256)"
        )

        for i in range(4):
            result = transform(img)

            # Reconstruct
            batch = {k: v.unsqueeze(0) for k, v in result.items()}
            recon = unpatchify(batch, patch=16)
            recon_viz = (recon[0] + 1) / 2

            save_image(recon_viz, f"pipeline_train_{i+1}_augmented")

    def test_inference_pipeline(self):
        """Visualize inference pipeline (no augmentation)."""
        images = create_test_images()

        transform = build_transform(
            "to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)"
        )

        for name, img in list(images.items())[:3]:
            # Save original
            save_image(img, f"pipeline_infer_{name}_1_input")

            # Process
            result = transform(img)

            # Reconstruct
            batch = {k: v.unsqueeze(0) for k, v in result.items()}
            recon = unpatchify(batch, patch=16)

            orig_h = result["original_height"].item()
            orig_w = result["original_width"].item()

            recon_cropped = recon[0, :, :orig_h, :orig_w]
            recon_viz = (recon_cropped + 1) / 2

            save_image(recon_viz, f"pipeline_infer_{name}_2_output")


if __name__ == "__main__":
    # Run all visual tests and save outputs
    import pytest
    pytest.main([__file__, "-v"])
