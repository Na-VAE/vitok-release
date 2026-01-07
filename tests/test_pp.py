"""Tests for the preprocessing pipeline (vitok.pp).

Tests use real images and verify the complete pipeline end-to-end.
"""

import math
import os
import tempfile
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vitok.pp import OPS, build_transform, parse_op
from vitok.data import create_dataloader, patch_collate_fn
from vitok.naflex_io import preprocess, postprocess, unpatchify


# =============================================================================
# Fixtures - Real test images
# =============================================================================


@pytest.fixture(scope="module")
def real_images():
    """Create a set of real test images with different sizes and aspect ratios."""
    images = {}

    # Landscape image (640x480)
    h, w = 480, 640
    r = np.linspace(0, 255, w, dtype=np.uint8).reshape(1, -1).repeat(h, axis=0)
    g = np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1).repeat(w, axis=1)
    b = np.full((h, w), 128, dtype=np.uint8)
    images["landscape"] = Image.fromarray(np.stack([r, g, b], axis=-1))

    # Portrait image (480x640)
    h, w = 640, 480
    r = np.linspace(0, 255, w, dtype=np.uint8).reshape(1, -1).repeat(h, axis=0)
    g = np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1).repeat(w, axis=1)
    b = np.full((h, w), 200, dtype=np.uint8)
    images["portrait"] = Image.fromarray(np.stack([r, g, b], axis=-1))

    # Square image (512x512)
    h, w = 512, 512
    x = np.linspace(-1, 1, w).reshape(1, -1).repeat(h, axis=0)
    y = np.linspace(-1, 1, h).reshape(-1, 1).repeat(w, axis=1)
    r = ((np.sin(x * 10) + 1) * 127).astype(np.uint8)
    g = ((np.cos(y * 10) + 1) * 127).astype(np.uint8)
    b = ((np.sin(x * y * 5) + 1) * 127).astype(np.uint8)
    images["square"] = Image.fromarray(np.stack([r, g, b], axis=-1))

    # Small image (128x96)
    h, w = 96, 128
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    images["small"] = Image.fromarray(img)

    # Large image (1920x1080)
    h, w = 1080, 1920
    r = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    g = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    b = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    images["large"] = Image.fromarray(np.stack([r, g, b], axis=-1))

    return images


@pytest.fixture
def webdataset_tar(real_images, tmp_path):
    """Create a real WebDataset tar file with test images."""
    tar_path = tmp_path / "test_dataset.tar"

    with tarfile.open(tar_path, "w") as tar:
        for i, (name, img) in enumerate(real_images.items()):
            # Save image to bytes
            img_bytes = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.save(img_bytes.name, "JPEG", quality=95)

            # Add to tar with proper naming
            tar.add(img_bytes.name, arcname=f"{i:05d}.jpg")

            os.unlink(img_bytes.name)

    return tar_path


# =============================================================================
# Test DSL Parsing
# =============================================================================


class TestParseOp:
    """Tests for DSL string parsing."""

    def test_simple_op_no_args(self):
        name, args, kwargs = parse_op("flip")
        assert name == "flip"
        assert args == ()
        assert kwargs == {}

    def test_op_with_int_arg(self):
        name, args, kwargs = parse_op("center_crop(256)")
        assert name == "center_crop"
        assert args == (256,)

    def test_op_with_multiple_args(self):
        name, args, kwargs = parse_op("patchify(16, 256)")
        assert name == "patchify"
        assert args == (16, 256)

    def test_op_with_tuple_kwarg(self):
        name, args, kwargs = parse_op("random_resized_crop(256, scale=(0.8, 1.0))")
        assert name == "random_resized_crop"
        assert args == (256,)
        assert kwargs == {"scale": (0.8, 1.0)}

    def test_op_with_unquoted_string(self):
        # This is a special case - unquoted strings become identifiers
        name, args, kwargs = parse_op("normalize(minus_one_to_one)")
        assert name == "normalize"
        assert args == ("minus_one_to_one",)

    def test_invalid_syntax_raises(self):
        with pytest.raises(ValueError):
            parse_op("")


class TestOPSDict:
    """Tests for the OPS dictionary."""

    def test_all_core_ops_registered(self):
        expected = ["center_crop", "flip", "normalize", "patchify", "random_resized_crop", "resize_longest_side", "to_tensor"]
        for op in expected:
            assert op in OPS, f"Missing core op: {op}"


# =============================================================================
# Test Individual Transforms with Real Images
# =============================================================================


class TestTransformsWithRealImages:
    """Test each transform with real images."""

    def test_center_crop(self, real_images):
        transform = build_transform("center_crop(256)")

        for name, img in real_images.items():
            result = transform(img)
            assert result.size == (256, 256), f"Failed for {name}"

    def test_random_resized_crop(self, real_images):
        transform = build_transform("random_resized_crop(224, scale=(0.5, 1.0))")

        for name, img in real_images.items():
            result = transform(img)
            assert result.size == (224, 224), f"Failed for {name}"

    def test_flip_deterministic(self, real_images):
        transform = build_transform("flip(1.0)")  # Always flip
        img = real_images["landscape"]

        result = transform(img)
        # Verify horizontal flip by checking corners
        orig_arr = np.array(img)
        result_arr = np.array(result)
        np.testing.assert_array_equal(
            orig_arr[0, 0], result_arr[0, -1],
            err_msg="Flip didn't reverse horizontally"
        )

    def test_to_tensor_value_range(self, real_images):
        transform = build_transform("to_tensor")

        for name, img in real_images.items():
            result = transform(img)
            assert isinstance(result, torch.Tensor)
            assert result.dtype == torch.float32
            assert result.min() >= 0.0, f"Min < 0 for {name}"
            assert result.max() <= 1.0, f"Max > 1 for {name}"
            assert result.shape[0] == 3  # CHW format

    def test_normalize_minus_one_to_one(self, real_images):
        transform = build_transform("to_tensor|normalize(minus_one_to_one)")

        for name, img in real_images.items():
            result = transform(img)
            assert result.min() >= -1.0 - 1e-5, f"Min < -1 for {name}"
            assert result.max() <= 1.0 + 1e-5, f"Max > 1 for {name}"

    def test_normalize_imagenet(self, real_images):
        transform = build_transform("to_tensor|normalize(imagenet)")

        for name, img in real_images.items():
            result = transform(img)
            # ImageNet normalization shifts values based on mean/std
            assert result.shape[0] == 3


# =============================================================================
# Test Patchify (NaFlex) in Detail
# =============================================================================


class TestPatchifyNaFlex:
    """Test patchify op which handles NaFlex variable resolution."""

    def test_patchify_output_keys(self, real_images):
        transform = build_transform("to_tensor|normalize(minus_one_to_one)|patchify(16, 256)")
        result = transform(real_images["landscape"])

        expected_keys = [
            "patches", "patch_mask", "row_idx", "col_idx", "time_idx",
            "orig_height", "orig_width", "grid_rows", "grid_cols", "attention_mask"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_patchify_shapes(self, real_images):
        patch_size = 16
        max_tokens = 256
        transform = build_transform(f"to_tensor|normalize(minus_one_to_one)|patchify({patch_size}, {max_tokens})")
        result = transform(real_images["landscape"])

        assert result["patches"].shape == (max_tokens, patch_size * patch_size * 3)
        assert result["patch_mask"].shape == (max_tokens,)
        assert result["row_idx"].shape == (max_tokens,)
        assert result["col_idx"].shape == (max_tokens,)
        assert result["attention_mask"].shape == (max_tokens, max_tokens)

    def test_patchify_respects_token_budget(self, real_images):
        """Large images should be resized to fit token budget."""
        max_tokens = 64
        transform = build_transform(f"to_tensor|normalize(minus_one_to_one)|patchify(16, {max_tokens})")

        result = transform(real_images["large"])  # 1920x1080
        n_valid = result["patch_mask"].sum().item()
        assert n_valid <= max_tokens, f"Exceeded token budget: {n_valid} > {max_tokens}"

    def test_patchify_grid_consistency(self, real_images):
        """Grid dimensions should match valid patch count."""
        transform = build_transform("to_tensor|normalize(minus_one_to_one)|patchify(16, 256)")

        for name, img in real_images.items():
            result = transform(img)
            grid_rows = result["grid_rows"].item()
            grid_cols = result["grid_cols"].item()
            n_valid = result["patch_mask"].sum().item()
            assert grid_rows * grid_cols == n_valid, f"Grid mismatch for {name}"

    def test_patchify_attention_mask(self, real_images):
        """Attention mask should only connect valid patches."""
        transform = build_transform("to_tensor|normalize(minus_one_to_one)|patchify(16, 256)")
        result = transform(real_images["landscape"])

        patch_mask = result["patch_mask"]
        attn = result["attention_mask"]

        # Valid patches should attend to each other
        n_valid = patch_mask.sum().item()
        valid_block = attn[:n_valid, :n_valid]
        assert valid_block.all(), "Valid patches should attend to each other"

        # Invalid patches should not be attended to (except diagonal)
        if n_valid < 256:
            invalid_attn = attn[:n_valid, n_valid:]
            assert not invalid_attn.any(), "Should not attend to invalid patches"

    def test_patchify_spatial_indices(self, real_images):
        """Spatial indices should form a valid grid."""
        transform = build_transform("to_tensor|normalize(minus_one_to_one)|patchify(16, 256)")
        result = transform(real_images["square"])

        patch_mask = result["patch_mask"]
        row_idx = result["row_idx"][patch_mask]
        col_idx = result["col_idx"][patch_mask]

        grid_rows = result["grid_rows"].item()
        grid_cols = result["grid_cols"].item()

        # All indices should be valid
        assert row_idx.max() < grid_rows
        assert col_idx.max() < grid_cols
        assert row_idx.min() >= 0
        assert col_idx.min() >= 0

    def test_patchify_different_sizes(self, real_images):
        """Test patchify handles different image sizes correctly."""
        transform = build_transform("to_tensor|normalize(minus_one_to_one)|patchify(16, 256)")

        results = {}
        for name, img in real_images.items():
            result = transform(img)
            results[name] = {
                "n_valid": result["patch_mask"].sum().item(),
                "grid_rows": result["grid_rows"].item(),
                "grid_cols": result["grid_cols"].item(),
                "orig_h": result["orig_height"].item(),
                "orig_w": result["orig_width"].item(),
            }

        # Large image should have fewer tokens due to budget
        assert results["large"]["n_valid"] <= 256

        # Small image should use fewer tokens
        assert results["small"]["n_valid"] < results["landscape"]["n_valid"]


# =============================================================================
# Test NaFlex Roundtrip (Pack and Unpack)
# =============================================================================


class TestNaFlexRoundtrip:
    """Test packing images to patches and unpacking back."""

    def test_unpatchify_basic(self, real_images):
        """Test unpatchify reconstructs image shape."""
        transform = build_transform("to_tensor|normalize(minus_one_to_one)|patchify(16, 256)")
        patch_dict = transform(real_images["landscape"])

        # Add batch dimension
        batch_dict = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                      for k, v in patch_dict.items()}

        reconstructed = unpatchify(batch_dict, patch=16)
        assert reconstructed.ndim == 4  # B, C, H, W
        assert reconstructed.shape[1] == 3

    def test_roundtrip_preserves_content(self, real_images):
        """Roundtrip should preserve image content."""
        transform = build_transform("to_tensor|normalize(minus_one_to_one)|patchify(16, 256)")
        img = real_images["square"]  # 512x512 -> fits exactly

        patch_dict = transform(img)
        batch_dict = {k: v.unsqueeze(0) for k, v in patch_dict.items()}

        reconstructed = unpatchify(batch_dict, patch=16)

        # Get original tensor for comparison
        tensor_transform = build_transform("to_tensor|normalize(minus_one_to_one)")
        # Need to resize to match what patchify did
        from torchvision.transforms.functional import resize
        orig_tensor = tensor_transform(img)
        orig_h = patch_dict["orig_height"].item()
        orig_w = patch_dict["orig_width"].item()
        orig_tensor = resize(orig_tensor, [orig_h, orig_w])

        # Compare valid region
        recon_crop = reconstructed[0, :, :orig_h, :orig_w]

        # Should be very close (within floating point tolerance)
        diff = (recon_crop - orig_tensor).abs().max().item()
        assert diff < 1e-5, f"Roundtrip error too large: {diff}"

    def test_postprocess_unpack(self, real_images):
        """Test postprocess with do_unpack=True."""
        transform = build_transform("to_tensor|normalize(minus_one_to_one)|patchify(16, 256)")

        patch_dicts = [transform(img) for img in [real_images["landscape"], real_images["portrait"]]]
        batch = patch_collate_fn(patch_dicts)

        # Postprocess with unpack
        images = postprocess(batch, output_format="zero_to_one", do_unpack=True, patch=16)

        assert isinstance(images, list)
        assert len(images) == 2

        # Each image should have original dimensions
        for i, img in enumerate(images):
            orig_h = batch["orig_height"][i].item()
            orig_w = batch["orig_width"][i].item()
            assert img.shape[1] == orig_h
            assert img.shape[2] == orig_w

    def test_postprocess_format_conversion(self, real_images):
        """Test format conversion in postprocess."""
        transform = build_transform("to_tensor|normalize(minus_one_to_one)|patchify(16, 256)")
        patch_dict = transform(real_images["landscape"])
        batch = {k: v.unsqueeze(0) for k, v in patch_dict.items()}

        # Convert to 0-255
        result = postprocess(
            batch,
            output_format="0_255",
            current_format="minus_one_to_one",
            do_unpack=False,
            patch=16
        )

        assert result.dtype == torch.uint8
        assert result.min() >= 0
        assert result.max() <= 255


# =============================================================================
# Test preprocess High-Level API
# =============================================================================


class TestPreprocess:
    """Test the high-level preprocess function."""

    def test_single_image(self, real_images):
        batch = preprocess(
            real_images["landscape"],
            pp="to_tensor|normalize(minus_one_to_one)|patchify(16, 256)",
            device="cpu"
        )

        assert batch["patches"].shape[0] == 1  # Batch size 1
        assert batch["patches"].shape[1] == 256  # Max tokens

    def test_multiple_images(self, real_images):
        images = [real_images["landscape"], real_images["portrait"], real_images["square"]]

        batch = preprocess(
            images,
            pp="to_tensor|normalize(minus_one_to_one)|patchify(16, 256)",
            device="cpu"
        )

        assert batch["patches"].shape[0] == 3


# =============================================================================
# Test WebDataset Integration
# =============================================================================


class TestWebDatasetIntegration:
    """Test WebDataset loading with the new pipeline."""

    def test_create_dataloader_local(self, webdataset_tar):
        """Test loading from local tar file."""
        loader = create_dataloader(
            source=str(webdataset_tar),
            pp="center_crop(256)|to_tensor|normalize(minus_one_to_one)|patchify(16, 256)",
            batch_size=2,
            num_workers=0,
            seed=42,
        )

        # Get one batch
        batch = next(iter(loader))

        assert "patches" in batch
        assert batch["patches"].shape[0] == 2  # Batch size
        assert batch["patches"].shape[1] == 256  # Max tokens

    def test_create_dataloader_variable_resolution(self, webdataset_tar):
        """Test NaFlex variable resolution with WebDataset."""
        loader = create_dataloader(
            source=str(webdataset_tar),
            pp="to_tensor|normalize(minus_one_to_one)|patchify(16, 256)",
            batch_size=2,
            num_workers=0,
            seed=42,
        )

        batch = next(iter(loader))

        # Different images should have different valid patch counts
        patch_mask = batch["patch_mask"]
        valid_counts = patch_mask.sum(dim=1)

        # All should be <= max_tokens
        assert (valid_counts <= 256).all()

    def test_patch_collate_fn_batching(self, real_images):
        """Test that collate function properly batches variable-sized patch dicts."""
        transform = build_transform("to_tensor|normalize(minus_one_to_one)|patchify(16, 256)")

        # Create patch dicts with different valid counts
        dicts = [
            transform(real_images["small"]),
            transform(real_images["landscape"]),
            transform(real_images["large"]),
        ]

        batch = patch_collate_fn(dicts)

        assert batch["patches"].shape[0] == 3

        # Each should have different valid counts
        valid_counts = batch["patch_mask"].sum(dim=1).tolist()
        assert len(set(valid_counts)) > 1, "Expected different valid counts"


# =============================================================================
# Test Complete Training Pipeline Simulation
# =============================================================================


class TestTrainingPipeline:
    """Simulate a complete training pipeline."""

    def test_training_iteration(self, webdataset_tar):
        """Simulate a training iteration with the new pipeline."""
        loader = create_dataloader(
            source=str(webdataset_tar),
            pp="random_resized_crop(256)|flip|to_tensor|normalize(minus_one_to_one)|patchify(16, 256)",
            batch_size=2,
            num_workers=0,
            seed=42,
        )

        # Simulate training loop
        for i, batch in enumerate(loader):
            if i >= 2:
                break

            # Verify batch structure
            assert batch["patches"].shape == (2, 256, 768)
            assert batch["patch_mask"].shape == (2, 256)
            assert batch["attention_mask"].shape == (2, 256, 256)

            # Simulate forward pass - just verify we can do math
            patches = batch["patches"]
            mean_patch = patches.mean(dim=-1, keepdim=True)
            assert mean_patch.shape == (2, 256, 1)

    def test_inference_pipeline(self, real_images):
        """Test inference pipeline produces valid output."""
        # Preprocess
        batch = preprocess(
            [real_images["landscape"], real_images["portrait"]],
            pp="to_tensor|normalize(minus_one_to_one)|patchify(16, 256)",
            device="cpu"
        )

        # Simulate model output (identity for testing)
        fake_output = batch.copy()

        # Postprocess
        images = postprocess(
            fake_output,
            output_format="zero_to_one",
            current_format="minus_one_to_one",
            do_unpack=True,
            patch=16
        )

        assert len(images) == 2
        for img in images:
            assert img.min() >= 0
            assert img.max() <= 1
