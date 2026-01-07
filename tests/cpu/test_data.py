"""Test data streaming and NaFlex dictionary format.

This test verifies:
1. Data streaming works with WebDataset
2. NaFlex dictionary contains required keys
3. Postprocessing pipeline

Run with: pytest tests/cpu/test_data.py -v
"""

import os
import sys
import tempfile
import tarfile
from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_test_tar(tar_path: str, num_images: int = 10):
    """Create a small test tar file with random images."""
    with tarfile.open(tar_path, 'w') as tar:
        for i in range(num_images):
            # Create random RGB image
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                img.save(f, 'JPEG')
                temp_path = f.name

            # Add to tar with proper naming
            tar.add(temp_path, arcname=f"{i:06d}.jpg")

            # Add class label
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cls', delete=False) as f:
                f.write(str(i % 10))  # Class 0-9
                cls_path = f.name
            tar.add(cls_path, arcname=f"{i:06d}.cls")

            # Cleanup temp files
            os.unlink(temp_path)
            os.unlink(cls_path)


@pytest.fixture
def test_tar_path(tmp_path):
    """Create a test tar file."""
    tar_path = str(tmp_path / "test_images.tar")
    create_test_tar(tar_path, num_images=10)
    return tar_path


def test_naflex_dict_format(test_tar_path):
    """Test that NaFlex dictionary contains required keys."""
    from vitok.pp import build_transform
    from vitok.data import create_dataloader, patch_collate_fn

    # Build transform with patchify
    pp_string = "to_tensor|normalize(minus_one_to_one)|patchify(16, 256)"

    # Create dataloader
    loader = create_dataloader(
        source=test_tar_path,
        pp=pp_string,
        batch_size=4,
        num_workers=0,
        seed=42,
    )

    batch = next(iter(loader))

    # Verify NaFlex dictionary keys (support both old and new key names)
    required_keys = ['patches']
    for key in required_keys:
        assert key in batch, f"Missing key: {key}"

    # Check for either old or new key names
    assert 'ptype' in batch or 'patch_mask' in batch, "Missing patch type/mask key"
    assert 'yidx' in batch or 'row_idx' in batch, "Missing row index key"
    assert 'xidx' in batch or 'col_idx' in batch, "Missing col index key"

    # Verify shapes
    B = batch['patches'].shape[0]
    L = batch['patches'].shape[1]  # Sequence length

    assert batch['patches'].ndim == 3, "patches should be [B, L, C]"

    ptype_key = 'ptype' if 'ptype' in batch else 'patch_mask'
    yidx_key = 'yidx' if 'yidx' in batch else 'row_idx'
    xidx_key = 'xidx' if 'xidx' in batch else 'col_idx'

    assert batch[ptype_key].shape == (B, L), f"patch_mask shape mismatch: {batch[ptype_key].shape}"
    assert batch[yidx_key].shape == (B, L), f"row_idx shape mismatch: {batch[yidx_key].shape}"
    assert batch[xidx_key].shape == (B, L), f"col_idx shape mismatch: {batch[xidx_key].shape}"

    # Verify original size metadata
    assert 'orig_height' in batch or 'original_height' in batch
    assert 'orig_width' in batch or 'original_width' in batch

    print("NaFlex dictionary format verified!")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {L}")
    print(f"  Patch dim: {batch['patches'].shape[2]}")


def test_postprocess_pipeline(test_tar_path, tmp_path):
    """Test postprocessing pipeline."""
    from vitok.pp import build_transform
    from vitok.data import create_dataloader
    from vitok.naflex_io import unpatchify, postprocess

    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True)

    # Build transform
    pp_string = "to_tensor|normalize(minus_one_to_one)|patchify(16, 256)"

    # Create dataloader
    loader = create_dataloader(
        source=test_tar_path,
        pp=pp_string,
        batch_size=4,
        num_workers=0,
        seed=42,
    )

    batch = next(iter(loader))

    # Reconstruct images using unpatchify
    images = unpatchify(batch, patch=16)

    # Verify output shape
    assert images.ndim == 4, f"Expected 4D tensor, got {images.ndim}D"
    assert images.shape[0] == batch['patches'].shape[0], "Batch size mismatch"
    assert images.shape[1] == 3, "Expected 3 channels"

    # Save some images
    for i, img in enumerate(images[:2]):
        # Convert from [-1, 1] to [0, 255]
        img_np = ((img.cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(output_dir / f"recon_{i:02d}.png")

    # Verify images were saved
    saved_files = list(output_dir.glob("*.png"))
    assert len(saved_files) > 0, "No images saved"

    print(f"Saved {len(saved_files)} images to {output_dir}")


def test_streaming_dataloader_iteration(test_tar_path):
    """Test that streaming dataloader can iterate multiple batches."""
    from vitok.pp import build_transform
    from vitok.data import create_dataloader

    pp_string = "to_tensor|normalize(minus_one_to_one)|patchify(16, 256)"

    loader = create_dataloader(
        source=test_tar_path,
        pp=pp_string,
        batch_size=2,
        num_workers=0,
        seed=42,
    )

    # Iterate multiple batches
    num_batches = 3
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if i >= num_batches - 1:
            break

    assert len(batches) == num_batches, f"Expected {num_batches} batches, got {len(batches)}"
    print(f"Successfully iterated {num_batches} batches")


if __name__ == "__main__":
    # Run tests manually
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = str(Path(tmp_dir) / "test_images.tar")
        create_test_tar(tar_path)

        print("=" * 60)
        print("Running test_naflex_dict_format...")
        print("=" * 60)
        test_naflex_dict_format(tar_path)

        print("\n" + "=" * 60)
        print("Running test_postprocess_pipeline...")
        print("=" * 60)
        test_postprocess_pipeline(tar_path, Path(tmp_dir))

        print("\n" + "=" * 60)
        print("Running test_streaming_dataloader_iteration...")
        print("=" * 60)
        test_streaming_dataloader_iteration(tar_path)

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
