"""Test data streaming and NaFlex dictionary format.

This test verifies:
1. Data streaming works with WebDataset
2. NaFlex dictionary contains required keys
3. Train/val images can be saved to folder

Run with: pytest tests/test_data_stream.py -v
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
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    from vitok.transforms import TransformCfg, build_transform
    from vitok.transforms.collate import patch_collate_fn
    from vitok.datasets.webdataset import WebDataset

    # Build transform
    cfg = TransformCfg(
        train=True,
        patch_size=16,
        max_tokens=256,
        max_size=256,
    )
    transform = build_transform(cfg)

    # Create dataset
    dataset = WebDataset(
        bucket_paths=[str(Path(test_tar_path).parent)],
        transform=transform,
        collate_fn=patch_collate_fn,
        batch_size=4,
        num_workers=0,
        seed=42,
        return_labels=True,
    )

    loader = dataset.create_dataloader()
    batch, labels = next(iter(loader))

    # Verify NaFlex dictionary keys
    required_keys = ['patches', 'ptype', 'yidx', 'xidx', 'attention_mask']
    for key in required_keys:
        assert key in batch, f"Missing key: {key}"

    # Verify shapes
    B = batch['patches'].shape[0]
    L = batch['patches'].shape[1]  # Sequence length

    assert batch['patches'].ndim == 3, "patches should be [B, L, C]"
    assert batch['ptype'].shape == (B, L), f"ptype shape mismatch: {batch['ptype'].shape}"
    assert batch['yidx'].shape == (B, L), f"yidx shape mismatch: {batch['yidx'].shape}"
    assert batch['xidx'].shape == (B, L), f"xidx shape mismatch: {batch['xidx'].shape}"

    # Attention mask can be None or [B, L, L]
    if batch['attention_mask'] is not None:
        assert batch['attention_mask'].shape[-2:] == (L, L)

    # Verify original size metadata
    assert 'original_height' in batch
    assert 'original_width' in batch

    print("NaFlex dictionary format verified!")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {L}")
    print(f"  Patch dim: {batch['patches'].shape[2]}")


def test_save_train_val_images(test_tar_path, tmp_path):
    """Test saving train/val images to folder."""
    from vitok.transforms import TransformCfg, build_transform
    from vitok.transforms.collate import patch_collate_fn
    from vitok.transforms.patch_ops import unpatchify
    from vitok.datasets.webdataset import WebDataset

    output_dir = tmp_path / "test_output"
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    # Train transform (with augmentation)
    train_cfg = TransformCfg(
        train=True,
        patch_size=16,
        max_tokens=256,
        max_size=256,
    )
    train_transform = build_transform(train_cfg)

    # Val transform (no augmentation)
    val_cfg = TransformCfg(
        train=False,
        patch_size=16,
        max_tokens=256,
        max_size=256,
    )
    val_transform = build_transform(val_cfg)

    # Create datasets
    tar_dir = str(Path(test_tar_path).parent)

    train_dataset = WebDataset(
        bucket_paths=[tar_dir],
        transform=train_transform,
        collate_fn=patch_collate_fn,
        batch_size=4,
        num_workers=0,
        seed=42,
    )

    val_dataset = WebDataset(
        bucket_paths=[tar_dir],
        transform=val_transform,
        collate_fn=patch_collate_fn,
        batch_size=4,
        num_workers=0,
        seed=42,
    )

    # Get batches
    train_loader = train_dataset.create_dataloader()
    val_loader = val_dataset.create_dataloader()

    train_batch, _ = next(iter(train_loader))
    val_batch, _ = next(iter(val_loader))

    # Reconstruct and save train images
    train_images = unpatchify(train_batch, patch=16, max_grid_size=64)

    for i, img in enumerate(train_images):
        if i >= 4:
            break
        # Convert from [-1, 1] to [0, 255]
        img_np = ((img.cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(train_dir / f"train_{i:02d}.png")

    # Reconstruct and save val images
    val_images = unpatchify(val_batch, patch=16, max_grid_size=64)

    for i, img in enumerate(val_images):
        if i >= 4:
            break
        img_np = ((img.cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(val_dir / f"val_{i:02d}.png")

    # Verify images were saved
    train_files = list(train_dir.glob("*.png"))
    val_files = list(val_dir.glob("*.png"))

    assert len(train_files) > 0, "No train images saved"
    assert len(val_files) > 0, "No val images saved"

    print(f"Saved {len(train_files)} train images to {train_dir}")
    print(f"Saved {len(val_files)} val images to {val_dir}")


def test_streaming_dataloader_iteration(test_tar_path):
    """Test that streaming dataloader can iterate multiple batches."""
    from vitok.transforms import TransformCfg, build_transform
    from vitok.transforms.collate import patch_collate_fn
    from vitok.datasets.webdataset import WebDataset

    cfg = TransformCfg(
        train=True,
        patch_size=16,
        max_tokens=256,
    )
    transform = build_transform(cfg)

    dataset = WebDataset(
        bucket_paths=[str(Path(test_tar_path).parent)],
        transform=transform,
        collate_fn=patch_collate_fn,
        batch_size=2,
        num_workers=0,
        seed=42,
    )

    loader = dataset.create_dataloader()

    # Iterate multiple batches
    num_batches = 3
    batches = []
    for i, (batch, labels) in enumerate(loader):
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
        print("Running test_save_train_val_images...")
        print("=" * 60)
        test_save_train_val_images(tar_path, Path(tmp_dir))

        print("\n" + "=" * 60)
        print("Running test_streaming_dataloader_iteration...")
        print("=" * 60)
        test_streaming_dataloader_iteration(tar_path)

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
