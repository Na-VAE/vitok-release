"""Data loading with preprocessing DSL.

Example:
    # WebDataset (tar shards) - auto-detected
    loader = create_dataloader(
        source="hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0099}.tar",
        pp="resize_longest_side(512)|to_tensor|normalize(minus_one_to_one)|patchify(16, 256)",
        batch_size=32,
    )

    # Image folder - auto-detected
    loader = create_dataloader(
        source="/path/to/images",
        pp="resize_longest_side(512)|to_tensor|normalize(minus_one_to_one)|patchify(16, 256)",
        batch_size=32,
    )
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.distributed as dist
import webdataset as wds
from PIL import Image, ImageOps

from vitok.pp import build_transform


def patch_collate_fn(batch):
    """Collate patch dicts into batched tensors."""
    if not batch:
        return {}

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, dim=0)

    collated = {}
    for k in batch[0].keys():
        items = [d[k] for d in batch]
        if isinstance(items[0], torch.Tensor):
            collated[k] = torch.stack(items, dim=0)
        elif isinstance(items[0], (int, float)):
            collated[k] = torch.tensor(items)
        else:
            collated[k] = items
    return collated


def _decode_label(value):
    """Parse class labels from WebDataset entries."""
    if value is None:
        return -1
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return -1
        return int(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.item())
        return value.to(dtype=torch.long)
    return int(value)


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert image to RGB, handling transparency and EXIF rotation."""
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode == "P" and "transparency" in getattr(img, "info", {}):
        img = img.convert("RGBA")

    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg.convert("RGBA"), img.convert("RGBA")).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img


def _is_image_folder(source: str) -> bool:
    """Check if source is a folder containing images (not tar files)."""
    path = Path(source)
    if not path.is_dir():
        return False
    # Check if it has image files but no tar files
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    has_images = any(f.suffix.lower() in image_exts for f in path.iterdir() if f.is_file())
    has_tars = any(f.suffix.lower() == ".tar" for f in path.iterdir() if f.is_file())
    return has_images and not has_tars


class ImageFolderDataset(torch.utils.data.Dataset):
    """Simple dataset for a folder of images."""

    def __init__(self, root: str, transform: Callable, seed: int = 0):
        self.root = Path(root)
        self.transform = transform
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        self.files = sorted(
            f for f in self.root.rglob("*")
            if f.is_file() and f.suffix.lower() in image_exts
        )
        # Shuffle with seed
        rng = random.Random(seed)
        rng.shuffle(self.files)

        # Split by rank
        if dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
            self.files = self.files[rank::world]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path)
        img = to_rgb(img)
        patch_dict = self.transform(img)
        patch_dict["label"] = -1
        return patch_dict


def create_dataloader(
    source: str,
    pp: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 0,
    shuffle_buffer: int = 10000,
    min_size: Optional[int] = None,
    drop_last: bool = False,
):
    """Create a dataloader from source with preprocessing.

    Auto-detects source type:
        - Image folder: directory containing images (jpg, png, etc.)
        - WebDataset: tar shards, glob patterns, or hf:// URLs

    Args:
        source: Data source:
            - "hf://org/repo/pattern.tar" for HuggingFace Hub
            - "/path/to/*.tar" for local WebDataset shards
            - "/path/to/image_folder" for a folder of images
        pp: Preprocessing string, e.g.:
            "resize_longest_side(512)|to_tensor|normalize(minus_one_to_one)|patchify(16, 256)"
        batch_size: Batch size
        num_workers: DataLoader workers
        seed: Random seed (used for shard assignment, must be same across ranks)
        shuffle_buffer: Shuffle buffer size (WebDataset only)
        min_size: Optional minimum image dimension filter (WebDataset only)
        drop_last: Drop last incomplete batch (useful for multi-GPU to ensure consistent batch sizes)

    Returns:
        DataLoader yielding batch dicts with 'label' key (class label, -1 if unavailable)
    """
    # Auto-detect: image folder vs WebDataset
    if _is_image_folder(source):
        transform = build_transform(pp)
        dataset = ImageFolderDataset(source, transform, seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Already shuffled in dataset
            num_workers=num_workers,
            collate_fn=patch_collate_fn,
            pin_memory=True,
            drop_last=drop_last,
        )

    # WebDataset path
    transform = build_transform(pp)
    urls = _resolve_source(source, seed)

    # Per-rank shuffle seed: different ranks get different sample orderings
    rank = dist.get_rank() if dist.is_initialized() else 0
    shuffle_seed = seed + rank

    def _transform_sample(s):
        """Transform image and include label in output dict."""
        patch_dict = transform(s["image"])
        label = s.get("cls") or s.get("cls.txt")
        patch_dict["label"] = _decode_label(label)
        return patch_dict

    dataset = (
        wds.WebDataset(urls, resampled=True, handler=wds.ignore_and_continue)
        .shuffle(shuffle_buffer, seed=shuffle_seed)
        .decode("pil", handler=wds.ignore_and_continue)
        .rename(image="jpg;jpeg;png;webp", keep=True)
        .map_dict(image=to_rgb)
    )

    if min_size is not None:
        dataset = dataset.select(lambda s: min(s["image"].size) >= min_size)

    dataset = (
        dataset
        .map(_transform_sample)
        .batched(batch_size, partial=False, collation_fn=patch_collate_fn)
    )

    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _resolve_source(source: str, seed: int = 0) -> list[str]:
    """Convert source string to list of URLs for WebDataset."""
    if source.startswith("hf://"):
        return _hf_to_urls(source, seed)
    return _local_to_urls(source, seed)


def _hf_to_urls(source: str, seed: int = 0) -> list[str]:
    """Convert hf://org/repo/pattern.tar to curl URLs.

    Supports brace expansion: hf://org/repo/data-{0000..0099}.tar
    """
    from huggingface_hub import get_token

    path = source[5:]  # Remove "hf://"
    token = get_token()
    auth = f" -H 'Authorization:Bearer {token}'" if token else ""

    match = re.search(r'\{(\d+)\.\.(\d+)\}', path)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        width = len(match.group(1))
        prefix, suffix = path[:match.start()], path[match.end():]

        parts = prefix.split('/')
        repo = '/'.join(parts[:2])
        subpath = '/'.join(parts[2:])

        urls = []
        for i in range(start, end + 1):
            num = str(i).zfill(width)
            rel = f"{subpath}{num}{suffix}"
            urls.append(f"pipe:curl -sL https://huggingface.co/datasets/{repo}/resolve/main/{rel}{auth}")

        rng = random.Random(seed)
        rng.shuffle(urls)

        if dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
            urls = urls[rank::world]

        return urls

    parts = path.split('/')
    repo = '/'.join(parts[:2])
    rel = '/'.join(parts[2:])
    return [f"pipe:curl -sL https://huggingface.co/datasets/{repo}/resolve/main/{rel}{auth}"]


def _local_to_urls(source: str, seed: int = 0) -> list[str]:
    """Resolve local path to list of tar files."""
    path = Path(source)

    if "*" in source or "?" in source:
        urls = sorted(str(f) for f in path.parent.glob(path.name))
    elif path.is_dir():
        urls = sorted(str(f) for f in path.rglob("*.tar"))
    else:
        urls = [str(path)]

    rng = random.Random(seed)
    rng.shuffle(urls)

    if dist.is_initialized():
        rank, world = dist.get_rank(), dist.get_world_size()
        urls = urls[rank::world]

    return urls


__all__ = [
    "create_dataloader",
    "ImageFolderDataset",
    "patch_collate_fn",
    "to_rgb",
]
