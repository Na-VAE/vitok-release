"""Data loading with preprocessing DSL.

Example:
    loader = create_dataloader(
        source="hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0099}.tar",
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
        else:
            collated[k] = items
    return collated


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


def create_dataloader(
    source: str,
    pp: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 0,
    shuffle_buffer: int = 10000,
    min_size: Optional[int] = None,
) -> wds.WebLoader:
    """Create a dataloader from source with preprocessing.

    Args:
        source: Data source:
            - "hf://org/repo/pattern.tar" for HuggingFace Hub
            - "/path/to/*.tar" for local WebDataset
        pp: Preprocessing string, e.g.:
            "resize_longest_side(512)|to_tensor|normalize(minus_one_to_one)|patchify(16, 256)"
        batch_size: Batch size
        num_workers: DataLoader workers
        seed: Random seed (used for shard assignment, must be same across ranks)
        shuffle_buffer: Shuffle buffer size
        min_size: Optional minimum image dimension filter

    Returns:
        WebLoader yielding batch dicts
    """
    transform = build_transform(pp)
    urls = _resolve_source(source, seed)

    # Per-rank shuffle seed: different ranks get different sample orderings
    # Even with same seed, ranks process different shards so samples differ
    rank = dist.get_rank() if dist.is_initialized() else 0
    shuffle_seed = seed + rank

    # Build pipeline
    dataset = (
        wds.WebDataset(urls, resampled=True, handler=wds.ignore_and_continue)
        .shuffle(shuffle_buffer, seed=shuffle_seed)
        .decode("pil", handler=wds.ignore_and_continue)
        .rename(image="jpg;jpeg;png;webp")
        .map_dict(image=to_rgb)
    )

    if min_size is not None:
        dataset = dataset.select(lambda s: min(s["image"].size) >= min_size)

    dataset = (
        dataset
        .map(lambda s: transform(s["image"]))
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

    Note: seed should be the SAME across all ranks for consistent shard assignment.
    The shuffle ensures different epoch orderings when seed changes.
    """
    from huggingface_hub import get_token

    path = source[5:]  # Remove "hf://"
    token = get_token()
    auth = f" -H 'Authorization:Bearer {token}'" if token else ""

    # Check for brace expansion {start..end}
    match = re.search(r'\{(\d+)\.\.(\d+)\}', path)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        width = len(match.group(1))
        prefix, suffix = path[:match.start()], path[match.end():]

        # Extract repo from path (org/repo/subdir/file -> org/repo)
        parts = prefix.split('/')
        repo = '/'.join(parts[:2])
        subpath = '/'.join(parts[2:])

        urls = []
        for i in range(start, end + 1):
            num = str(i).zfill(width)
            rel = f"{subpath}{num}{suffix}"
            urls.append(f"pipe:curl -sL https://huggingface.co/datasets/{repo}/resolve/main/{rel}{auth}")

        # Shuffle with SAME seed across all ranks for consistent assignment
        rng = random.Random(seed)
        rng.shuffle(urls)

        # Split by rank - each rank gets non-overlapping shards
        if dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
            urls = urls[rank::world]

        return urls

    # No brace expansion - single URL or let WebDataset handle it
    parts = path.split('/')
    repo = '/'.join(parts[:2])
    rel = '/'.join(parts[2:])
    return [f"pipe:curl -sL https://huggingface.co/datasets/{repo}/resolve/main/{rel}{auth}"]


def _local_to_urls(source: str, seed: int = 0) -> list[str]:
    """Resolve local path to list of tar files.

    Note: seed should be the SAME across all ranks for consistent shard assignment.
    """
    path = Path(source)

    if "*" in source or "?" in source:
        urls = sorted(str(f) for f in path.parent.glob(path.name))
    elif path.is_dir():
        urls = sorted(str(f) for f in path.rglob("*.tar"))
    else:
        urls = [str(path)]

    # Shuffle with SAME seed across all ranks for consistent assignment
    rng = random.Random(seed)
    rng.shuffle(urls)

    # Split by rank - each rank gets non-overlapping shards
    if dist.is_initialized():
        rank, world = dist.get_rank(), dist.get_world_size()
        urls = urls[rank::world]

    return urls


__all__ = ["create_dataloader", "patch_collate_fn", "to_rgb"]
