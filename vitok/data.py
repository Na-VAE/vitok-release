"""Data loading with preprocessing DSL.

Example:
    loader = create_dataloader(
        source="hf://imagenet-1k/train/*.tar",
        pp="random_resized_crop(512)|flip|to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)",
        batch_size=32,
    )
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.distributed as dist
import webdataset as wds
from PIL import Image, ImageOps

from vitok.pp import build_transform


def patch_collate_fn(batch):
    """Collate function for patchified data.

    Handles:
    - dicts from patchified pipelines
    - tensors from square image pipelines
    """
    if len(batch) == 0:
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


def create_dataloader(
    source: str,
    pp: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 0,
    prefetch_factor: int = 4,
    min_size: Optional[int] = None,
    shuffle_buffer: int = 100000,
) -> wds.WebLoader:
    """Create a dataloader from source with preprocessing.

    Args:
        source: Data source. Formats:
            - "hf://repo/subdir/*.tar" for HuggingFace Hub
            - "/path/to/*.tar" or "/path/to/dir" for local WebDataset
        pp: Preprocessing string, e.g.:
            "random_resized_crop(512)|flip|to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)"
        batch_size: Batch size
        num_workers: DataLoader workers
        seed: Random seed for shuffling
        prefetch_factor: Prefetch factor
        min_size: Optional minimum image size filter
        shuffle_buffer: Shuffle buffer size

    Returns:
        WebLoader yielding batch dicts
    """
    transform = build_transform(pp)

    if source.startswith("hf://"):
        # HuggingFace Hub streaming
        tar_files = _get_hf_shard_urls(source)
    else:
        # Local WebDataset
        tar_files = _get_local_tar_files(source)

    if not tar_files:
        raise ValueError(f"No tar files found for source: {source}")

    dataset = _build_webdataset_pipeline(
        tar_files=tar_files,
        transform=transform,
        batch_size=batch_size,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
        min_size=min_size,
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader


def _get_hf_shard_urls(source: str) -> List[str]:
    """Parse hf://repo/subdir/*.tar and get streaming URLs.

    Supports glob patterns like:
        hf://timm/imagenet-22k-wds/imagenet22k-train-{0000..0099}.tar  (brace expansion)
        hf://timm/imagenet-22k-wds/imagenet22k-train-00*.tar  (glob via HF API)
        hf://timm/imagenet-22k-wds/*.tar  (all shards)
    """
    import fnmatch
    import os
    import re

    try:
        from huggingface_hub import HfFileSystem, get_token
    except ImportError:
        raise ImportError("huggingface_hub is required for HuggingFace sources")

    if not source.startswith("hf://"):
        raise ValueError(f"Expected hf:// source, got: {source}")

    path = source[5:]  # Remove "hf://"

    # Get HF token for authentication
    hf_token = get_token()
    auth_header = f" -H 'Authorization:Bearer {hf_token}'" if hf_token else ""

    # Check for brace expansion pattern like {0000..0099}
    brace_match = re.search(r'\{(\d+)\.\.(\d+)\}', path)
    if brace_match:
        # Handle brace expansion directly without HF API call
        start = int(brace_match.group(1))
        end = int(brace_match.group(2))
        width = len(brace_match.group(1))

        # Extract repo and file pattern
        # path = "timm/imagenet-22k-wds/imagenet22k-train-{0000..0049}.tar"
        # prefix = "timm/imagenet-22k-wds/imagenet22k-train-"
        # suffix = ".tar"
        prefix = path[:brace_match.start()]
        suffix = path[brace_match.end():]

        # Find the last / before the brace to separate repo path from filename
        last_slash = prefix.rfind('/')
        if last_slash == -1:
            raise ValueError(f"Invalid hf:// source format: {source}")

        repo_path = prefix[:last_slash]  # "timm/imagenet-22k-wds"
        file_prefix = prefix[last_slash + 1:]  # "imagenet22k-train-"

        # Parse repo (org/repo) from repo_path
        repo_parts = repo_path.split('/')
        if len(repo_parts) >= 2:
            repo = '/'.join(repo_parts[:2])
            subdir = '/'.join(repo_parts[2:]) if len(repo_parts) > 2 else ''
        else:
            raise ValueError(f"Invalid hf:// source format: {source}")

        print(f"[Data] Brace expansion: repo={repo}, subdir={subdir}, file_prefix={file_prefix}, range={start}-{end}")

        shard_urls = []
        for i in range(start, end + 1):
            num_str = str(i).zfill(width)
            filename = f"{file_prefix}{num_str}{suffix}"
            if subdir:
                rel_path = f"{subdir}/{filename}"
            else:
                rel_path = filename
            url = f"pipe:curl -s -L --connect-timeout 30 --retry 3 https://huggingface.co/datasets/{repo}/resolve/main/{rel_path}{auth_header}"
            shard_urls.append(url)

        print(f"[Data] Generated {len(shard_urls)} shard URLs")
        return shard_urls

    # Parse for glob patterns (* or ?)
    parts = path.split("/")

    if len(parts) < 2:
        raise ValueError(f"Invalid hf:// source format: {source}")

    # Find pattern (last part with * or ?)
    pattern_idx = None
    for i, part in enumerate(parts):
        if "*" in part or "?" in part:
            pattern_idx = i
            break

    if pattern_idx is None:
        # No pattern, assume *.tar
        repo = "/".join(parts)
        subdir = None
        pattern = "*.tar"
    else:
        repo = "/".join(parts[:pattern_idx])
        pattern = parts[pattern_idx]
        subdir = "/".join(parts[pattern_idx + 1:]) if pattern_idx + 1 < len(parts) else None

    # Handle repo/subdir split (org/repo/subdir case)
    if "/" in repo and not any(c in repo for c in ["*", "?"]):
        repo_parts = repo.split("/")
        if len(repo_parts) > 2:
            repo = "/".join(repo_parts[:2])
            subdir = "/".join(repo_parts[2:])

    # Use HfFileSystem with timeout handling
    print(f"[Data] Listing HF shards for {repo} with pattern '{pattern}'...")
    fs = HfFileSystem()
    prefix = f"datasets/{repo}"
    if subdir:
        prefix = f"{prefix}/{subdir}"

    glob_pattern = f"{prefix}/{pattern}"
    shard_urls = []

    try:
        # Set a timeout for the glob operation
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"HfFileSystem.glob timed out after 60s for {glob_pattern}")

        # Only use signal on main thread (Unix)
        use_signal = hasattr(signal, 'SIGALRM') and os.name != 'nt'
        if use_signal:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout

        try:
            matches = fs.glob(glob_pattern)
        finally:
            if use_signal:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        print(f"[Data] Found {len(matches)} shards")

        for match in matches:
            # Build streaming URL with timeout and retry
            rel_path = match.replace(f"datasets/{repo}/", "")
            url = f"pipe:curl -s -L --connect-timeout 30 --retry 3 https://huggingface.co/datasets/{repo}/resolve/main/{rel_path}{auth_header}"
            shard_urls.append(url)
    except TimeoutError as e:
        raise ValueError(str(e))
    except Exception as e:
        raise ValueError(f"Failed to find shards at {glob_pattern}: {e}")

    return sorted(shard_urls)


def _get_local_tar_files(source: str) -> List[str]:
    """Get local tar files from path or glob pattern."""
    path = Path(source)

    if "*" in source:
        # Glob pattern
        parent = path.parent
        pattern = path.name
        if parent.exists():
            return sorted(str(f) for f in parent.glob(pattern))
        return []

    if path.is_file() and path.suffix == ".tar":
        return [str(path)]

    if path.is_dir():
        return sorted(str(f) for f in path.rglob("*.tar"))

    return []


def _build_webdataset_pipeline(
    tar_files: List[str],
    transform: Callable,
    batch_size: int,
    seed: int,
    shuffle_buffer: int,
    min_size: Optional[int],
):
    """Build the WebDataset pipeline."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Shuffle and split tar files by rank
    tar_rng = random.Random(seed)
    tar_files_copy = tar_files[:]
    tar_rng.shuffle(tar_files_copy)
    rank_tar_files = [tar_files_copy[i] for i in range(rank, len(tar_files_copy), world_size)]

    sample_shuffle_seed = seed + rank

    def has_image(sample: dict) -> bool:
        return any(k in sample for k in ("jpg", "jpeg", "png", "webp", "image"))

    def normalize_keys(sample: dict) -> dict:
        for key in ("jpg", "jpeg", "png", "webp"):
            if key in sample:
                sample["image"] = sample[key]
                return sample
        return sample

    def ensure_rgb(img: Image.Image) -> Image.Image:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        if img.mode == "P" and "transparency" in getattr(img, "info", {}):
            img = img.convert("RGBA")

        if img.mode in ("RGBA", "LA"):
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img).convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        return img

    def filter_size(sample: dict) -> bool:
        if min_size is None:
            return True
        w, h = sample["image"].size
        return min(w, h) >= min_size

    def process(sample: dict):
        image = sample["image"]
        return transform(image)

    dataset = (
        wds.WebDataset(
            rank_tar_files,
            handler=wds.ignore_and_continue,
            nodesplitter=wds.split_by_node,
            shardshuffle=False,
            resampled=True,
        )
        .shuffle(shuffle_buffer, seed=sample_shuffle_seed)
        .decode("pil", handler=wds.ignore_and_continue)
        .select(has_image)
        .map(normalize_keys)
        .map_dict(image=ensure_rgb)
        .select(filter_size)
        .map(process)
        .batched(batch_size, partial=False, collation_fn=patch_collate_fn)
    )
    return dataset


__all__ = ["create_dataloader", "patch_collate_fn"]
