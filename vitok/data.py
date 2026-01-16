"""Data loading with preprocessing DSL.

Supports three data source types:
- ImageFolder: directory of images (jpg, png, etc.)
- WebDataset: tar shards (local or hf:// URLs)
- HuggingFace streaming: stream from HuggingFace datasets
- Video: MP4/WebM files via WebDataset or folder

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

    # HuggingFace streaming (auto-detected by source name)
    loader = create_dataloader(
        source="coco",  # or "div8k", "nature", "portraits", etc.
        pp="resize_longest_side(512)|center_crop(512)|to_tensor",
        batch_size=32,
        num_samples=1000,
    )

    # Video WebDataset
    loader = create_dataloader(
        source="hf://org/video-dataset/videos-{0000..0099}.tar",
        pp="to_tensor|normalize(minus_one_to_one)|patchify(16, 2, 256)",
        batch_size=8,
        video_mode=True,
        max_frames=16,
        frame_stride=1,
    )

Note on drop_last behavior:
- ImageFolder: respects drop_last parameter
- WebDataset: always drops incomplete batches (partial=False is hardcoded)
- HuggingFace streaming: always yields complete batches except final batch
"""

from __future__ import annotations

import io
import random
import re
import warnings
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.distributed as dist
import webdataset as wds
from PIL import Image, ImageOps

from vitok.pp import build_transform

# Optional video decoding
try:
    import decord
    decord.bridge.set_bridge('torch')
    DECORD_AVAILABLE = True
except ImportError:
    decord = None
    DECORD_AVAILABLE = False


class DataSourceType(Enum):
    """Data source types supported by create_dataloader."""
    IMAGE_FOLDER = "image_folder"
    WEBDATASET = "webdataset"
    HF_STREAMING = "hf_streaming"


# HuggingFace datasets for streaming (dataset_name -> (repo, split, image_key))
HF_DATASETS = {
    "coco": ("detection-datasets/coco", "val", "image"),
    "div8k": ("Iceclear/DIV8K_TrainingSet", "train", "image"),
    "nature": ("eugenesiow/Div2k", "validation", "hr"),
    "portraits": ("jlbaker361/celebrity-100k", "train", "image"),
    "text": ("nielsr/funsd", "train", "image"),
    "architecture": ("GATE-engine/mini-Unsplash", "train", "image"),
    "animals": ("cats_vs_dogs", "train", "image"),
    # Blog visual categories (aliases + new)
    "foliage": ("eugenesiow/Div2k", "validation", "hr"),  # alias for nature
    "faces": ("nielsr/CelebA-faces", "train", "image"),  # close-up faces
    "urban": ("GATE-engine/mini-Unsplash", "train", "image"),  # alias for architecture
}


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


def decode_video(
    video_bytes: bytes,
    max_frames: int = 16,
    frame_stride: int = 1,
) -> torch.Tensor:
    """Decode video to tensor using decord.

    Args:
        video_bytes: Raw video file bytes (MP4, WebM, etc.)
        max_frames: Maximum number of frames to sample
        frame_stride: Sample every Nth frame

    Returns:
        Tensor of shape (T, C, H, W) where T <= max_frames, values in [0, 1]
    """
    if not DECORD_AVAILABLE:
        raise ImportError("decord is required for video loading. Install with: pip install decord")

    vr = decord.VideoReader(io.BytesIO(video_bytes), num_threads=4)
    total_frames = len(vr)

    # Sample frames uniformly with stride
    indices = list(range(0, total_frames, frame_stride))[:max_frames]

    if len(indices) == 0:
        indices = [0]  # At least get one frame

    # Get batch of frames: (T, H, W, C) in uint8
    frames = vr.get_batch(indices)

    # Convert to (T, C, H, W) float32 in [0, 1]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0

    return frames


def video_collate_fn(batch):
    """Collate video patch dicts into batched tensors.

    Same as patch_collate_fn but handles video-specific keys.
    """
    return patch_collate_fn(batch)


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
        transformed = self.transform(img)
        # Handle both tensor and dict outputs from transform
        if isinstance(transformed, dict):
            transformed["label"] = -1
            return transformed
        else:
            return {"image": transformed, "label": -1}


def _create_hf_streaming_loader(
    dataset_name: str,
    pp: str,
    batch_size: int,
    num_samples: int,
):
    """Create a dataloader that streams from HuggingFace.

    Args:
        dataset_name: Key from HF_DATASETS
        pp: Preprocessing string (applied to PIL images)
        batch_size: Batch size
        num_samples: Max number of samples to stream

    Returns:
        Iterator yielding batch dicts with 'image' tensor and 'label' key
    """
    from datasets import load_dataset

    repo, split, image_key = HF_DATASETS[dataset_name]
    ds = load_dataset(repo, split=split, streaming=True, trust_remote_code=True)
    transform = build_transform(pp)

    def batch_iterator():
        batch = []
        count = 0
        for example in ds:
            if count >= num_samples:
                break
            img = example[image_key]
            img = to_rgb(img)
            transformed = transform(img)
            # Handle both tensor and dict outputs from transform
            if isinstance(transformed, dict):
                transformed["label"] = -1
                batch.append(transformed)
            else:
                batch.append({"image": transformed, "label": -1})
            count += 1
            if len(batch) == batch_size:
                yield patch_collate_fn(batch)
                batch = []
        if batch:
            yield patch_collate_fn(batch)

    return batch_iterator()


def _create_video_webdataset_loader(
    source: str,
    pp: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle_buffer: int,
    max_frames: int,
    frame_stride: int,
):
    """Create a WebDataset loader for video files.

    Args:
        source: WebDataset source (hf:// URL or local path)
        pp: Preprocessing string for video tensors (T, C, H, W)
        batch_size: Batch size
        num_workers: DataLoader workers
        seed: Random seed
        shuffle_buffer: Shuffle buffer size
        max_frames: Maximum frames per video
        frame_stride: Sample every Nth frame

    Returns:
        WebLoader yielding batched video patch dicts
    """
    transform = build_transform(pp)
    urls = _resolve_source(source, seed)

    # Per-rank shuffle seed
    rank = dist.get_rank() if dist.is_initialized() else 0
    shuffle_seed = seed + rank

    def _decode_and_transform(sample):
        """Decode video and apply transform."""
        # Try different video extensions
        video_bytes = None
        for ext in ('mp4', 'webm', 'avi', 'mov', 'mkv'):
            if ext in sample:
                video_bytes = sample[ext]
                break

        if video_bytes is None:
            raise ValueError(f"No video file found in sample. Keys: {list(sample.keys())}")

        # Decode video to (T, C, H, W) tensor
        frames = decode_video(video_bytes, max_frames=max_frames, frame_stride=frame_stride)

        # Apply transform (expects tensor input)
        transformed = transform(frames)

        # Handle both tensor and dict outputs
        if isinstance(transformed, dict):
            transformed["label"] = -1
            return transformed
        else:
            return {"video": transformed, "label": -1}

    dataset = (
        wds.WebDataset(urls, resampled=True, handler=wds.ignore_and_continue)
        .shuffle(shuffle_buffer, seed=shuffle_seed)
        .map(_decode_and_transform, handler=wds.ignore_and_continue)
        .batched(batch_size, partial=False, collation_fn=video_collate_fn)
    )

    return wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def create_dataloader(
    source: str,
    pp: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 0,
    shuffle_buffer: int = 10000,
    min_size: Optional[int] = None,
    drop_last: bool = True,
    num_samples: Optional[int] = None,
    # Video mode parameters
    video_mode: bool = False,
    max_frames: int = 16,
    frame_stride: int = 1,
):
    """Create a dataloader from source with preprocessing.

    Auto-detects source type:
        - HuggingFace streaming: if source is a dataset name (e.g., "coco", "div8k")
        - Image folder: directory containing images (jpg, png, etc.)
        - WebDataset: tar shards, glob patterns, or hf:// URLs

    Args:
        source: Data source:
            - Dataset name (e.g., "coco", "div8k") for HuggingFace streaming
            - "hf://org/repo/pattern.tar" for HuggingFace Hub WebDataset
            - "/path/to/*.tar" for local WebDataset shards
            - "/path/to/image_folder" for a folder of images
        pp: Preprocessing string, e.g.:
            "resize_longest_side(512)|to_tensor|normalize(minus_one_to_one)|patchify(16, 256)"
            For video: "normalize(minus_one_to_one)|patchify(16, 2, 256)"
        batch_size: Batch size
        num_workers: DataLoader workers
        seed: Random seed (used for shard assignment, must be same across ranks)
        shuffle_buffer: Shuffle buffer size (WebDataset only)
        min_size: Optional minimum image dimension filter (WebDataset only)
        drop_last: Drop last incomplete batch (useful for multi-GPU to ensure consistent batch sizes)
        num_samples: Max samples (required for HF streaming, optional otherwise)
        video_mode: If True, load video files instead of images
        max_frames: Maximum frames to sample per video (video_mode only)
        frame_stride: Sample every Nth frame (video_mode only)

    Returns:
        DataLoader yielding batch dicts with 'label' key (class label, -1 if unavailable)
    """
    # Video mode: use video WebDataset loader
    if video_mode:
        return _create_video_webdataset_loader(
            source=source,
            pp=pp,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
            max_frames=max_frames,
            frame_stride=frame_stride,
        )

    # HuggingFace streaming (auto-detected by source name)
    if source in HF_DATASETS:
        if num_samples is None:
            raise ValueError(f"num_samples is required for streaming dataset '{source}'")
        return _create_hf_streaming_loader(source, pp, batch_size, num_samples)

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
        transformed = transform(s["image"])
        label = _decode_label(s.get("cls") or s.get("cls.txt"))
        # Handle both tensor and dict outputs from transform
        if isinstance(transformed, dict):
            transformed["label"] = label
            return transformed
        else:
            return {"image": transformed, "label": label}

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
    """Convert source string to list of URLs for WebDataset.

    Supports comma-separated sources for mixing multiple datasets:
        "hf://repo1/data-{0000..0099}.tar,hf://repo2/data-{0000..0049}.tar"
    """
    # Handle comma-separated multiple sources
    if ',' in source:
        all_urls = []
        for s in source.split(','):
            s = s.strip()
            if s.startswith("hf://"):
                all_urls.extend(_hf_to_urls(s, seed))
            else:
                all_urls.extend(_local_to_urls(s, seed))
        # Shuffle combined URLs
        rng = random.Random(seed)
        rng.shuffle(all_urls)
        return all_urls

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
    "video_collate_fn",
    "decode_video",
    "to_rgb",
    "HF_DATASETS",
    "DECORD_AVAILABLE",
]
