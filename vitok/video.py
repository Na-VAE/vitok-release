"""Video frame extraction utilities.

Extract frames from video files or image sequences for processing
through the ViTok pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from PIL import Image


def extract_frames(
    source: Union[str, Path, List[str], List[Path]],
    max_frames: Optional[int] = None,
    temporal_stride: int = 1,
    start_frame: int = 0,
) -> List[Image.Image]:
    """Extract frames from video file or image sequence.

    Args:
        source: One of:
            - Video file path (.mp4, .webm, .avi, .mov)
            - Directory containing images
            - List of image file paths
        max_frames: Maximum frames to extract (None = all available)
        temporal_stride: Sample every Nth frame (1 = all frames).
            Note: Only stride=1 is currently implemented.
        start_frame: Frame index to start extraction from

    Returns:
        List of PIL Images in RGB format

    Raises:
        ValueError: If source type cannot be determined or stride != 1
        FileNotFoundError: If source path doesn't exist

    Example:
        >>> frames = extract_frames("video.mp4", max_frames=8)
        >>> frames = extract_frames("frames_dir/", max_frames=16)
        >>> frames = extract_frames(["img1.png", "img2.png", "img3.png"])
    """
    if temporal_stride != 1:
        raise NotImplementedError(
            f"temporal_stride={temporal_stride} not yet implemented. Use stride=1."
        )

    source = Path(source) if isinstance(source, str) else source

    # Handle list of paths
    if isinstance(source, list):
        return _load_image_sequence(
            [Path(p) for p in source], max_frames, start_frame
        )

    # Handle single path
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if source.is_dir():
        return _load_image_directory(source, max_frames, start_frame)

    if _is_video_file(source):
        return _load_video_file(source, max_frames, start_frame)

    # Single image file
    return _load_image_sequence([source], max_frames, start_frame)


def _is_video_file(path: Path) -> bool:
    """Check if path is a video file based on extension."""
    video_extensions = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    return path.suffix.lower() in video_extensions


def _load_video_file(
    path: Path,
    max_frames: Optional[int],
    start_frame: int,
) -> List[Image.Image]:
    """Load frames from video file using torchvision.

    Args:
        path: Path to video file
        max_frames: Maximum frames to load
        start_frame: Frame index to start from

    Returns:
        List of PIL Images
    """
    try:
        import torchvision.io as io
    except ImportError:
        raise ImportError(
            "torchvision is required for video loading. "
            "Install with: pip install torchvision"
        )

    # Read video - returns (T, H, W, C) tensor and audio/info
    video_tensor, _, info = io.read_video(str(path), pts_unit="sec")

    # Apply start_frame offset
    if start_frame > 0:
        video_tensor = video_tensor[start_frame:]

    # Limit frames
    if max_frames is not None and len(video_tensor) > max_frames:
        video_tensor = video_tensor[:max_frames]

    # Convert to PIL Images
    frames = []
    for i in range(len(video_tensor)):
        frame_np = video_tensor[i].numpy()
        frames.append(Image.fromarray(frame_np).convert("RGB"))

    return frames


def _load_image_directory(
    directory: Path,
    max_frames: Optional[int],
    start_frame: int,
) -> List[Image.Image]:
    """Load images from directory in sorted order.

    Supports common image formats: png, jpg, jpeg, webp, bmp, gif.

    Args:
        directory: Directory containing images
        max_frames: Maximum images to load
        start_frame: Index to start from (after sorting)

    Returns:
        List of PIL Images
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

    # Find all image files, sorted for consistent ordering
    image_paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in image_extensions
    )

    return _load_image_sequence(image_paths, max_frames, start_frame)


def _load_image_sequence(
    paths: List[Path],
    max_frames: Optional[int],
    start_frame: int,
) -> List[Image.Image]:
    """Load a sequence of image files.

    Args:
        paths: List of image file paths
        max_frames: Maximum images to load
        start_frame: Index to start from

    Returns:
        List of PIL Images in RGB format
    """
    # Apply start offset
    if start_frame > 0:
        paths = paths[start_frame:]

    # Limit frames
    if max_frames is not None:
        paths = paths[:max_frames]

    # Load images
    frames = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        frames.append(img)

    return frames


__all__ = ["extract_frames"]
