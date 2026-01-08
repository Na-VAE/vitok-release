"""Image and video preprocessing and postprocessing utilities.

Simple high-level API for converting images/videos to/from patch dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from vitok.pp.registry import build_transform
from vitok.pp.ops import unpatchify, unpack
from vitok.data import patch_collate_fn


def preprocess(
    images: Union[Image.Image, List[Image.Image]],
    pp: str = "to_tensor|normalize(minus_one_to_one)|patchify(16, 256)",
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Preprocess PIL images into patch dictionary.

    Args:
        images: PIL Image(s) - must be PIL, not tensors
        pp: Preprocessing string, e.g.:
            "resize_longest_side(512)|to_tensor|normalize(minus_one_to_one)|patchify(16, 256)"
        device: Target device

    Returns:
        Batched patch dictionary with keys:
            - patches: [B, N, D] flattened patch pixels
            - patch_mask: [B, N] valid patch mask
            - row_idx, col_idx: [B, N] spatial indices
            - attention_mask: [B, N, N]
            - orig_height, orig_width: [B]

    Example:
        patches = preprocess([img1, img2], device="cuda")
    """
    if not isinstance(images, (list, tuple)):
        images = [images]

    transform = build_transform(pp)
    patch_dicts = [transform(img) for img in images]

    batched = patch_collate_fn(patch_dicts)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batched.items()}


def postprocess(
    output: Union[torch.Tensor, Dict[str, torch.Tensor]],
    output_format: str = "minus_one_to_one",
    current_format: str = "minus_one_to_one",
    do_unpack: bool = True,
    patch: int = 16,
    max_grid_size: Optional[int] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Postprocess model output into images.

    Args:
        output: Image tensor (B,C,H,W) or patch dict with 'patches' or 'images'
        output_format: Target format ("minus_one_to_one", "zero_to_one", "0_255")
        current_format: Current format of the output
        do_unpack: Whether to crop to original sizes (requires dict input)
        patch: Patch size for unpatchify
        max_grid_size: Maximum grid size for unpatchify

    Returns:
        Images tensor or list of tensors (if do_unpack=True)
    """
    if isinstance(output, dict):
        if 'images' in output:
            images = output['images']
        elif 'patches' in output:
            images = unpatchify(output, patch=patch, max_grid_size=max_grid_size)
        else:
            raise KeyError("Expected 'images' or 'patches' in output dict")
    else:
        images = output

    images = _convert_format(images, current_format, output_format)

    if do_unpack and isinstance(output, dict):
        orig_h = output.get('orig_height')
        orig_w = output.get('orig_width')
        if orig_h is None or orig_w is None:
            raise ValueError("do_unpack=True requires 'orig_height' and 'orig_width' in output")
        return unpack(images, orig_h, orig_w)

    return images


def _convert_format(images: torch.Tensor, from_format: str, to_format: str) -> torch.Tensor:
    """Convert between image formats.

    Clamps output to valid ranges to handle interpolation overshoot.
    """
    if from_format == to_format:
        return images

    if to_format == "minus_one_to_one":
        if from_format == "0_255":
            result = images.float() / 127.5 - 1.0
        elif from_format == "zero_to_one":
            result = images * 2.0 - 1.0
        else:
            return images
        return result.clamp(-1.0, 1.0)
    elif to_format == "zero_to_one":
        if from_format == "0_255":
            result = images.float() / 255.0
        elif from_format == "minus_one_to_one":
            result = (images + 1.0) / 2.0
        else:
            return images
        return result.clamp(0.0, 1.0)
    elif to_format == "0_255":
        if from_format == "minus_one_to_one":
            return ((images.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255).round().to(torch.uint8)
        elif from_format == "zero_to_one":
            return (images.clamp(0.0, 1.0) * 255).round().to(torch.uint8)

    return images


# Aliases for backwards compatibility
preprocess_images = preprocess
postprocess_images = postprocess


# =============================================================================
# Video Preprocessing
# =============================================================================


def preprocess_video(
    source: Union[str, Path, List[Image.Image], List[str], List[Path]],
    pp: str = "to_tensor|normalize(minus_one_to_one)|patchify(16, 256)",
    max_frames: Optional[int] = None,
    temporal_stride: int = 1,
    mode: str = "batch",
    device: str = "cuda",
    # Sequence mode params (TODO)
    max_tokens_per_frame: int = 256,
    max_total_tokens: int = 1024,
    cross_frame_attention: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocess video into patch dictionary.

    Extracts frames from video file or image sequence and processes them
    through the naflex pipeline.

    Args:
        source: One of:
            - Video file path (.mp4, .webm, etc.)
            - Directory of images
            - List of image paths
            - List of PIL Images
        pp: Preprocessing pipeline string
        max_frames: Maximum frames to process (None = all)
        temporal_stride: Sample every Nth frame (currently only stride=1)
        mode: Processing mode:
            - "batch": Each frame is a separate batch item (default)
            - "sequence": All frames in one sequence (TODO)
        device: Target device ("cuda" or "cpu")
        max_tokens_per_frame: Max patches per frame (sequence mode only)
        max_total_tokens: Total token budget (sequence mode only)
        cross_frame_attention: Allow attention across frames (sequence mode only)

    Returns:
        Batched patch dictionary with keys:
            - patches: [B, N, D] where B = num_frames (batch mode)
            - patch_mask: [B, N]
            - row_idx, col_idx: [B, N]
            - time_idx: [B, N] (all zeros in batch mode)
            - attention_mask: [B, N, N]
            - orig_height, orig_width: [B]

    Example:
        >>> batch = preprocess_video("video.mp4", max_frames=8)
        >>> batch = preprocess_video(["frame1.png", "frame2.png"])
    """
    from vitok.video import extract_frames

    # Extract frames if not already PIL Images
    if isinstance(source, list) and len(source) > 0 and isinstance(source[0], Image.Image):
        frames = source
    else:
        frames = extract_frames(source, max_frames, temporal_stride)

    if mode == "batch":
        # Simple! Just use existing preprocess with list of frames
        return preprocess(frames, pp=pp, device=device)

    elif mode == "sequence":
        # TODO: Implement sequence mode
        # This needs more design thought around:
        # - Token budgeting per frame
        # - Attention mask construction (cross_frame_attention)
        # - 3D positional encoding
        raise NotImplementedError(
            "Sequence mode not yet implemented. Use mode='batch' for now."
        )

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'batch' or 'sequence'.")


def postprocess_video(
    output: Dict[str, torch.Tensor],
    mode: str = "batch",
    patch: int = 16,
    output_format: str = "0_255",
    output_path: Optional[str] = None,
    fps: int = 30,
) -> Union[List[Image.Image], List[List[Image.Image]], str]:
    """Convert model output back to video frames.

    Handles both single video and batched video outputs automatically.
    If output contains 'frames_per_video' (from video_collate_fn), returns
    List[List[PIL.Image]] with one list per video. Otherwise returns List[PIL.Image].

    Args:
        output: Model output dict with 'patches' key
        mode: "batch" (implemented) or "sequence" (TODO)
        patch: Patch size for unpatchify
        output_format: Target format for pixels ("0_255" for video writing)
        output_path: If provided, write video file (.mp4, .gif) and return path.
            Only works for single video output.
        fps: Frames per second for video output

    Returns:
        - Single video: List[PIL.Image] or path string if output_path provided
        - Batched videos: List[List[PIL.Image]] (one list per video)

    Example:
        >>> frames = postprocess_video(decoded)
        >>> path = postprocess_video(decoded, output_path="output.mp4", fps=30)
    """
    if mode != "batch":
        raise NotImplementedError(
            "Sequence mode postprocessing not yet implemented. Use mode='batch'."
        )

    # Get list of frame tensors [C, H, W] each
    frame_tensors = postprocess(
        output, output_format="0_255", do_unpack=True, patch=patch
    )

    # Convert to PIL Images
    all_frames = [
        Image.fromarray(t.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        for t in frame_tensors
    ]

    # Check if this is batched output (has frames_per_video from video_collate_fn)
    frames_per_video = output.get("frames_per_video")

    if frames_per_video is not None:
        # Batched output - split by video boundaries
        videos = []
        offset = 0
        for num_frames in frames_per_video:
            videos.append(all_frames[offset : offset + num_frames])
            offset += num_frames
        return videos

    # Single video output
    if output_path is not None:
        return _write_video(all_frames, output_path, fps)

    return all_frames


def _write_video(frames: List[Image.Image], path: str, fps: int) -> str:
    """Write frames to video file.

    Args:
        frames: List of PIL Images
        path: Output path (.mp4, .gif, etc.)
        fps: Frames per second

    Returns:
        The output path
    """
    if path.endswith(".gif"):
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "imageio is required for GIF output. Install with: pip install imageio"
            )
        imageio.mimsave(path, [np.array(f) for f in frames], fps=fps)
    else:
        # .mp4, .avi, etc - use torchvision
        try:
            import torchvision.io as io
        except ImportError:
            raise ImportError(
                "torchvision is required for video output. "
                "Install with: pip install torchvision"
            )
        # write_video expects [T, H, W, C] uint8 tensor
        video_tensor = torch.stack(
            [torch.from_numpy(np.array(f)) for f in frames]
        )  # [T, H, W, C]
        io.write_video(path, video_tensor, fps=fps)

    return path


def video_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate multiple video patch dicts into a single batch.

    This function concatenates frames from multiple videos along the batch
    dimension and tracks video boundaries for later splitting.

    Args:
        batch: List of patch dicts, each from preprocess_video().
            Each dict has patches shape [T, N, D] where T is num_frames.

    Returns:
        Collated dict with:
            - patches: [sum(T), N, D] - all frames concatenated
            - patch_mask: [sum(T), N]
            - row_idx, col_idx, time_idx: [sum(T), N]
            - orig_height, orig_width: [sum(T)]
            - video_idx: [sum(T)] - which video each frame belongs to
            - frames_per_video: List[int] - frame counts for splitting output

    Example:
        >>> video1 = preprocess_video(path1)  # [T1, N, D]
        >>> video2 = preprocess_video(path2)  # [T2, N, D]
        >>> batch = video_collate_fn([video1, video2])  # [T1+T2, N, D]
        >>> # After model forward...
        >>> decoded['frames_per_video'] = batch['frames_per_video']
        >>> videos = postprocess_video(decoded)  # List of 2 video frame lists
    """
    all_patches = []
    all_masks = []
    all_row_idx = []
    all_col_idx = []
    all_time_idx = []
    all_orig_h = []
    all_orig_w = []
    video_idx = []
    frames_per_video = []

    for vid_i, video_dict in enumerate(batch):
        T = video_dict["patches"].shape[0]  # num frames in this video
        all_patches.append(video_dict["patches"])
        all_masks.append(video_dict["patch_mask"])
        all_row_idx.append(video_dict["row_idx"])
        all_col_idx.append(video_dict["col_idx"])
        all_time_idx.append(
            video_dict.get("time_idx", torch.zeros_like(video_dict["row_idx"]))
        )
        all_orig_h.append(video_dict["orig_height"])
        all_orig_w.append(video_dict["orig_width"])
        video_idx.append(torch.full((T,), vid_i, dtype=torch.long))
        frames_per_video.append(T)

    result = {
        "patches": torch.cat(all_patches, dim=0),
        "patch_mask": torch.cat(all_masks, dim=0),
        "row_idx": torch.cat(all_row_idx, dim=0),
        "col_idx": torch.cat(all_col_idx, dim=0),
        "time_idx": torch.cat(all_time_idx, dim=0),
        "orig_height": torch.cat(all_orig_h, dim=0),
        "orig_width": torch.cat(all_orig_w, dim=0),
        "video_idx": torch.cat(video_idx, dim=0),
        "frames_per_video": frames_per_video,  # List, not tensor
    }

    # Concatenate attention masks if present
    # Attention masks are [B, N, N] per video, need to create block-diagonal
    has_attn_mask = "attention_mask" in batch[0]
    if has_attn_mask:
        # Collect all 2D attention masks (flatten batch dimension)
        all_attn_2d = []
        for video_dict in batch:
            attn = video_dict["attention_mask"]  # [T, N, N] for T frames
            for t in range(attn.shape[0]):
                all_attn_2d.append(attn[t])  # [N, N]
        # Create block-diagonal from all 2D masks
        result["attention_mask"] = torch.block_diag(*all_attn_2d)

    return result


__all__ = [
    "preprocess",
    "postprocess",
    "preprocess_images",
    "postprocess_images",
    "preprocess_video",
    "postprocess_video",
    "video_collate_fn",
]
