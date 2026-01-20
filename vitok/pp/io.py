"""Image preprocessing and postprocessing utilities.

Simple high-level API for converting images to/from patch dictionaries.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

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
        output: Either a patch dict with 'patches' key, or an image tensor [B, C, H, W]
        output_format: Target format ("minus_one_to_one", "zero_to_one", "0_255")
        current_format: Current format of the output
        do_unpack: Whether to crop to original sizes (requires dict input with orig_height/width)
        patch: Patch size for unpatchify (only used for patch dicts)
        max_grid_size: Maximum grid size for unpatchify (only used for patch dicts)

    Returns:
        Images tensor [B, C, H, W] or list of tensors (if do_unpack=True with patch dict)
    """
    # Handle both patch dicts and raw image tensors
    if isinstance(output, torch.Tensor):
        # Already an image tensor [B, C, H, W], just convert format
        return _convert_format(output, current_format, output_format)

    # Patch dict - unpatchify first
    images = unpatchify(output, patch=patch, max_grid_size=max_grid_size)
    images = _convert_format(images, current_format, output_format)
    if do_unpack:
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


__all__ = [
    "preprocess",
    "postprocess",
    "preprocess_images",
    "postprocess_images",
]
