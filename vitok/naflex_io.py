"""Image preprocessing and postprocessing utilities.

Simple high-level API for converting images to/from patch dictionaries.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vitok.pp import build_transform
from vitok.data import patch_collate_fn


def preprocess_images(
    images: Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]],
    pp: str = "to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)",
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Preprocess images into the patch dictionary expected by models.

    Args:
        images: PIL Image(s) or tensor(s)
        pp: Preprocessing string, e.g.:
            "to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)"
        device: Target device

    Returns:
        Batched patch dictionary with keys:
            - patches: [B, N, D] flattened patch pixels
            - ptype: [B, N] patch type mask
            - yidx, xidx: [B, N] spatial indices
            - attention_mask: [B, N, N]
            - original_height, original_width: [B]

    Example:
        patches = preprocess_images(
            [img1, img2],
            pp="to_tensor|normalize(minus_one_to_one)|patchify(512, 16, 256)",
            device="cuda",
        )
    """
    if not isinstance(images, (list, tuple)):
        images = [images]

    transform = build_transform(pp)

    patch_dicts = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = _tensor_to_pil(img)
        patch_dicts.append(transform(img))

    batched_dict = patch_collate_fn([(d, 0) for d in patch_dicts])[0]
    batched_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batched_dict.items()}
    return batched_dict


def postprocess_images(
    output: Union[torch.Tensor, Dict[str, torch.Tensor]],
    output_format: str = "minus_one_to_one",
    current_format: str = "minus_one_to_one",
    unpack: bool = True,
    patch: int = 16,
    max_grid_size: Optional[int] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Postprocess model output into images.

    Args:
        output: Image tensor (B,C,H,W) or patch dict with 'patches' or 'images'
        output_format: Target format ("minus_one_to_one", "zero_to_one", "0_255")
        current_format: Current format of the output
        unpack: Whether to crop to original sizes (requires dict input)
        patch: Patch size for unpatchify
        max_grid_size: Maximum grid size for unpatchify

    Returns:
        Images tensor or list of tensors (if unpack=True)

    Example:
        images = postprocess_images(model_output, output_format="0_255")
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

    # Auto-detect format if needed
    if current_format is None:
        if images.dtype == torch.uint8:
            current_format = "0_255"
        elif images.min() >= -1.5 and images.max() <= 1.5:
            current_format = "minus_one_to_one"
        else:
            current_format = "zero_to_one"

    # Convert formats
    images = _convert_format(images, current_format, output_format)

    if unpack and isinstance(output, dict):
        if 'original_height' not in output or 'original_width' not in output:
            raise ValueError("unpack=True requires 'original_height' and 'original_width' in output")
        return _unpack_images(images, output['original_height'], output['original_width'])

    return images


def unpatchify(
    patch_dict: dict,
    patch: int = 16,
    max_grid_size: Optional[int] = None,
) -> torch.Tensor:
    """Convert patches back to image tensor.

    Args:
        patch_dict: Dictionary with 'patches', 'ptype', 'yidx', 'xidx'
        patch: Patch size
        max_grid_size: Optional max grid size

    Returns:
        Image tensor (B, C, H, W)
    """
    patches = patch_dict['patches']
    ptype = patch_dict['ptype']
    yidx = patch_dict['yidx']
    xidx = patch_dict['xidx']

    B, N, dim = patches.shape
    C = 3

    valid_mask = ptype.bool()
    if max_grid_size is None:
        max_y = yidx[valid_mask].max().item() + 1
        max_x = xidx[valid_mask].max().item() + 1
    else:
        max_y = max_grid_size
        max_x = max_grid_size

    patches = patches.masked_fill(~ptype.unsqueeze(-1), 0.0)
    patches = patches.transpose(1, 2)
    flat_idx = yidx * max_x + xidx

    tokens_dense = torch.zeros(B, C * patch * patch, max_y * max_x,
                               dtype=patches.dtype, device=patches.device)
    tokens_dense = tokens_dense.scatter(2, flat_idx.unsqueeze(1).expand(-1, C * patch * patch, -1), patches)
    first_idx = torch.zeros(B, C * patch * patch, 1, dtype=torch.long, device=patches.device)
    tokens_dense = tokens_dense.scatter(2, first_idx, patches[:, :, 0:1])

    img = F.fold(tokens_dense, output_size=(max_y * patch, max_x * patch), kernel_size=patch, stride=patch)
    return img


def _tensor_to_pil(img: torch.Tensor, assume_format: str = "minus_one_to_one") -> Image.Image:
    """Convert tensor to PIL Image."""
    if img.ndim == 4:
        img = img.squeeze(0)
    if assume_format == "minus_one_to_one":
        img = (img + 1.0) / 2.0
    img = torch.clamp(img, 0.0, 1.0)
    img = (img * 255).to(torch.uint8)
    return Image.fromarray(img.cpu().numpy().transpose(1, 2, 0))


def _convert_format(images: torch.Tensor, from_format: str, to_format: str) -> torch.Tensor:
    """Convert between image formats."""
    if from_format == to_format:
        return images

    if to_format == "minus_one_to_one":
        if from_format == "0_255":
            return images.float() / 127.5 - 1.0
        elif from_format == "zero_to_one":
            return images * 2.0 - 1.0
    elif to_format == "zero_to_one":
        if from_format == "0_255":
            return images.float() / 255.0
        elif from_format == "minus_one_to_one":
            return (images + 1.0) / 2.0
    elif to_format == "0_255":
        if from_format == "minus_one_to_one":
            return ((images + 1.0) / 2.0 * 255).round().to(torch.uint8)
        elif from_format == "zero_to_one":
            return (images * 255).round().to(torch.uint8)

    return images


def _unpack_images(images: torch.Tensor, orig_h: torch.Tensor, orig_w: torch.Tensor) -> List[torch.Tensor]:
    """Crop images to their original sizes."""
    if images.ndim == 3:
        images = images.unsqueeze(0)

    cropped = []
    for img, h, w in zip(images, orig_h, orig_w):
        h_val = int(h.item() if isinstance(h, torch.Tensor) else h)
        w_val = int(w.item() if isinstance(w, torch.Tensor) else w)
        cropped.append(img[:, :h_val, :w_val])
    return cropped


__all__ = [
    "preprocess_images",
    "postprocess_images",
    "unpatchify",
]
