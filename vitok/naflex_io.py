"""Image preprocessing and postprocessing utilities.

Simple high-level API for converting images to/from patch dictionaries.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image

from vitok.pp import build_transform
from vitok.data import patch_collate_fn


def preprocess(
    images: Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]],
    pp: str = "to_tensor|normalize(minus_one_to_one)|patchify(256, 16)",
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Preprocess images into the patch dictionary expected by models.

    Args:
        images: PIL Image(s) or tensor(s)
        pp: Preprocessing string, e.g.:
            "to_tensor|normalize(minus_one_to_one)|patchify(256, 16)"
        device: Target device

    Returns:
        Batched patch dictionary with keys:
            - patches: [B, N, D] flattened patch pixels
            - patch_mask: [B, N] validity mask
            - row_idx, col_idx: [B, N] spatial indices
            - attention_mask: [B, N, N]
            - orig_height, orig_width: [B]

    Example:
        patches = preprocess(
            [img1, img2],
            pp="to_tensor|normalize(minus_one_to_one)|patchify(256, 16)",
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

    batched_dict = patch_collate_fn(patch_dicts)
    batched_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batched_dict.items()}
    return batched_dict


def postprocess(
    output: Union[torch.Tensor, Dict[str, torch.Tensor]],
    current_format: str = "minus_one_to_one",
    output_format: str = "0_255",
    patch: int = 16,
    max_grid_size: int = 32,
) -> torch.Tensor:
    """Postprocess model output into images.

    Args:
        output: Image tensor (B,C,H,W) or patch dict with 'patches' or 'images'
        current_format: Current format of the output (required, no auto-detect)
        output_format: Target format ("minus_one_to_one", "zero_to_one", "0_255")
        patch: Patch size for unpatchify
        max_grid_size: Maximum grid size for unpatchify (required)

    Returns:
        Images tensor (B, C, H, W)

    Example:
        images = postprocess(model_output, current_format="minus_one_to_one", output_format="0_255")
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

    # Convert formats
    images = _convert_format(images, current_format, output_format)
    return images


def unpack(
    images: torch.Tensor,
    orig_height: torch.Tensor,
    orig_width: torch.Tensor,
) -> List[torch.Tensor]:
    """Crop images to their original sizes.

    Args:
        images: Image tensor (B, C, H, W)
        orig_height: Original heights per image (B,)
        orig_width: Original widths per image (B,)

    Returns:
        List of cropped image tensors, one per batch element
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)

    cropped = []
    for img, h, w in zip(images, orig_height, orig_width):
        h_val = int(h.item() if isinstance(h, torch.Tensor) else h)
        w_val = int(w.item() if isinstance(w, torch.Tensor) else w)
        cropped.append(img[:, :h_val, :w_val])
    return cropped


def unpatchify(
    patch_dict: dict,
    patch: int = 16,
    max_grid_size: int = 32,
) -> torch.Tensor:
    """Convert patches back to image tensor.

    Args:
        patch_dict: Dictionary with 'patches', 'patch_mask', 'row_idx', 'col_idx'
        patch: Patch size
        max_grid_size: Maximum grid size (required)

    Returns:
        Image tensor (B, C, H, W)
    """
    patches = patch_dict['patches']

    # Support both old and new key names during transition
    patch_mask = patch_dict.get('patch_mask', patch_dict.get('ptype'))
    row_idx = patch_dict.get('row_idx', patch_dict.get('yidx'))
    col_idx = patch_dict.get('col_idx', patch_dict.get('xidx'))

    B, N, dim = patches.shape
    C = 3

    valid_mask = patch_mask.bool()
    max_y = max_grid_size
    max_x = max_grid_size

    patches = patches.masked_fill(~patch_mask.unsqueeze(-1), 0.0)
    patches = patches.transpose(1, 2)
    flat_idx = row_idx * max_x + col_idx

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


__all__ = [
    "preprocess",
    "postprocess",
    "unpatchify",
    "unpack",
]
