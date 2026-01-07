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
        orig_h = output.get('orig_height', output.get('original_height'))
        orig_w = output.get('orig_width', output.get('original_width'))
        if orig_h is None or orig_w is None:
            raise ValueError("do_unpack=True requires 'orig_height' and 'orig_width' in output")
        return unpack(images, orig_h, orig_w)

    return images


def unpatchify(
    patch_dict: dict,
    patch: int = 16,
    max_grid_size: Optional[int] = None,
) -> torch.Tensor:
    """Convert patches back to image tensor.

    Args:
        patch_dict: Dictionary with patches, patch_mask/ptype, row_idx/yidx, col_idx/xidx
        patch: Patch size
        max_grid_size: Optional max grid size

    Returns:
        Image tensor (B, C, H, W)
    """
    patches = patch_dict['patches']
    # Support both old and new key names
    mask = patch_dict.get('patch_mask', patch_dict.get('ptype'))
    row = patch_dict.get('row_idx', patch_dict.get('yidx'))
    col = patch_dict.get('col_idx', patch_dict.get('xidx'))

    B, N, dim = patches.shape
    C = 3

    valid_mask = mask.bool()
    if max_grid_size is None:
        max_y = row[valid_mask].max().item() + 1
        max_x = col[valid_mask].max().item() + 1
    else:
        max_y = max_grid_size
        max_x = max_grid_size

    patches = patches.masked_fill(~mask.unsqueeze(-1), 0.0)
    patches = patches.transpose(1, 2)
    flat_idx = row * max_x + col

    tokens = torch.zeros(B, C * patch * patch, max_y * max_x, dtype=patches.dtype, device=patches.device)
    tokens = tokens.scatter(2, flat_idx.unsqueeze(1).expand(-1, C * patch * patch, -1), patches)
    first_idx = torch.zeros(B, C * patch * patch, 1, dtype=torch.long, device=patches.device)
    tokens = tokens.scatter(2, first_idx, patches[:, :, 0:1])

    return F.fold(tokens, output_size=(max_y * patch, max_x * patch), kernel_size=patch, stride=patch)


def unpack(
    images: torch.Tensor,
    orig_h: torch.Tensor,
    orig_w: torch.Tensor,
) -> List[torch.Tensor]:
    """Crop images to their original sizes.

    Args:
        images: Image tensor (B, C, H, W)
        orig_h: Original heights per image
        orig_w: Original widths per image

    Returns:
        List of cropped image tensors
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)

    return [
        img[:, :int(h.item() if isinstance(h, torch.Tensor) else h),
               :int(w.item() if isinstance(w, torch.Tensor) else w)]
        for img, h, w in zip(images, orig_h, orig_w)
    ]


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


def sample_tiles(
    images: torch.Tensor,
    orig_h: torch.Tensor,
    orig_w: torch.Tensor,
    n_tiles: int = 2,
    tile_size: tuple = (256, 256),
    indices: Optional[tuple] = None,
) -> tuple:
    """Sample random tiles from images for perceptual losses.

    Args:
        images: Images tensor (B, C, H, W)
        orig_h: Original heights
        orig_w: Original widths
        n_tiles: Number of tiles per image
        tile_size: (height, width) of tiles
        indices: Optional precomputed indices for determinism

    Returns:
        tiles: (B, n_tiles, C, tile_h, tile_w)
        indices: (start_y, start_x) for reproducibility
    """
    device = images.device
    B, C, H, W = images.shape
    tile_h, tile_w = tile_size

    # Pad if needed
    pad_h, pad_w = max(tile_h - H, 0), max(tile_w - W, 0)
    if pad_h or pad_w:
        images = F.pad(images, (0, pad_w, 0, pad_h), value=-1.0)
    _, _, pH, pW = images.shape

    if indices is None:
        # Sample random positions
        max_sy = torch.clamp(orig_h - tile_h, min=0).to(device)
        max_sx = torch.clamp(orig_w - tile_w, min=0).to(device)

        r_y = torch.rand(B, n_tiles, device=device)
        r_x = torch.rand(B, n_tiles, device=device)
        start_y = (r_y * (max_sy.unsqueeze(1).float() + 1)).floor().long()
        start_x = (r_x * (max_sx.unsqueeze(1).float() + 1)).floor().long()
    else:
        start_y, start_x = indices

    # Clamp to canvas
    start_y = torch.clamp(start_y, 0, pH - tile_h)
    start_x = torch.clamp(start_x, 0, pW - tile_w)

    # Gather tiles
    off_y = torch.arange(tile_h, device=device)
    off_x = torch.arange(tile_w, device=device)
    y_idx = start_y[:, :, None, None] + off_y[None, None, :, None]
    x_idx = start_x[:, :, None, None] + off_x[None, None, None, :]
    batch_idx = torch.arange(B, device=device)[:, None, None, None]

    imgs_nhwc = images.permute(0, 2, 3, 1)
    tiles = imgs_nhwc[batch_idx, y_idx, x_idx]
    tiles = tiles.permute(0, 1, 4, 2, 3).contiguous()

    return tiles, (start_y, start_x)


__all__ = [
    "preprocess",
    "postprocess",
    "unpatchify",
    "unpack",
    "sample_tiles",
]
