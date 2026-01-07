"""Preprocessing ops.

All ops follow the factory pattern:
    def op_name(arg1, arg2, ...):
        def _op(input):
            # transform input
            return output
        return _op
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# =============================================================================
# Resize ops (PIL -> PIL or Tensor -> Tensor)
# =============================================================================


def resize_longest_side(max_size: int):
    """Resize so longest side is at most max_size, preserving aspect ratio.

    Works on both PIL Images and Tensors.
    """
    def _resize(img):
        if isinstance(img, Image.Image):
            w, h = img.size
            if max(h, w) <= max_size:
                return img
            scale = max_size / max(h, w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            return TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.LANCZOS, antialias=True)
        else:
            # Tensor (C, H, W) - use BICUBIC since LANCZOS not supported for tensors
            c, h, w = img.shape
            if max(h, w) <= max_size:
                return img
            scale = max_size / max(h, w)
            new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
            return TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    return _resize


# =============================================================================
# Crop ops (PIL -> PIL)
# =============================================================================


def center_crop(size: int):
    """Center crop to size x size."""
    def _center_crop(img: Image.Image) -> Image.Image:
        return TF.center_crop(img, [size, size])
    return _center_crop


def random_resized_crop(
    size: int,
    scale: Tuple[float, float] = (0.8, 1.0),
    ratio: Tuple[float, float] = (0.75, 1.333),
):
    """Random resized crop (standard ImageNet augmentation)."""
    transform = T.RandomResizedCrop(
        size,
        scale=scale,
        ratio=ratio,
        interpolation=TF.InterpolationMode.LANCZOS,
        antialias=True,
    )
    return transform


# =============================================================================
# Augmentation ops (PIL -> PIL)
# =============================================================================


def flip(p: float = 0.5):
    """Random horizontal flip with probability p."""
    return T.RandomHorizontalFlip(p)


# =============================================================================
# Conversion ops
# =============================================================================


def to_tensor():
    """Convert PIL Image to Tensor (0-1 range)."""
    return T.ToTensor()


def normalize(mode: str = "minus_one_to_one"):
    """Normalize tensor.

    Args:
        mode: Normalization mode
            - "minus_one_to_one": maps [0,1] to [-1,1]
            - "imagenet": ImageNet mean/std normalization
            - "zero_to_one": identity (already 0-1 from to_tensor)
    """
    if mode == "minus_one_to_one":
        return T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif mode == "imagenet":
        return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif mode == "zero_to_one":
        return lambda x: x
    else:
        raise ValueError(f"Unknown normalize mode: '{mode}'. Use 'minus_one_to_one', 'imagenet', or 'zero_to_one'")


# =============================================================================
# Patchify (Tensor -> dict)
# =============================================================================


def _fit_to_token_budget(
    h: int,
    w: int,
    patch: int,
    max_tokens: int,
    max_grid_size: int | None = None,
) -> Tuple[int, int]:
    """Find the largest (h', w') <= (h, w) that fits within token budget."""
    h_p = math.ceil(h / patch)
    w_p = math.ceil(w / patch)

    within_tokens = (h_p * w_p) <= max_tokens
    within_grid = (max_grid_size is None) or (h_p <= max_grid_size and w_p <= max_grid_size)
    if within_tokens and within_grid:
        return h, w

    scale = math.sqrt(max_tokens / (h_p * w_p))
    new_h_p = max(1, int(h_p * scale))
    new_w_p = max(1, int(w_p * scale))

    if max_grid_size is not None:
        new_h_p = min(new_h_p, max_grid_size)
        new_w_p = min(new_w_p, max_grid_size)

    new_h = min(new_h_p * patch, h)
    new_w = min(new_w_p * patch, w)

    return new_h, new_w


def patchify(
    patch: int = 16,
    max_tokens: int = 256,
    max_grid_size: int | None = None,
):
    """Convert tensor to patch dictionary.

    Resizes to fit token budget, pads to patch boundary, and unfolds to patches.

    Args:
        patch: Patch size in pixels
        max_tokens: Maximum number of patches (token budget)
        max_grid_size: Optional maximum grid dimension

    Input: Tensor (C, H, W) - normalized
    Output: Dict with patches, patch_mask, row_idx, col_idx, attention_mask, etc.
    """
    def _patchify(img: torch.Tensor) -> dict:
        c, h, w = img.shape

        # Step 1: Fit to token budget (resize if needed)
        target_h, target_w = _fit_to_token_budget(h, w, patch, max_tokens, max_grid_size)
        if (target_h, target_w) != (h, w):
            img = TF.resize(img, [target_h, target_w], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
            h, w = target_h, target_w

        orig_h, orig_w = h, w

        # Step 2: Pad to patch boundary
        pad_h = (patch - h % patch) % patch
        pad_w = (patch - w % patch) % patch
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

        # Step 3: Unfold to patches
        c, h_padded, w_padded = img.shape
        patches = F.unfold(img.unsqueeze(0), kernel_size=patch, stride=patch)
        patches = patches.squeeze(0).T  # (N, C*patch*patch)

        grid_rows = h_padded // patch
        grid_cols = w_padded // patch
        n_patches = grid_rows * grid_cols

        # Step 4: Generate spatial indices
        y_coords, x_coords = torch.meshgrid(
            torch.arange(grid_rows),
            torch.arange(grid_cols),
            indexing='ij'
        )
        row_idx = y_coords.flatten()
        col_idx = x_coords.flatten()

        # Step 5: Pad to max_tokens
        patches_full = torch.zeros(max_tokens, patches.shape[1], dtype=patches.dtype)
        patches_full[:n_patches] = patches

        patch_mask = torch.zeros(max_tokens, dtype=torch.bool)
        patch_mask[:n_patches] = True

        row_idx_full = torch.zeros(max_tokens, dtype=torch.long)
        row_idx_full[:n_patches] = row_idx

        col_idx_full = torch.zeros(max_tokens, dtype=torch.long)
        col_idx_full[:n_patches] = col_idx

        time_idx = torch.zeros(max_tokens, dtype=torch.long)

        # Step 6: Create attention mask
        attn_mask = patch_mask.unsqueeze(0) & patch_mask.unsqueeze(1)
        attn_mask = attn_mask | torch.eye(max_tokens, dtype=torch.bool)

        return {
            'patches': patches_full,
            'patch_mask': patch_mask,
            'row_idx': row_idx_full,
            'col_idx': col_idx_full,
            'time_idx': time_idx,
            'orig_height': torch.tensor(orig_h, dtype=torch.long),
            'orig_width': torch.tensor(orig_w, dtype=torch.long),
            'grid_rows': torch.tensor(grid_rows, dtype=torch.long),
            'grid_cols': torch.tensor(grid_cols, dtype=torch.long),
            'attention_mask': attn_mask,
        }

    return _patchify


# =============================================================================
# Unpatchify (inverse of patchify)
# =============================================================================

from typing import Optional, List


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


# =============================================================================
# Tile sampling (for perceptual losses)
# =============================================================================


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


# =============================================================================
# OPS registry (simple dict)
# =============================================================================

OPS = {
    "center_crop": center_crop,
    "random_resized_crop": random_resized_crop,
    "resize_longest_side": resize_longest_side,
    "flip": flip,
    "to_tensor": to_tensor,
    "normalize": normalize,
    "patchify": patchify,
}


__all__ = [
    "center_crop",
    "random_resized_crop",
    "resize_longest_side",
    "flip",
    "to_tensor",
    "normalize",
    "patchify",
    "unpatchify",
    "unpack",
    "sample_tiles",
    "OPS",
]
