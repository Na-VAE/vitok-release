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
import random
from typing import Callable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# =============================================================================
# Resize ops (PIL -> PIL)
# =============================================================================


def resize_longest_side(max_size: int):
    """Resize so longest side is at most max_size, preserving aspect ratio."""
    def _resize(img: Image.Image) -> Image.Image:
        w, h = img.size
        if max(h, w) <= max_size:
            return img
        scale = max_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        return TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.LANCZOS, antialias=True)
    return _resize


# =============================================================================
# Crop ops (PIL -> PIL)
# =============================================================================


def center_crop(size: int):
    """ADM-style center crop with anti-aliasing.

    Reference:
        https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py#L126
    """
    def _center_crop(img: Image.Image) -> Image.Image:
        # Downsample by 2x while min side >= 2 * target (reduces aliasing)
        while min(*img.size) >= 2 * size:
            img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
        # Scale so min side == target
        scale = size / min(*img.size)
        img = img.resize(tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC)
        # Center crop
        arr = np.array(img)
        crop_y = (arr.shape[0] - size) // 2
        crop_x = (arr.shape[1] - size) // 2
        return Image.fromarray(arr[crop_y:crop_y + size, crop_x:crop_x + size])
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
# Composition ops
# =============================================================================


def identity() -> Callable:
    """No-op transform."""
    def _identity(x):
        return x
    return _identity


def random_choice(ops: Sequence[str], probs: Sequence[float]) -> Callable:
    """Randomly apply one of several ops.

    Args:
        ops: Sequence of op spec strings (e.g., ['random_resized_crop(256)', 'identity'])
        probs: Probability weights (same length as ops)

    Raises:
        ValueError: If ops/probs are empty or have mismatched lengths
    """
    if not ops:
        raise ValueError("ops cannot be empty")
    if len(ops) != len(probs):
        raise ValueError(f"ops and probs must have same length: {len(ops)} != {len(probs)}")

    from vitok.pp.registry import parse_op

    resolved = []
    for op in ops:
        name, args, kwargs = parse_op(op)
        resolved.append(OPS[name](*args, **kwargs))

    def _random_choice(x):
        idx = random.choices(range(len(resolved)), weights=probs, k=1)[0]
        return resolved[idx](x)

    return _random_choice


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
    eps: float = 1e-5,
) -> Tuple[int, int]:
    """Find the largest (h', w') <= (h, w) that fits within token budget.

    Uses closed-form calculation without iterative fallback.
    """
    h_p = math.ceil(h / patch)
    w_p = math.ceil(w / patch)

    # Early exit if already within budget
    if h_p * w_p <= max_tokens:
        return h, w

    # Scale down to fit token budget (floor + eps for stability)
    scale = math.sqrt(max_tokens / (h_p * w_p))
    new_h_p = max(1, math.floor(h_p * scale + eps))
    new_w_p = max(1, math.floor(w_p * scale + eps))

    new_h = new_h_p * patch
    new_w = new_w_p * patch

    return min(new_h, h), min(new_w, w)


def resize_to_token_budget(patch: int, max_tokens: int):
    """Resize tensor to fit within token budget.

    Args:
        patch: Patch size in pixels
        max_tokens: Maximum number of patches allowed

    Input: Tensor (C, H, W)
    Output: Tensor (C, H', W') where ceil(H'/patch) * ceil(W'/patch) <= max_tokens
    """
    def _resize(img: torch.Tensor) -> torch.Tensor:
        c, h, w = img.shape
        target_h, target_w = _fit_to_token_budget(h, w, patch, max_tokens)
        if (target_h, target_w) != (h, w):
            img = TF.resize(img, [target_h, target_w], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        return img
    return _resize


def patchify(patch: int = 16, temporal_patch: int = 1, max_tokens: int = 256):
    """Convert image/video tensor to patch dictionary.

    Unified patchification for both images and videos. Images are treated as
    single-frame videos (T=1). Pads to patch boundaries and creates tubelets.

    Args:
        patch: Spatial patch size in pixels
        temporal_patch: Temporal patch size in frames (1 for images, >1 for video)
        max_tokens: Maximum number of patches (for padding output)

    Input:
        - Image: Tensor (C, H, W) → treated as (1, C, H, W)
        - Video: Tensor (T, C, H, W)

    Output: Dict with patches, patch_mask, time_idx, row_idx, col_idx, etc.

    When temporal_patch=1: equivalent to current 2D patchify (backwards compat)
    When temporal_patch>1: creates 3D tubelets for video
    """
    def _patchify(x: torch.Tensor) -> dict:
        # Handle both (C, H, W) and (T, C, H, W) inputs
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (C, H, W) → (1, C, H, W)

        t, c, h, w = x.shape
        orig_t, orig_h, orig_w = t, h, w

        # Step 1: Pad to temporal and spatial boundaries
        pad_t = (temporal_patch - t % temporal_patch) % temporal_patch
        pad_h = (patch - h % patch) % patch
        pad_w = (patch - w % patch) % patch

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            # Pad format: (left, right, top, bottom, front, back)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0, 0, pad_t), mode='constant', value=0.0)

        t_padded, c, h_padded, w_padded = x.shape

        # Step 2: Reshape to tubelets
        grid_t = t_padded // temporal_patch
        grid_h = h_padded // patch
        grid_w = w_padded // patch

        # Reshape: (T, C, H, W) → (grid_t, temporal_patch, C, grid_h, patch, grid_w, patch)
        x = x.reshape(grid_t, temporal_patch, c, grid_h, patch, grid_w, patch)
        # Permute to: (grid_t, grid_h, grid_w, temporal_patch, C, patch, patch)
        x = x.permute(0, 3, 5, 1, 2, 4, 6)
        # Flatten tubelet: (grid_t * grid_h * grid_w, temporal_patch * C * patch * patch)
        patches = x.reshape(-1, temporal_patch * c * patch * patch)

        n_patches = grid_t * grid_h * grid_w

        # Step 3: Generate 3D indices
        t_coords, y_coords, x_coords = torch.meshgrid(
            torch.arange(grid_t),
            torch.arange(grid_h),
            torch.arange(grid_w),
            indexing='ij'
        )
        time_idx = t_coords.flatten()
        row_idx = y_coords.flatten()
        col_idx = x_coords.flatten()

        # Step 4: Pad to max_tokens
        patches_full = torch.zeros(max_tokens, patches.shape[1], dtype=patches.dtype)
        patches_full[:n_patches] = patches

        patch_mask = torch.zeros(max_tokens, dtype=torch.bool)
        patch_mask[:n_patches] = True

        time_idx_full = torch.zeros(max_tokens, dtype=torch.long)
        time_idx_full[:n_patches] = time_idx

        row_idx_full = torch.zeros(max_tokens, dtype=torch.long)
        row_idx_full[:n_patches] = row_idx

        col_idx_full = torch.zeros(max_tokens, dtype=torch.long)
        col_idx_full[:n_patches] = col_idx

        return {
            'patches': patches_full,
            'patch_mask': patch_mask,
            'time_idx': time_idx_full,
            'row_idx': row_idx_full,
            'col_idx': col_idx_full,
            'orig_frames': torch.tensor(orig_t, dtype=torch.long),
            'orig_height': torch.tensor(orig_h, dtype=torch.long),
            'orig_width': torch.tensor(orig_w, dtype=torch.long),
            'grid_t': torch.tensor(grid_t, dtype=torch.long),
            'grid_rows': torch.tensor(grid_h, dtype=torch.long),
            'grid_cols': torch.tensor(grid_w, dtype=torch.long),
        }

    return _patchify


# =============================================================================
# Unpatchify (inverse of patchify)
# =============================================================================

from typing import Optional, List


def unpatchify(
    patch_dict: dict,
    patch: int = 16,
    temporal_patch: int = 1,
    max_grid_size: Optional[int] = None,
) -> torch.Tensor:
    """Convert patches back to image/video tensor.

    Unified unpatchification for both images and videos. Inverse of patchify().

    Args:
        patch_dict: Dictionary with patches, patch_mask, time_idx, row_idx, col_idx
        patch: Spatial patch size
        temporal_patch: Temporal patch size (1 for images)
        max_grid_size: Optional max spatial grid size

    Returns:
        - Image: tensor (B, C, H, W) when grid_t=1
        - Video: tensor (B, T, C, H, W) when grid_t>1
    """
    patches = patch_dict['patches']
    mask = patch_dict['patch_mask']
    time = patch_dict['time_idx']
    row = patch_dict['row_idx']
    col = patch_dict['col_idx']

    B, N, dim = patches.shape
    C = 3

    valid_mask = mask.bool()

    # Determine grid dimensions
    if max_grid_size is None:
        max_t = time[valid_mask].max().item() + 1 if valid_mask.any() else 1
        max_y = row[valid_mask].max().item() + 1 if valid_mask.any() else 1
        max_x = col[valid_mask].max().item() + 1 if valid_mask.any() else 1
    else:
        max_t = time[valid_mask].max().item() + 1 if valid_mask.any() else 1
        max_y = max_grid_size
        max_x = max_grid_size

    # Handle 2D (image) case with simple fold
    if max_t == 1 and temporal_patch == 1:
        patches = patches.masked_fill(~mask.unsqueeze(-1), 0.0)
        patches = patches.transpose(1, 2)
        flat_idx = row * max_x + col

        tokens = torch.zeros(B, C * patch * patch, max_y * max_x, dtype=patches.dtype, device=patches.device)
        tokens = tokens.scatter(2, flat_idx.unsqueeze(1).expand(-1, C * patch * patch, -1), patches)
        first_idx = torch.zeros(B, C * patch * patch, 1, dtype=torch.long, device=patches.device)
        tokens = tokens.scatter(2, first_idx, patches[:, :, 0:1])

        return F.fold(tokens, output_size=(max_y * patch, max_x * patch), kernel_size=patch, stride=patch)

    # Handle 3D (video) case
    patches = patches.masked_fill(~mask.unsqueeze(-1), 0.0)

    # Reshape patches back to tubelet form
    # patches: (B, N, temporal_patch * C * patch * patch)
    tubelet_dim = temporal_patch * C * patch * patch
    patches = patches.reshape(B, N, temporal_patch, C, patch, patch)

    # Create output tensor
    T_out = max_t * temporal_patch
    H_out = max_y * patch
    W_out = max_x * patch
    output = torch.zeros(B, T_out, C, H_out, W_out, dtype=patches.dtype, device=patches.device)

    # Scatter tubelets back to output
    for b in range(B):
        for n in range(N):
            if not mask[b, n]:
                continue
            t_idx = time[b, n].item()
            y_idx = row[b, n].item()
            x_idx = col[b, n].item()

            t_start = t_idx * temporal_patch
            y_start = y_idx * patch
            x_start = x_idx * patch

            output[b, t_start:t_start + temporal_patch, :,
                   y_start:y_start + patch, x_start:x_start + patch] = patches[b, n]

    return output


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
    "resize_to_token_budget": resize_to_token_budget,
    "flip": flip,
    "identity": identity,
    "random_choice": random_choice,
    "to_tensor": to_tensor,
    "normalize": normalize,
    "patchify": patchify,
}


__all__ = [
    "center_crop",
    "random_resized_crop",
    "resize_longest_side",
    "resize_to_token_budget",
    "flip",
    "identity",
    "random_choice",
    "to_tensor",
    "normalize",
    "patchify",
    "unpatchify",
    "unpack",
    "sample_tiles",
    "OPS",
]
