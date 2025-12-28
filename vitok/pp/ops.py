"""Registered preprocessing ops.

All ops follow the factory pattern:
    @Registry.register("op_name")
    def get_op_name(arg1, arg2, ...):
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

from vitok.pp.registry import Registry


# =============================================================================
# Crop ops (PIL -> PIL)
# =============================================================================


@Registry.register("center_crop")
def get_center_crop(size: int):
    """Center crop to size x size."""
    def _center_crop(img: Image.Image) -> Image.Image:
        return TF.center_crop(img, [size, size])
    return _center_crop


@Registry.register("random_resized_crop")
def get_random_resized_crop(
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


@Registry.register("flip")
def get_flip(p: float = 0.5):
    """Random horizontal flip with probability p."""
    return T.RandomHorizontalFlip(p)


# =============================================================================
# Conversion ops
# =============================================================================


@Registry.register("to_tensor")
def get_to_tensor():
    """Convert PIL Image to Tensor (0-1 range)."""
    return T.ToTensor()


@Registry.register("normalize")
def get_normalize(mode: str = "minus_one_to_one"):
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


@Registry.register("patchify")
def get_patchify(
    max_size: int = 512,
    patch: int = 16,
    max_tokens: int = 256,
    max_grid_size: int | None = None,
):
    """Convert tensor to patch dictionary.

    This is an all-in-one op that:
    1. Resizes image to fit within token budget (longest side to max_size first)
    2. Pads to patch boundary
    3. Unfolds to patches
    4. Creates full patch dict

    Args:
        max_size: Maximum size for longest side before budget fitting
        patch: Patch size in pixels
        max_tokens: Maximum number of patches (token budget)
        max_grid_size: Optional maximum grid dimension

    Input: Tensor (C, H, W) - normalized
    Output: Dict with patches, ptype, yidx, xidx, attention_mask, etc.
    """
    def _patchify(img: torch.Tensor) -> dict:
        c, h, w = img.shape

        # Step 1: Resize longest side to max_size (preserving aspect ratio)
        long_side = max(h, w)
        if long_side > max_size:
            scale = max_size / long_side
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            img = TF.resize(img, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            h, w = new_h, new_w

        # Step 2: Fit to token budget
        target_h, target_w = _fit_to_token_budget(h, w, patch, max_tokens, max_grid_size)
        if (target_h, target_w) != (h, w):
            img = TF.resize(img, [target_h, target_w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            h, w = target_h, target_w

        original_h, original_w = h, w

        # Step 3: Pad to patch boundary
        pad_h = (patch - h % patch) % patch
        pad_w = (patch - w % patch) % patch
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

        # Step 4: Unfold to patches
        c, h_padded, w_padded = img.shape
        patches = F.unfold(img.unsqueeze(0), kernel_size=patch, stride=patch)
        patches = patches.squeeze(0).T  # (N, C*patch*patch)

        n_patches_h = h_padded // patch
        n_patches_w = w_padded // patch
        n_patches = n_patches_h * n_patches_w

        # Step 5: Generate spatial indices
        y_coords, x_coords = torch.meshgrid(
            torch.arange(n_patches_h),
            torch.arange(n_patches_w),
            indexing='ij'
        )
        yidx = y_coords.flatten()
        xidx = x_coords.flatten()

        # Step 6: Pad to max_tokens
        patches_full = torch.zeros(max_tokens, patches.shape[1], dtype=patches.dtype)
        patches_full[:n_patches] = patches

        ptype = torch.zeros(max_tokens, dtype=torch.bool)
        ptype[:n_patches] = True

        yidx_full = torch.zeros(max_tokens, dtype=torch.long)
        yidx_full[:n_patches] = yidx

        xidx_full = torch.zeros(max_tokens, dtype=torch.long)
        xidx_full[:n_patches] = xidx

        tidx = torch.zeros(max_tokens, dtype=torch.long)

        # Step 7: Create attention mask
        attn_mask = ptype.unsqueeze(0) & ptype.unsqueeze(1)
        attn_mask = attn_mask | torch.eye(max_tokens, dtype=torch.bool)

        return {
            'patches': patches_full,
            'ptype': ptype,
            'yidx': yidx_full,
            'xidx': xidx_full,
            'tidx': tidx,
            'original_height': torch.tensor(original_h, dtype=torch.long),
            'original_width': torch.tensor(original_w, dtype=torch.long),
            'grid_h': torch.tensor(n_patches_h, dtype=torch.long),
            'grid_w': torch.tensor(n_patches_w, dtype=torch.long),
            'attention_mask': attn_mask,
        }

    return _patchify


__all__ = [
    "get_center_crop",
    "get_random_resized_crop",
    "get_flip",
    "get_to_tensor",
    "get_normalize",
    "get_patchify",
]
