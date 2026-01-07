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


class RandomTileSampler:
    """Sample random tiles from images for perceptual losses.

    Supports multiple sampling strategies:
    - uniform_pixel: Sample tiles centered on uniformly random pixels
    - uniform_topleft: Sample tiles with uniformly random top-left corners

    The sampler uses a mixture of strategies:
    - 25% corners (top-left, top-right, bottom-left, bottom-right)
    - 25% center
    - 50% random (using the specified coverage strategy)
    """

    def __init__(
        self,
        n_tiles: int = 2,
        tile_size: tuple = (256, 256),
        spatial_stride: int = 16,
        coverage: str = "uniform_pixel",
    ):
        assert coverage in ("uniform_pixel", "uniform_topleft")
        self.n_tiles = n_tiles
        self.tile_h, self.tile_w = tile_size
        self.spatial_stride = spatial_stride
        self.coverage = coverage

    @staticmethod
    def _sample_int(max_inclusive: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Sample random integers in [0, max_inclusive] for each batch item."""
        if max_inclusive.dim() == 0:
            max_inclusive = max_inclusive.unsqueeze(0)
        B = max_inclusive.shape[0]
        r = torch.rand(B, n_samples, device=max_inclusive.device)
        max_expanded = max_inclusive.unsqueeze(1).expand(-1, n_samples).float()
        return (r * (max_expanded + 1.0)).floor().long()

    def __call__(
        self,
        images_2d: torch.Tensor,
        patches_dict: Dict[str, torch.Tensor],
        indices: Optional[tuple] = None,
    ) -> tuple:
        """Sample tiles from images.

        Args:
            images_2d: Images tensor (B, C, H, W)
            patches_dict: Dict with 'original_height' and 'original_width'
            indices: Optional precomputed indices (batch_idx, start_y, start_x)

        Returns:
            tiles: (B, n_tiles, C, tile_h, tile_w)
            indices: (batch_idx_flat, start_y_flat, start_x_flat)
        """
        device = images_2d.device
        B, _, H, W = images_2d.shape

        orig_h = patches_dict["original_height"].to(device).long()
        orig_w = patches_dict["original_width"].to(device).long()

        pad_h = max(self.tile_h - H, 0)
        pad_w = max(self.tile_w - W, 0)
        if pad_h or pad_w:
            images_2d = F.pad(images_2d, (0, pad_w, 0, pad_h), value=-1.0)
        _, _, padded_H, padded_W = images_2d.shape

        if indices is None:
            max_sy = torch.clamp(orig_h - self.tile_h, min=0)
            max_sx = torch.clamp(orig_w - self.tile_w, min=0)

            sampling_type = torch.rand(B, self.n_tiles, device=device)
            bottom_y = torch.clamp(orig_h - self.tile_h, min=0)
            right_x = torch.clamp(orig_w - self.tile_w, min=0)

            corner_y = torch.stack([
                torch.zeros_like(max_sy), torch.zeros_like(max_sy),
                bottom_y, bottom_y
            ], dim=-1)
            corner_x = torch.stack([
                torch.zeros_like(max_sx), right_x,
                torch.zeros_like(max_sx), right_x
            ], dim=-1)

            center_y = max_sy // 2
            center_x = max_sx // 2

            if self.coverage == "uniform_topleft":
                random_y = self._sample_int(max_sy, self.n_tiles)
                random_x = self._sample_int(max_sx, self.n_tiles)
            else:
                pixel_y = self._sample_int(orig_h - 1, self.n_tiles)
                pixel_x = self._sample_int(orig_w - 1, self.n_tiles)
                random_y = pixel_y - self.tile_h // 2
                random_x = pixel_x - self.tile_w // 2

            corner_indices = torch.randint(0, 4, (B, self.n_tiles), device=device)
            selected_corner_y = corner_y.gather(-1, corner_indices.clamp(0, 3))
            selected_corner_x = corner_x.gather(-1, corner_indices.clamp(0, 3))

            center_y_expanded = center_y.unsqueeze(-1).expand(-1, self.n_tiles)
            center_x_expanded = center_x.unsqueeze(-1).expand(-1, self.n_tiles)

            corner_mask = sampling_type < 0.25
            center_mask = (sampling_type >= 0.25) & (sampling_type < 0.5)

            start_y = torch.where(corner_mask, selected_corner_y,
                                  torch.where(center_mask, center_y_expanded, random_y))
            start_x = torch.where(corner_mask, selected_corner_x,
                                  torch.where(center_mask, center_x_expanded, random_x))

            # Clamp to original image bounds (when tile fits within image)
            max_sy_exp = max_sy.unsqueeze(-1).expand(-1, self.n_tiles)
            max_sx_exp = max_sx.unsqueeze(-1).expand(-1, self.n_tiles)
            start_y = torch.maximum(start_y, torch.zeros_like(start_y))
            start_y = torch.minimum(start_y, max_sy_exp)
            start_x = torch.maximum(start_x, torch.zeros_like(start_x))
            start_x = torch.minimum(start_x, max_sx_exp)
        else:
            _, start_y_flat, start_x_flat = indices
            start_y = start_y_flat.view(B, self.n_tiles)
            start_x = start_x_flat.view(B, self.n_tiles)

        start_y = torch.clamp(start_y, min=0, max=padded_H - self.tile_h)
        start_x = torch.clamp(start_x, min=0, max=padded_W - self.tile_w)

        off_y = torch.arange(self.tile_h, device=device)
        off_x = torch.arange(self.tile_w, device=device)
        y_idx = start_y[:, :, None, None] + off_y[None, None, :, None]
        x_idx = start_x[:, :, None, None] + off_x[None, None, None, :]
        batch_idx = torch.arange(B, device=device)[:, None, None, None]

        imgs_nhwc = images_2d.permute(0, 2, 3, 1)
        tiles = imgs_nhwc[batch_idx, y_idx, x_idx]
        tiles = tiles.permute(0, 1, 4, 2, 3).contiguous()

        batch_idx_flat = torch.repeat_interleave(
            torch.arange(B, device=device), self.n_tiles
        )
        return tiles, (batch_idx_flat, start_y.flatten(), start_x.flatten())


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
    "RandomTileSampler",
]
