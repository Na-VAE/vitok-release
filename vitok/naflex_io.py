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


class RandomTileSampler:
    """Sample random tiles from images for computing perceptual losses.

    For computing SSIM/LPIPS/DINO losses on variable-resolution images,
    we sample fixed-size tiles rather than resizing the full image.

    Args:
        n_tiles: Number of tiles to sample per image
        tile_size: (height, width) of each tile
        spatial_stride: Patch size used by the model
        coverage: Sampling strategy - "uniform_pixel" or "uniform_topleft"
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

    def _sample_start_positions(
        self,
        orig_h: torch.Tensor,
        orig_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tile top-lefts in original image coordinates."""
        max_sy = torch.clamp(orig_h - self.tile_h, min=0)
        max_sx = torch.clamp(orig_w - self.tile_w, min=0)

        if self.coverage == "uniform_topleft":
            start_y = self._sample_int(max_sy, self.n_tiles)
            start_x = self._sample_int(max_sx, self.n_tiles)
        else:  # uniform_pixel
            half_h = self.tile_h // 2
            half_w = self.tile_w // 2
            # Prefer centers whose tiles fully fit in the original image.
            min_center_y = torch.full_like(orig_h, half_h)
            min_center_x = torch.full_like(orig_w, half_w)
            max_center_y = orig_h - self.tile_h + half_h
            max_center_x = orig_w - self.tile_w + half_w

            valid_y = max_center_y >= min_center_y
            valid_x = max_center_x >= min_center_x

            range_y = torch.where(valid_y, max_center_y - min_center_y, orig_h - 1)
            range_x = torch.where(valid_x, max_center_x - min_center_x, orig_w - 1)
            range_y = torch.clamp(range_y, min=0)
            range_x = torch.clamp(range_x, min=0)

            offset_y = torch.where(valid_y, min_center_y, torch.zeros_like(min_center_y))
            offset_x = torch.where(valid_x, min_center_x, torch.zeros_like(min_center_x))

            pixel_y = self._sample_int(range_y, self.n_tiles) + offset_y.unsqueeze(1)
            pixel_x = self._sample_int(range_x, self.n_tiles) + offset_x.unsqueeze(1)
            start_y = pixel_y - half_h
            start_x = pixel_x - half_w

        return start_y, start_x

    def __call__(
        self,
        images_2d: torch.Tensor,
        patches_dict: Dict[str, torch.Tensor],
        indices: Optional[tuple] = None,
    ) -> tuple:
        """Sample tiles from images.

        Args:
            images_2d: Reconstructed images (B, C, H, W)
            patches_dict: Patch dict with 'original_height', 'original_width'
            indices: Optional precomputed indices for deterministic sampling

        Returns:
            tiles: (B, n_tiles, C, tile_h, tile_w) tensor
            indices: (batch_idx, start_y, start_x) tuple for reproducibility
        """
        device = images_2d.device
        B, C, H, W = images_2d.shape

        orig_h = patches_dict["original_height"].to(device).long()
        orig_w = patches_dict["original_width"].to(device).long()

        # Pad if canvas is smaller than tile
        pad_h = max(self.tile_h - H, 0)
        pad_w = max(self.tile_w - W, 0)
        if pad_h or pad_w:
            images_2d = F.pad(images_2d, (0, pad_w, 0, pad_h), value=-1.0)
        _, _, padded_H, padded_W = images_2d.shape

        if indices is None:
            start_y, start_x = self._sample_start_positions(orig_h, orig_w)
        else:
            batch_idx_flat, start_y_flat, start_x_flat = indices
            start_y = start_y_flat.view(B, self.n_tiles)
            start_x = start_x_flat.view(B, self.n_tiles)

        # Clamp to padded canvas
        start_y = torch.clamp(start_y, min=0, max=padded_H - self.tile_h)
        start_x = torch.clamp(start_x, min=0, max=padded_W - self.tile_w)

        # Build index grids and gather tiles
        off_y = torch.arange(self.tile_h, device=device)
        off_x = torch.arange(self.tile_w, device=device)
        y_idx = start_y[:, :, None, None] + off_y[None, None, :, None]
        x_idx = start_x[:, :, None, None] + off_x[None, None, None, :]
        batch_idx = torch.arange(B, device=device)[:, None, None, None]

        # Gather tiles
        imgs_nhwc = images_2d.permute(0, 2, 3, 1)  # (B, H, W, C)
        tiles = imgs_nhwc[batch_idx, y_idx, x_idx]  # (B, n, h, w, C)
        tiles = tiles.permute(0, 1, 4, 2, 3).contiguous()  # (B, n, C, h, w)

        # Return indices for reproducibility
        batch_idx_flat = torch.repeat_interleave(
            torch.arange(B, device=device), self.n_tiles
        )
        return tiles, (batch_idx_flat, start_y.flatten(), start_x.flatten())


__all__ = [
    "preprocess_images",
    "postprocess_images",
    "unpatchify",
    "RandomTileSampler",
]
