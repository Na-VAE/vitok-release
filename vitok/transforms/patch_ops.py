"""Patch operations: patchify, unpatchify, and patch dict creation."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

from vitok.transforms.positional_encoding import PositionalEncoding2D


def ceil_div(a: torch.Tensor | float, b: int | float) -> torch.Tensor | float:
    """Ceiling division."""
    return (a + b - 1) // b if isinstance(a, torch.Tensor) else math.ceil(a / b)


def patchify(
    img: torch.Tensor,
    patch: int = 16,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Convert image to patches.

    Args:
        img: Image tensor (C, H, W)
        patch: Patch size

    Returns:
        Tuple of (patches [N, C*patch*patch], (original_h, original_w))
    """
    c, h, w = img.shape
    original_h, original_w = h, w

    pad_h = (patch - h % patch) % patch
    pad_w = (patch - w % patch) % patch

    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

    patches = F.unfold(img, kernel_size=patch, stride=patch).T
    return patches, (original_h, original_w)


def unpatchify(
    patch_dict: dict,
    patch: int = 16,
    max_grid_size: Optional[int] = None,
):
    """Convert patches back to image.

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
    flat_idx = (yidx * max_x + xidx)

    tokens_dense = torch.zeros(B, C * patch * patch, max_y * max_x,
                               dtype=patches.dtype, device=patches.device)
    tokens_dense = tokens_dense.scatter(2, flat_idx.unsqueeze(1).expand(-1, C * patch * patch, -1), patches)
    first_idx = torch.zeros(B, C * patch * patch, 1, dtype=torch.long, device=patches.device)
    tokens_dense = tokens_dense.scatter(2, first_idx, patches[:, :, 0:1])

    img = F.fold(tokens_dense, output_size=(max_y * patch, max_x * patch), kernel_size=patch, stride=patch)
    return img


def create_patch_dict(
    img: torch.Tensor,
    patch: int = 16,
    max_seq_len: Optional[int] = None,
    posemb_encoder: Optional[PositionalEncoding2D] = None
) -> dict:
    """Create a complete patch dictionary from an image tensor.

    Args:
        img: Image tensor (C, H, W)
        patch: Patch size
        max_seq_len: If provided, pad sequences to this length
        posemb_encoder: Optional PositionalEncoding2D for positional embeddings

    Returns:
        Patch dictionary with all required keys
    """
    assert img.ndim == 3, f"Expected single image (C, H, W), got shape {img.shape}"

    C, H, W = img.shape
    device = img.device
    patches, (orig_h, orig_w) = patchify(img, patch)
    n_patches_h = ceil_div(H, patch)
    n_patches_w = ceil_div(W, patch)
    n_patches = n_patches_h * n_patches_w

    y_coords, x_coords = torch.meshgrid(
        torch.arange(n_patches_h, device=device),
        torch.arange(n_patches_w, device=device),
        indexing='ij'
    )
    yidx = y_coords.flatten()
    xidx = x_coords.flatten()

    patches_full = torch.zeros(max_seq_len, patches.shape[1], device=device, dtype=patches.dtype)
    patches_full[:n_patches] = patches

    ptype = torch.zeros(max_seq_len, dtype=torch.bool, device=device)
    ptype[:n_patches] = True

    yidx_full = torch.zeros(max_seq_len, dtype=torch.long, device=device)
    yidx_full[:n_patches] = yidx

    xidx_full = torch.zeros(max_seq_len, dtype=torch.long, device=device)
    xidx_full[:n_patches] = xidx

    tidx = torch.zeros(max_seq_len, dtype=torch.long, device=device)

    patches = patches_full
    yidx = yidx_full
    xidx = xidx_full

    attn_mask = ptype.unsqueeze(0) & ptype.unsqueeze(1)
    attn_mask = attn_mask | torch.eye(max_seq_len, dtype=torch.bool, device=device)

    patch_dict = {
        'patches': patches,
        'ptype': ptype,
        'yidx': yidx,
        'xidx': xidx,
        'tidx': tidx,
        'original_height': torch.tensor(orig_h, dtype=torch.long, device=device),
        'original_width': torch.tensor(orig_w, dtype=torch.long, device=device),
        'grid_h': torch.tensor(n_patches_h, dtype=torch.long, device=device),
        'grid_w': torch.tensor(n_patches_w, dtype=torch.long, device=device),
        'attention_mask': attn_mask
    }

    if posemb_encoder is not None:
        posemb = posemb_encoder(patch_dict)
        patch_dict['naflex_posemb'] = posemb

    return patch_dict


class RandomTileSampler:
    """Random tile sampler for perceptual losses."""

    def __init__(
        self,
        n_tiles: int = 2,
        tile_size: Tuple[int, int] = (256, 256),
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
        indices: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ):
        device = images_2d.device
        B, C, H, W = images_2d.shape

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

            if self.coverage == "uniform_topleft":
                start_y = self._sample_int(max_sy, self.n_tiles)
                start_x = self._sample_int(max_sx, self.n_tiles)
            else:
                pixel_y = self._sample_int(orig_h - 1, self.n_tiles)
                pixel_x = self._sample_int(orig_w - 1, self.n_tiles)
                start_y = pixel_y - self.tile_h // 2
                start_x = pixel_x - self.tile_w // 2
        else:
            batch_idx_flat, start_y_flat, start_x_flat = indices
            start_y = start_y_flat.view(B, self.n_tiles)
            start_x = start_x_flat.view(B, self.n_tiles)

        max_start_y = padded_H - self.tile_h
        max_start_x = padded_W - self.tile_w
        start_y = torch.clamp(start_y, min=0, max=max_start_y)
        start_x = torch.clamp(start_x, min=0, max=max_start_x)

        off_y = torch.arange(self.tile_h, device=device).view(1, 1, self.tile_h, 1)
        off_x = torch.arange(self.tile_w, device=device).view(1, 1, 1, self.tile_w)
        y_idx = start_y.unsqueeze(-1).unsqueeze(-1) + off_y
        x_idx = start_x.unsqueeze(-1).unsqueeze(-1) + off_x
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(-1, self.n_tiles, self.tile_h, self.tile_w)

        imgs_nhwc = images_2d.permute(0, 2, 3, 1)
        tiles = imgs_nhwc[batch_idx, y_idx, x_idx]
        tiles = tiles.permute(0, 1, 4, 2, 3).contiguous()

        batch_idx_flat = torch.repeat_interleave(torch.arange(B, device=device), self.n_tiles)
        return tiles, (batch_idx_flat, start_y.flatten(), start_x.flatten())
