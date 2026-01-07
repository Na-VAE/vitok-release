import torch
import torch.nn.functional as F
import pytest

from vitok.naflex_io import RandomTileSampler


class LegacyRandomTileSampler:
    """Legacy sampler kept for compatibility tests."""

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
        patches_dict: dict[str, torch.Tensor],
        indices: tuple | None = None,
    ) -> tuple:
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


def _make_inputs():
    B, C, H, W = 2, 3, 40, 52
    images = torch.arange(B * C * H * W, dtype=torch.float32).view(B, C, H, W)
    patches_dict = {
        "original_height": torch.tensor([32, 40]),
        "original_width": torch.tensor([48, 52]),
    }
    return images, patches_dict


def _assert_indices_in_bounds(indices, padded_H, padded_W, tile_h, tile_w):
    _, start_y, start_x = indices
    assert start_y.min().item() >= 0
    assert start_x.min().item() >= 0
    assert start_y.max().item() <= padded_H - tile_h
    assert start_x.max().item() <= padded_W - tile_w


def _assert_indices_in_original(indices, orig_h, orig_w, tile_h, tile_w):
    _, start_y, start_x = indices
    B = orig_h.numel()
    start_y = start_y.view(B, -1)
    start_x = start_x.view(B, -1)
    max_sy = (orig_h - tile_h).view(B, 1)
    max_sx = (orig_w - tile_w).view(B, 1)
    assert (start_y >= 0).all()
    assert (start_x >= 0).all()
    assert (start_y <= max_sy).all()
    assert (start_x <= max_sx).all()


@pytest.mark.parametrize("coverage", ["uniform_pixel", "uniform_topleft"])
def test_random_tile_sampler_compatibility(coverage):
    images, patches_dict = _make_inputs()
    B, C, H, W = images.shape
    tile_h, tile_w = 16, 20
    n_tiles = 5

    legacy = LegacyRandomTileSampler(
        n_tiles=n_tiles,
        tile_size=(tile_h, tile_w),
        coverage=coverage,
    )
    new = RandomTileSampler(
        n_tiles=n_tiles,
        tile_size=(tile_h, tile_w),
        coverage=coverage,
    )

    torch.manual_seed(123)
    legacy_tiles, legacy_idx = legacy(images, patches_dict)
    torch.manual_seed(123)
    new_tiles, new_idx = new(images, patches_dict)

    assert legacy_tiles.shape == (B, n_tiles, C, tile_h, tile_w)
    assert new_tiles.shape == (B, n_tiles, C, tile_h, tile_w)

    expected_batch = torch.repeat_interleave(torch.arange(B), n_tiles)
    assert torch.equal(legacy_idx[0], expected_batch)
    assert torch.equal(new_idx[0], expected_batch)

    padded_H = max(H, tile_h)
    padded_W = max(W, tile_w)
    _assert_indices_in_bounds(legacy_idx, padded_H, padded_W, tile_h, tile_w)
    _assert_indices_in_bounds(new_idx, padded_H, padded_W, tile_h, tile_w)

    tiles_from_legacy, _ = new(images, patches_dict, indices=legacy_idx)
    tiles_from_new, _ = legacy(images, patches_dict, indices=new_idx)
    assert torch.allclose(tiles_from_legacy, legacy_tiles)
    assert torch.allclose(tiles_from_new, new_tiles)


def test_uniform_pixel_stays_in_original_bounds_when_possible():
    images, patches_dict = _make_inputs()
    tile_h, tile_w = 16, 20
    n_tiles = 7

    sampler = RandomTileSampler(
        n_tiles=n_tiles,
        tile_size=(tile_h, tile_w),
        coverage="uniform_pixel",
    )
    torch.manual_seed(7)
    _, indices = sampler(images, patches_dict)
    _assert_indices_in_original(
        indices,
        patches_dict["original_height"],
        patches_dict["original_width"],
        tile_h,
        tile_w,
    )
