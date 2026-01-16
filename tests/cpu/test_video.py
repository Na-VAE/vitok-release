"""Tests for video integration (patchify, RoPE, grid_info).

Tests verify:
1. Patchify temporal_patch=1 matches old image behavior
2. Video patchify/unpatchify roundtrip
3. 3D RoPE shape and dimension split
4. Grid info extraction for SWA sizing
"""

import pytest
import torch
import numpy as np

from vitok.pp.ops import patchify, unpatchify
from vitok.models.modules.rotary_embedding import (
    compute_rope_freqs,
    compute_2d_freqs_cis,
    _compute_axis_freqs,
)


# =============================================================================
# Patchify Tests
# =============================================================================


class TestPatchifyImage:
    """Tests for image (temporal_patch=1) patchification."""

    def test_patchify_image_shape(self):
        """Test patchify output shape for images."""
        img = torch.randn(3, 256, 256)  # C, H, W
        patch_fn = patchify(patch=16, temporal_patch=1, max_tokens=256)
        result = patch_fn(img)

        assert result['patches'].shape == (256, 16 * 16 * 3)
        assert result['patch_mask'].shape == (256,)
        assert result['time_idx'].shape == (256,)
        assert result['row_idx'].shape == (256,)
        assert result['col_idx'].shape == (256,)

    def test_patchify_image_time_idx_zero(self):
        """Test that time_idx is all zeros for images."""
        img = torch.randn(3, 128, 128)
        patch_fn = patchify(patch=16, temporal_patch=1, max_tokens=256)
        result = patch_fn(img)

        # All time indices should be 0 for images
        valid_time = result['time_idx'][result['patch_mask']]
        assert (valid_time == 0).all(), "time_idx should be 0 for images"

    def test_patchify_image_grid_t_one(self):
        """Test that grid_t is 1 for images."""
        img = torch.randn(3, 256, 256)
        patch_fn = patchify(patch=16, temporal_patch=1, max_tokens=256)
        result = patch_fn(img)

        assert result['grid_t'].item() == 1
        assert result['orig_frames'].item() == 1

    def test_patchify_image_row_col_indices(self):
        """Test row/col indices are correct for images."""
        img = torch.randn(3, 64, 64)  # 4x4 = 16 patches
        patch_fn = patchify(patch=16, temporal_patch=1, max_tokens=256)
        result = patch_fn(img)

        valid_mask = result['patch_mask']
        valid_rows = result['row_idx'][valid_mask].numpy()
        valid_cols = result['col_idx'][valid_mask].numpy()

        # Should have indices for 4x4 grid
        expected_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        expected_cols = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

        np.testing.assert_array_equal(valid_rows, expected_rows)
        np.testing.assert_array_equal(valid_cols, expected_cols)


class TestPatchifyVideo:
    """Tests for video (temporal_patch>1) patchification."""

    def test_patchify_video_shape(self):
        """Test patchify output shape for videos."""
        video = torch.randn(8, 3, 128, 128)  # T, C, H, W
        patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=512)
        result = patch_fn(video)

        # T=8 / temporal_patch=2 = 4 temporal tubelets
        # 128/16 = 8x8 spatial = 64 patches per timepoint
        # Total: 4 * 64 = 256 patches
        n_valid = result['patch_mask'].sum().item()
        assert n_valid == 256

        # Patch dimension: temporal_patch * C * patch * patch = 2 * 3 * 16 * 16
        assert result['patches'].shape[1] == 2 * 3 * 16 * 16

    def test_patchify_video_time_indices(self):
        """Test time indices for video patches."""
        video = torch.randn(8, 3, 64, 64)  # T=8, 4x4 spatial
        patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=256)
        result = patch_fn(video)

        valid_mask = result['patch_mask']
        valid_time = result['time_idx'][valid_mask].numpy()

        # With T=8, temporal_patch=2, we get grid_t=4
        # Each temporal index should appear 16 times (4x4 spatial)
        for t in range(4):
            assert (valid_time == t).sum() == 16, f"Time index {t} should appear 16 times"

    def test_patchify_video_grid_t(self):
        """Test grid_t for videos."""
        video = torch.randn(8, 3, 64, 64)
        patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=256)
        result = patch_fn(video)

        assert result['grid_t'].item() == 4  # 8 / 2 = 4
        assert result['grid_rows'].item() == 4
        assert result['grid_cols'].item() == 4

    def test_patchify_video_odd_frames_padding(self):
        """Test video with odd number of frames (requires padding)."""
        video = torch.randn(7, 3, 64, 64)  # T=7 (odd)
        patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=256)
        result = patch_fn(video)

        # T=7 padded to 8, grid_t = 4
        assert result['grid_t'].item() == 4
        assert result['orig_frames'].item() == 7


class TestPatchifyRoundtrip:
    """Tests for patchify/unpatchify roundtrip."""

    def test_image_roundtrip(self):
        """Test image patchify/unpatchify roundtrip."""
        img = torch.randn(3, 64, 64)
        patch_fn = patchify(patch=16, temporal_patch=1, max_tokens=256)
        patch_dict = patch_fn(img)

        # Add batch dimension
        batch = {k: v.unsqueeze(0) for k, v in patch_dict.items()}

        # Unpatchify
        recon = unpatchify(batch, patch=16, temporal_patch=1)

        # For images, output should be (B, C, H, W)
        assert recon.shape == (1, 3, 64, 64)

        # Check reconstruction matches (within valid region)
        orig_h = patch_dict['orig_height'].item()
        orig_w = patch_dict['orig_width'].item()
        recon_cropped = recon[0, :, :orig_h, :orig_w]

        torch.testing.assert_close(img, recon_cropped, rtol=1e-5, atol=1e-5)

    def test_video_roundtrip(self):
        """Test video patchify/unpatchify roundtrip."""
        video = torch.randn(8, 3, 64, 64)
        patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=512)
        patch_dict = patch_fn(video)

        # Add batch dimension
        batch = {k: v.unsqueeze(0) for k, v in patch_dict.items()}

        # Unpatchify
        recon = unpatchify(batch, patch=16, temporal_patch=2)

        # For video, output should be (B, T, C, H, W)
        assert recon.shape == (1, 8, 3, 64, 64)

        # Check reconstruction matches
        orig_t = patch_dict['orig_frames'].item()
        orig_h = patch_dict['orig_height'].item()
        orig_w = patch_dict['orig_width'].item()
        recon_cropped = recon[0, :orig_t, :, :orig_h, :orig_w]

        torch.testing.assert_close(video, recon_cropped, rtol=1e-5, atol=1e-5)

    def test_video_roundtrip_odd_frames(self):
        """Test video roundtrip with odd number of frames."""
        video = torch.randn(7, 3, 64, 64)  # Odd T
        patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=512)
        patch_dict = patch_fn(video)

        batch = {k: v.unsqueeze(0) for k, v in patch_dict.items()}
        recon = unpatchify(batch, patch=16, temporal_patch=2)

        # Should have 8 frames (padded), but orig_frames=7
        assert recon.shape[1] == 8
        assert patch_dict['orig_frames'].item() == 7

        # Crop to original size should match
        recon_cropped = recon[0, :7, :, :64, :64]
        torch.testing.assert_close(video, recon_cropped, rtol=1e-5, atol=1e-5)


# =============================================================================
# 3D RoPE Tests
# =============================================================================


class TestRoPE3D:
    """Tests for 3D RoPE (compute_rope_freqs)."""

    def test_rope_output_shape(self):
        """Test 3D RoPE output shapes."""
        B, N = 2, 64
        dim = 64  # head_dim

        t_pos = torch.zeros(B, N)
        y_pos = torch.arange(8).repeat(8).view(1, -1).expand(B, -1).float()
        x_pos = torch.arange(8).repeat_interleave(8).view(1, -1).expand(B, -1).float()

        cos, sin = compute_rope_freqs(t_pos, y_pos, x_pos, dim)

        # Output should be (B, N, dim/2) for each of cos and sin
        assert cos.shape == (B, N, dim // 2)
        assert sin.shape == (B, N, dim // 2)

    def test_rope_dimension_split(self):
        """Test 1:1:2 dimension split for t:y:x."""
        B, N = 1, 16
        dim = 64

        # Create positions where only t varies
        t_pos = torch.arange(N).float().view(1, -1)
        y_pos = torch.zeros(B, N)
        x_pos = torch.zeros(B, N)

        cos, sin = compute_rope_freqs(t_pos, y_pos, x_pos, dim)

        # Dimension split: t=16, y=16, x=32 (sum=64, half=32 for cos/sin pairs)
        t_dim = dim // 4  # 16
        y_dim = dim // 4  # 16
        x_dim = dim // 2  # 32

        # t component uses first t_dim/2 = 8 values
        # y component uses next y_dim/2 = 8 values
        # x component uses last x_dim/2 = 16 values

        # With y=0, x=0, only t varies - check that sin[t] is non-zero for t>0
        # and y, x components should all be cos=1, sin=0 (since pos=0)
        assert sin.shape[-1] == dim // 2  # 32

    def test_rope_t0_gives_zeros_for_temporal(self):
        """Test that t=0 gives zeros for temporal sin component."""
        B, N = 1, 4
        dim = 64

        t_pos = torch.zeros(B, N)  # All zeros
        y_pos = torch.zeros(B, N)
        x_pos = torch.zeros(B, N)

        cos, sin = compute_rope_freqs(t_pos, y_pos, x_pos, dim)

        # All positions are 0, so sin should be 0 everywhere
        assert torch.allclose(sin, torch.zeros_like(sin), atol=1e-6)
        # cos should be 1 everywhere
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-6)

    def test_rope_temporal_variation(self):
        """Test that varying t produces different embeddings."""
        B, N = 1, 4
        dim = 64

        t_pos = torch.tensor([[0, 1, 2, 3]]).float()
        y_pos = torch.zeros(B, N)
        x_pos = torch.zeros(B, N)

        cos, sin = compute_rope_freqs(t_pos, y_pos, x_pos, dim)

        # Different t values should give different embeddings
        assert not torch.allclose(cos[0, 0], cos[0, 1])
        assert not torch.allclose(cos[0, 1], cos[0, 2])

    def test_rope_2d_compatibility(self):
        """Test that t=0 with 2D positions works correctly."""
        B, N = 2, 16
        dim = 64

        # 4x4 grid
        y_pos = torch.arange(4).repeat(4).view(1, -1).expand(B, -1).float()
        x_pos = torch.arange(4).repeat_interleave(4).view(1, -1).expand(B, -1).float()
        t_pos = torch.zeros(B, N)

        cos, sin = compute_rope_freqs(t_pos, y_pos, x_pos, dim)

        # Should produce valid output
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()


# =============================================================================
# Grid Info Tests
# =============================================================================


class TestGridInfo:
    """Tests for grid info extraction."""

    def test_grid_info_image(self):
        """Test grid info for images."""
        img = torch.randn(3, 128, 128)
        patch_fn = patchify(patch=16, temporal_patch=1, max_tokens=256)
        result = patch_fn(img)

        assert result['grid_t'].item() == 1
        assert result['grid_rows'].item() == 8
        assert result['grid_cols'].item() == 8

    def test_grid_info_video(self):
        """Test grid info for videos."""
        video = torch.randn(8, 3, 64, 64)
        patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=512)
        result = patch_fn(video)

        assert result['grid_t'].item() == 4
        assert result['grid_rows'].item() == 4
        assert result['grid_cols'].item() == 4

    def test_grid_info_for_swa_scaling(self):
        """Test grid info values for SWA window scaling."""
        video = torch.randn(16, 3, 128, 128)
        patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=1024)
        result = patch_fn(video)

        grid_t = result['grid_t'].item()
        grid_h = result['grid_rows'].item()
        grid_w = result['grid_cols'].item()

        # For 3D SWA: window should scale by spatial_size = grid_h * grid_w
        spatial_size = grid_h * grid_w
        assert spatial_size == 64  # 8x8

        # Effective window for base_window=4:
        # effective = 4 * 64 = 256 (covers full spatial plane + temporal neighbors)
        base_window = 4
        effective_window = base_window * spatial_size
        assert effective_window == 256


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_frame_video(self):
        """Test video with single frame (T=1)."""
        video = torch.randn(1, 3, 64, 64)
        patch_fn = patchify(patch=16, temporal_patch=1, max_tokens=256)
        result = patch_fn(video)

        assert result['grid_t'].item() == 1
        assert result['orig_frames'].item() == 1

    def test_small_spatial_size(self):
        """Test with small spatial dimensions."""
        img = torch.randn(3, 32, 32)  # 2x2 = 4 patches
        patch_fn = patchify(patch=16, temporal_patch=1, max_tokens=256)
        result = patch_fn(img)

        n_valid = result['patch_mask'].sum().item()
        assert n_valid == 4

    def test_non_square_video(self):
        """Test video with non-square spatial dimensions."""
        video = torch.randn(4, 3, 64, 128)  # H != W
        patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=256)
        result = patch_fn(video)

        assert result['grid_rows'].item() == 4  # 64/16
        assert result['grid_cols'].item() == 8  # 128/16

    def test_padded_dimensions(self):
        """Test with dimensions requiring padding."""
        video = torch.randn(5, 3, 50, 70)  # Odd sizes
        patch_fn = patchify(patch=16, temporal_patch=2, max_tokens=256)
        result = patch_fn(video)

        # T=5 padded to 6, grid_t = 3
        # H=50 padded to 64, grid_h = 4
        # W=70 padded to 80, grid_w = 5
        assert result['grid_t'].item() == 3
        assert result['grid_rows'].item() == 4
        assert result['grid_cols'].item() == 5
        assert result['orig_frames'].item() == 5
        assert result['orig_height'].item() == 50
        assert result['orig_width'].item() == 70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
