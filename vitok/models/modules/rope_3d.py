"""3D Rotary Position Embeddings (RoPE) for spatiotemporal sequences.

Extends 2D spatial RoPE with temporal dimension for video processing.

NOTE: This module is created for future use but is NOT yet integrated
into the AE model. Current image ViToks use 2D RoPE. This is prepared
for future finetuning on video data.
"""

from typing import Optional, Tuple

import torch

from vitok.models.modules.rotary_embedding import _compute_axis_freqs


def compute_3d_freqs_cis(
    y_positions: torch.Tensor,
    x_positions: torch.Tensor,
    t_positions: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
    axis_inv_freq: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """3D rotary frequencies for spatiotemporal coordinates.

    Splits the head dimension into 3 equal parts for row, column, and time axes.
    Each axis gets dim//3 dimensions, and the final frequencies are concatenated.

    Args:
        y_positions: Row indices [B, N] or [N]
        x_positions: Column indices [B, N] or [N]
        t_positions: Time/frame indices [B, N] or [N]
        dim: Head dimension (must be divisible by 6 for 3 axes with even splits)
        theta: RoPE base frequency (default: 10000.0)
        axis_inv_freq: Optional precomputed inverse frequencies for one axis

    Returns:
        freqs_cos: Cosine frequencies [B, N, dim//2] or [N, dim//2]
        freqs_sin: Sine frequencies [B, N, dim//2] or [N, dim//2]

    Raises:
        ValueError: If dim is not divisible by 6, or if position tensor shapes don't match

    Example:
        >>> B, N, head_dim = 2, 256, 96  # head_dim divisible by 6
        >>> row_idx = torch.randint(0, 16, (B, N))
        >>> col_idx = torch.randint(0, 16, (B, N))
        >>> time_idx = torch.zeros(B, N)  # All same frame
        >>> freqs_cos, freqs_sin = compute_3d_freqs_cis(row_idx, col_idx, time_idx, head_dim)
        >>> freqs_cos.shape
        torch.Size([2, 256, 48])  # dim // 2
    """
    if y_positions.shape != x_positions.shape or y_positions.shape != t_positions.shape:
        raise ValueError(
            "y_positions, x_positions, and t_positions must have matching shapes. "
            f"Got y:{y_positions.shape}, x:{x_positions.shape}, t:{t_positions.shape}"
        )

    if dim % 6 != 0:
        raise ValueError(
            f"3D RoPE requires head dimension divisible by 6 (3 axes x 2 for real/imag). "
            f"Got dim={dim}. Try dim={dim - (dim % 6)} or dim={(dim // 6 + 1) * 6}."
        )

    axis_dim = dim // 3  # Each axis gets 1/3 of the head dimension

    cos_y, sin_y = _compute_axis_freqs(
        y_positions,
        axis_dim,
        theta,
        inv_freq=axis_inv_freq,
    )
    cos_x, sin_x = _compute_axis_freqs(
        x_positions,
        axis_dim,
        theta,
        inv_freq=axis_inv_freq,
    )
    cos_t, sin_t = _compute_axis_freqs(
        t_positions,
        axis_dim,
        theta,
        inv_freq=axis_inv_freq,
    )

    freqs_cos = torch.cat((cos_y, cos_x, cos_t), dim=-1)
    freqs_sin = torch.cat((sin_y, sin_x, sin_t), dim=-1)

    return freqs_cos, freqs_sin


__all__ = ["compute_3d_freqs_cis"]
