"""Rotary Position Embeddings (RoPE) for 1D and 2D sequences."""

import torch
from typing import Optional, Tuple


def compute_inv_freq(dim: int, theta: float, device: torch.device) -> torch.Tensor:
    """Precompute inverse frequencies for a given rotary dimension."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE axis dimension must be even, got dim={dim}")

    with torch.amp.autocast(enabled=False, device_type="cuda"):
        return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))


def _compute_axis_freqs(
    positions: torch.Tensor,
    dim: int,
    theta: float,
    inv_freq: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cosine/sine pairs for a single RoPE axis."""
    if inv_freq is None:
        inv_freq = compute_inv_freq(dim, theta, positions.device)
    else:
        inv_freq = inv_freq.to(device=positions.device)

    with torch.amp.autocast(enabled=False, device_type="cuda"):
        freqs = positions.to(torch.float32)[..., None] * inv_freq
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def compute_freqs_cis(
    t: torch.Tensor,
    dim: int = 768,
    theta: float = 10000.0,
    inv_freq: Optional[torch.Tensor] = None,
):
    """1D rotary frequencies for sequence indices ``t``."""
    freqs_cos, freqs_sin = _compute_axis_freqs(t, dim, theta, inv_freq=inv_freq)
    return freqs_cos, freqs_sin


def compute_2d_freqs_cis(
    y_positions: torch.Tensor,
    x_positions: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
    axis_inv_freq: Optional[torch.Tensor] = None,
):
    """2D rotary frequencies for spatial coordinates."""
    if y_positions.shape != x_positions.shape:
        raise ValueError("x_positions and y_positions must have matching shapes")
    if dim % 4 != 0:
        raise ValueError("2D RoPE requires head dimension divisible by 4")

    axis_dim = dim // 2
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

    freqs_cos = torch.cat((cos_y, cos_x), dim=-1)
    freqs_sin = torch.cat((sin_y, sin_x), dim=-1)
    return freqs_cos, freqs_sin


def compute_rope_freqs(
    t_positions: torch.Tensor,
    y_positions: torch.Tensor,
    x_positions: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
    axis_inv_freq: Optional[torch.Tensor] = None,
):
    """Unified RoPE for 2D/3D positions.

    For images (t=0 for all tokens): temporal frequencies contribute nothing,
    reducing to 2D RoPE behavior.
    For video (t>0): full 3D positional encoding.

    Dimension split: t:y:x = 1:1:2 for head_dim=64 compatibility
        - t_dim = dim // 4
        - y_dim = dim // 4
        - x_dim = dim // 2

    Args:
        t_positions: [B, N] temporal indices (0 for images)
        y_positions: [B, N] row indices
        x_positions: [B, N] col indices
        dim: Head dimension (must be divisible by 4)
        theta: RoPE base frequency
        axis_inv_freq: Optional precomputed inverse frequencies

    Returns:
        (freqs_cos, freqs_sin): Concatenated frequencies for all axes
    """
    if not (y_positions.shape == x_positions.shape == t_positions.shape):
        raise ValueError("All position tensors must have matching shapes")
    if dim % 4 != 0:
        raise ValueError("3D RoPE requires head dimension divisible by 4")

    # 1:1:2 split for t:y:x
    t_dim = dim // 4
    y_dim = dim // 4
    x_dim = dim // 2

    # Compute frequencies for each axis
    cos_t, sin_t = _compute_axis_freqs(t_positions, t_dim, theta, inv_freq=axis_inv_freq)
    cos_y, sin_y = _compute_axis_freqs(y_positions, y_dim, theta, inv_freq=axis_inv_freq)
    cos_x, sin_x = _compute_axis_freqs(x_positions, x_dim, theta, inv_freq=axis_inv_freq)

    # Concatenate: [t_dim/2, y_dim/2, x_dim/2] pairs = dim/2 total pairs
    freqs_cos = torch.cat((cos_t, cos_y, cos_x), dim=-1)
    freqs_sin = torch.cat((sin_t, sin_y, sin_x), dim=-1)

    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape frequency tensor for broadcasting with input."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [1] * (ndim - 2) + [x.shape[-2], x.shape[-1]]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [1] * (ndim - 3) + [x.shape[-3], x.shape[-2], x.shape[-1]]
    elif freqs_cis.shape == (x.shape[0], x.shape[-2], x.shape[-1]):
        shape = [x.shape[0]] + [1] * (ndim - 3) + [x.shape[-2], x.shape[-1]]
    else:
        raise ValueError(
            "freqs_cis shape must match either sequence, head, or batch dimensions"
        )
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Note: Rotation is done in the input dtype (bf16) to save memory at high
    resolutions. Frequency computation uses float32 for numerical stability,
    but the multiply-add rotation is safe in bf16.
    """
    xq_r, xq_i = xq.reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    # Cast freqs to input dtype (bf16) - rotation is numerically safe in bf16
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r).to(xq_r.dtype)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r).to(xq_r.dtype)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out, xk_out
