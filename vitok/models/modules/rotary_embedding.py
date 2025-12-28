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
    """Apply rotary embeddings to query and key tensors."""
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
