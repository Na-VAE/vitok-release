"""Attention modules with optional sliding window."""

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Dict, Optional, Tuple

from vitok.models.modules.norm import RMSNorm, LayerNorm
from vitok.models.modules.rotary_embedding import apply_rotary_emb


def _make_2d_mask_mod(window: int, grid_size: int, num_special: int):
    """2D sliding window mask_mod with special tokens having full attention.

    Special tokens (indices 0 to S-1) can attend to everything.
    Patch tokens (indices S onwards) use 2D local box [r±window, c±window].
    """
    def mask_mod(b, h, q_idx, kv_idx):
        special_case = (q_idx < num_special) | (kv_idx < num_special)

        q_patch = q_idx - num_special
        kv_patch = kv_idx - num_special
        q_row = q_patch // grid_size
        q_col = q_patch % grid_size
        kv_row = kv_patch // grid_size
        kv_col = kv_patch % grid_size
        swa_case = ((q_row - kv_row).abs() <= window) & ((q_col - kv_col).abs() <= window)

        return special_case | swa_case
    return mask_mod


def create_2d_block_mask(
    window: int,
    seq_len: int,
    num_special: int,
    device: torch.device,
):
    """Create 2D sliding window block mask for SWA attention.

    IMPORTANT: Call this function OUTSIDE of torch.compile traced code.

    Args:
        window: Window radius for 2D local attention
        seq_len: Total sequence length including special tokens
        num_special: Number of special tokens (CLS, register tokens)
        device: Target device

    Returns:
        BlockMask object for use with flex_attention
    """
    patch_len = seq_len - num_special
    gs = int(patch_len ** 0.5)
    if gs * gs != patch_len:
        raise ValueError(
            f"patch_len {patch_len} (seq_len={seq_len} - num_special={num_special}) "
            f"must be a perfect square for 2D SWA. Got sqrt={patch_len**0.5:.2f}"
        )
    mask_mod = _make_2d_mask_mod(window, gs, num_special)
    return create_block_mask(
        mask_mod, B=None, H=None,
        Q_LEN=seq_len, KV_LEN=seq_len, device=device,
    )


class Attention(nn.Module):
    """Multi-headed attention with optional sliding window."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_special_tokens: int = 0,
        fused: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_special_tokens = num_special_tokens
        self.fused = fused

        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

        if self.fused:
            self.qkv_proj = nn.Linear(self.dim, 3 * self.dim, bias=False)
        else:
            self.q = nn.Linear(self.dim, self.dim, bias=False)
            self.k = nn.Linear(self.dim, self.dim, bias=False)
            self.v = nn.Linear(self.dim, self.dim, bias=False)

        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        score_mod=None,
        block_mask=None,
    ):
        """
        Args:
            hidden_states: [B, N, C] where N = num_special_tokens + patch_len
            freqs_cis: RoPE (cos, sin) tuple
            sliding_window: Window radius for 2D local attention (None = full attention)
            score_mod: Optional score modification function for padding masks
            block_mask: Pre-computed block mask for SWA (for torch.compile fullgraph)
        """
        B, N, C = hidden_states.shape

        if self.fused:
            qkv = self.qkv_proj(hidden_states).reshape(B, N, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = self.q(hidden_states).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k(hidden_states).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(hidden_states).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.norm_q(q), self.norm_k(k)
        q, k = apply_rotary_emb(q, k, freqs_cis[0], freqs_cis[1])
        attn = flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(attn)


class CrossAttention(Attention):
    """Cross attention for conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
    ):
        super().__init__(dim=dim, num_heads=num_heads, fused=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        """
        Args:
            x: [B, L1, C] queries
            context: [B, L2, C] keys/values
        """
        B = x.shape[0]
        q = self.q(x).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.norm_q(q), self.norm_k(k)
        attn = flex_attention(q, k, v)
        attn = attn.transpose(1, 2).flatten(2)
        return self.out_proj(attn)
