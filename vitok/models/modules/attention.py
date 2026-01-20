"""Attention module using Flash Attention."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from vitok.models.modules.norm import RMSNorm
from vitok.models.modules.rotary_embedding import apply_rotary_emb


class Attention(nn.Module):
    """Multi-headed attention with Flash Attention backend."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        fused: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.fused = fused

        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)

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
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sliding_window: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            hidden_states: [B, N, C]
            freqs_cis: RoPE (cos, sin) tuple
            sliding_window: Window size for local attention (None = full attention)
        """
        B, N, C = hidden_states.shape

        if self.fused:
            qkv = self.qkv_proj(hidden_states).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
        else:
            q = self.q(hidden_states).reshape(B, N, self.num_heads, self.head_dim)
            k = self.k(hidden_states).reshape(B, N, self.num_heads, self.head_dim)
            v = self.v(hidden_states).reshape(B, N, self.num_heads, self.head_dim)

        # QK norm
        q, k = self.norm_q(q), self.norm_k(k)

        # RoPE
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis[0], freqs_cis[1])

        # Flash attention expects [B, N, H, D] and bf16
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        # Sliding window: flash_attn uses (left, right) tuple, -1 means infinite
        window_size = (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)

        from flash_attn import flash_attn_func
        attn = flash_attn_func(q, k, v, window_size=window_size)
        attn = attn.reshape(B, N, C)

        return self.out_proj(attn)
