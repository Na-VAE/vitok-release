"""Attention module with Flash Attention and SDPA backends."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple

from vitok.models.modules.norm import RMSNorm
from vitok.models.modules.rotary_embedding import apply_rotary_emb

# Optional flash_attn import
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False


class Attention(nn.Module):
    """Multi-headed attention with Flash Attention or SDPA backend."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        fused: bool = True,
        eps: float = 1e-6,
        backend: Literal["flash", "sdpa"] = "flash",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.fused = fused
        self.backend = backend

        # Validate backend
        if backend == "flash" and not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "flash_attn is not available. Install it with: pip install flash-attn --no-build-isolation, "
                "or use backend='sdpa' for PyTorch native SDPA."
            )

        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)

        if self.fused:
            self.qkv_proj = nn.Linear(self.dim, 3 * self.dim, bias=False)
        else:
            self.q = nn.Linear(self.dim, self.dim, bias=False)
            self.k = nn.Linear(self.dim, self.dim, bias=False)
            self.v = nn.Linear(self.dim, self.dim, bias=False)

        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SDPA attention with optional mask. No SWA support.

        Args:
            q, k, v: [B, H, N, D]
            attn_mask: [B, 1, N, N] bool mask where True = attend, False = mask out

        Returns:
            [B, H, N, D]
        """
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sliding_window: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            hidden_states: [B, N, C]
            freqs_cis: RoPE (cos, sin) tuple
            sliding_window: Window size for local attention (None = full attention)
                           Only supported with flash backend; ignored with sdpa.
            attn_mask: [B, 1, N, N] bool attention mask for SDPA backend
                      True = attend, False = mask out. Ignored with flash backend.
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

        if self.backend == "flash":
            # Flash attention expects [B, N, H, D] and bf16
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

            # Sliding window: flash_attn uses (left, right) tuple, -1 means infinite
            window_size = (sliding_window, sliding_window) if sliding_window is not None else (-1, -1)

            attn = flash_attn_func(q, k, v, window_size=window_size)
            attn = attn.reshape(B, N, C)
        else:
            # SDPA backend: transpose for [B, H, N, D] format
            q = q.transpose(1, 2)  # [B, N, H, D] -> [B, H, N, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn = self._sdpa_attention(q, k, v, attn_mask)

            # Back to [B, N, H, D] then reshape
            attn = attn.transpose(1, 2).reshape(B, N, C)

        return self.out_proj(attn)
