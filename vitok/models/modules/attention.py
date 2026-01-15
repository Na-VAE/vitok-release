"""Attention modules with backend selection (flex, flash, sdpa)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from typing import Dict, Literal, Optional, Tuple

from vitok.models.modules.norm import RMSNorm, LayerNorm
from vitok.models.modules.rotary_embedding import apply_rotary_emb

AttnBackend = Literal["flex", "flash", "sdpa"]

# Optional flash-attn import
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False


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
    """Multi-headed attention with backend selection (flex, flash, sdpa)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_special_tokens: int = 0,
        fused: bool = True,
        qk_norm: str = "rmsnorm",
        eps: float = 1e-6,
        backend: AttnBackend = "flex",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_special_tokens = num_special_tokens
        self.fused = fused
        self.backend = backend

        # Validate flash backend availability
        if backend == "flash" and not FLASH_ATTN_AVAILABLE:
            raise ImportError("flash-attn not installed. Install with: pip install flash-attn")

        if qk_norm == "rmsnorm":
            self.norm_q = RMSNorm(self.head_dim, eps=eps)
            self.norm_k = RMSNorm(self.head_dim, eps=eps)
        elif qk_norm == "layernorm":
            self.norm_q = LayerNorm(self.head_dim, eps=eps)
            self.norm_k = LayerNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        if self.fused:
            self.qkv_proj = nn.Linear(self.dim, 3 * self.dim, bias=False)
        else:
            self.q = nn.Linear(self.dim, self.dim, bias=False)
            self.k = nn.Linear(self.dim, self.dim, bias=False)
            self.v = nn.Linear(self.dim, self.dim, bias=False)

        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self._block_mask_cache: Dict[Tuple, object] = {}

    def _get_block_mask(
        self,
        window: int,
        seq_len: int,
        device: torch.device,
    ):
        """Get cached block mask for sliding window attention."""
        S = self.num_special_tokens
        key = (window, seq_len, S, str(device))

        if key not in self._block_mask_cache:
            self._block_mask_cache[key] = create_2d_block_mask(
                window=window,
                seq_len=seq_len,
                num_special=S,
                device=device,
            )
        return self._block_mask_cache[key]

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sliding_window: Optional[int] = None,
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

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis[0], freqs_cis[1])

        # Select attention backend
        if self.backend == "flash":
            attn = self._flash_attention(q, k, v, sliding_window)
        elif self.backend == "flex":
            attn = self._flex_attention(q, k, v, sliding_window, score_mod, block_mask, N)
        else:  # sdpa
            attn = self._sdpa_attention(q, k, v)

        attn = attn.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(attn)

    def _flash_attention(self, q, k, v, sliding_window):
        """Flash Attention with 1D sliding window."""
        # flash_attn expects [B, N, H, D], we have [B, H, N, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if sliding_window is not None and sliding_window >= 0:
            window_size = (64, 64)  # 1D window of 64 tokens each direction
        else:
            window_size = (-1, -1)  # Full attention

        attn = flash_attn_func(q, k, v, window_size=window_size)
        return attn.transpose(1, 2)  # Back to [B, H, N, D]

    def _flex_attention(self, q, k, v, sliding_window, score_mod, block_mask, N):
        """Flex Attention with 2D sliding window."""
        if sliding_window is not None and sliding_window >= 0:
            if block_mask is None:
                block_mask = self._get_block_mask(sliding_window, N, q.device)
            return flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)
        else:
            return flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)

    def _sdpa_attention(self, q, k, v):
        """Standard scaled dot-product attention (full attention, no SWA)."""
        return F.scaled_dot_product_attention(q, k, v)


class CrossAttention(nn.Module):
    """Cross attention for conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: str = "rmsnorm",
        eps: float = 1e-6,
        backend: AttnBackend = "flex",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.backend = backend

        # Validate flash backend availability
        if backend == "flash" and not FLASH_ATTN_AVAILABLE:
            raise ImportError("flash-attn not installed. Install with: pip install flash-attn")

        if qk_norm == "rmsnorm":
            self.norm_q = RMSNorm(self.head_dim, eps=eps)
            self.norm_k = RMSNorm(self.head_dim, eps=eps)
        elif qk_norm == "layernorm":
            self.norm_q = LayerNorm(self.head_dim, eps=eps)
            self.norm_k = LayerNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        self.q = nn.Linear(self.dim, self.dim, bias=False)
        self.k = nn.Linear(self.dim, self.dim, bias=False)
        self.v = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        """
        Args:
            x: [B, L1, C] queries
            context: [B, L2, C] keys/values
        """
        B = x.size(0)

        q = self.q(x).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.norm_q(q), self.norm_k(k)

        if self.backend == "flash":
            # flash_attn expects [B, N, H, D]
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            attn = flash_attn_func(q, k, v)
            attn = attn.flatten(2)
        elif self.backend == "flex":
            attn = flex_attention(q, k, v)
            attn = attn.transpose(1, 2).flatten(2)
        else:  # sdpa
            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).flatten(2)

        return self.out_proj(attn)
