"""Diffusion Transformer for class-conditioned image generation."""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint as _checkpoint

from vitok.models.modules.attention import Attention, create_2d_block_mask
from vitok.models.modules.norm import LayerNorm
from vitok.models.modules.mlp import SwiGLU
from vitok.models.modules.layerscale import LayerScale
from vitok.models.modules.rotary_embedding import compute_2d_freqs_cis, compute_inv_freq

try:
    from torchao.float8 import convert_to_float8_training
except Exception:
    convert_to_float8_training = None


def timestep_embedding(t, dim, max_period=10000, dtype=torch.float32, device=None):
    """Create sinusoidal timestep embeddings."""
    t = t.to(dtype=dtype, device=device).unsqueeze(-1)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=dtype) / half)
    angles = t * freqs
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


NUM_MOD_PARAMS = 4


class DiTBlock(nn.Module):
    """DiT block with parallel attention + MLP and adaLN modulation."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.67,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-4,
        sliding_window: Optional[int] = None,
        mod_tanh: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.sliding_window = sliding_window
        self.mod_tanh = float(mod_tanh or 0.0)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = LayerNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, qk_norm="rmsnorm")
        self.mlp = SwiGLU(dim, hidden_dim=mlp_hidden_dim)
        self.modulation = nn.Parameter(torch.zeros(1, NUM_MOD_PARAMS, dim))
        self.layer_scale = LayerScale(dim, layer_scale_init) if use_layer_scale else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mod_params: torch.Tensor,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        block_mask=None,
    ) -> torch.Tensor:
        mod = mod_params + self.modulation
        shift, scale, gate_attn, gate_mlp = mod.unbind(dim=1)

        if self.mod_tanh > 0:
            scale = torch.tanh(scale) * self.mod_tanh
            gate_attn = torch.tanh(gate_attn) * self.mod_tanh
            gate_mlp = torch.tanh(gate_mlp) * self.mod_tanh

        x_mod = (1 + scale[:, None, :]) * self.norm(x) + shift[:, None, :]

        attn_out = self.attn(
            x_mod,
            freqs_cis=freqs_cis,
            score_mod=attn_mask,
            sliding_window=self.sliding_window,
            block_mask=block_mask,
        )
        mlp_out = self.mlp(x_mod)

        output = gate_attn[:, None, :] * attn_out + gate_mlp[:, None, :] * mlp_out
        return x + self.layer_scale(output)


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.linear = nn.Linear(dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, head_mod: torch.Tensor) -> torch.Tensor:
        shift, scale = head_mod.unbind(dim=1)
        x = (1 + scale[:, None, :]) * self.norm(x) + shift[:, None, :]
        return self.linear(x)


class DiT(nn.Module):
    """Diffusion Transformer for class-conditioned image generation."""

    def __init__(
        self,
        text_dim: int = 1000,
        code_width: int = 16,
        width: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        num_tokens: int = 256,
        checkpoint: int = 0,
        float8: bool = False,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-4,
        sw: Optional[int] = None,
        class_token: bool = False,
        reg_tokens: int = 0,
        train_seq_len: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        # Hardcoded hyperparameters
        rope_theta = 10000.0
        mlp_ratio = 2.67
        mod_tanh = 3.0
        sw_every = 2
        freq_dim = 256

        self.text_dim = text_dim
        self.code_width = code_width
        self.width = width
        self.depth = depth
        self.num_heads = num_heads
        self.checkpoint = checkpoint
        self.num_tokens = num_tokens
        self.rope_theta = rope_theta
        self.sw = sw if (sw is None or sw > 0) else None
        self.sw_every = sw_every
        self.mod_tanh = mod_tanh
        self.class_token = class_token
        self.reg_tokens = reg_tokens
        self.num_special_tokens = (1 if class_token else 0) + reg_tokens
        self.train_seq_len = train_seq_len

        self._inv_freq_cache: dict = {}
        self._block_mask = None

        self.input_proj = nn.Linear(code_width, width, bias=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, width)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, width)) if reg_tokens > 0 else None

        self.time_embedder = nn.Sequential(
            nn.Linear(freq_dim, width, bias=True),
            nn.SiLU(),
            nn.Linear(width, width, bias=True),
        )

        self.class_embedder = nn.Embedding(text_dim + 1, width)

        self.mod_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, NUM_MOD_PARAMS * width, bias=False),
        )

        self.head_mod_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, 2 * width, bias=False),
        )

        blocks = []
        for layer_idx in range(depth):
            sliding_window = None
            if self.sw is not None and ((layer_idx + 1) % sw_every == 0):
                sliding_window = self.sw
            block = DiTBlock(
                dim=width,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_layer_scale=use_layer_scale,
                layer_scale_init=layer_scale_init,
                sliding_window=sliding_window,
                mod_tanh=self.mod_tanh,
            )
            if float8:
                if convert_to_float8_training is None:
                    raise ImportError("torchao is required for float8 training.")
                convert_to_float8_training(block)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = FinalLayer(dim=width, out_dim=code_width)

        if self.sw is not None and self.train_seq_len is not None:
            self._block_mask = create_2d_block_mask(
                window=self.sw,
                seq_len=self.train_seq_len,
                num_special=self.num_special_tokens,
                device=torch.device('cuda'),
            )

    def _should_checkpoint(self, layer_idx: int) -> bool:
        if self.checkpoint == 0 or not self.training:
            return False
        return layer_idx % self.checkpoint == 0

    def _get_axis_inv_freq(self, axis_dim: int, device: torch.device) -> torch.Tensor:
        key = (str(device), axis_dim)
        cached = self._inv_freq_cache.get(key)
        if cached is None:
            cached = compute_inv_freq(axis_dim, self.rope_theta, device=device)
            self._inv_freq_cache[key] = cached
        return cached

    def _prepare_2d_positions(
        self,
        source: dict,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        yidx = source['yidx']
        xidx = source['xidx']
        if yidx.ndim == 1:
            yidx = yidx.unsqueeze(0)
        if xidx.ndim == 1:
            xidx = xidx.unsqueeze(0)
        if yidx.shape[0] == 1 and batch_size > 1:
            yidx = yidx.expand(batch_size, -1)
        if xidx.shape[0] == 1 and batch_size > 1:
            xidx = xidx.expand(batch_size, -1)
        return torch.stack(
            (yidx.to(device=device, dtype=torch.float32),
             xidx.to(device=device, dtype=torch.float32)),
            dim=-1,
        )

    def _make_square_positions(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        side = int(round(length ** 0.5))
        if side * side != length:
            raise ValueError(f"Cannot infer 2D positions: length {length} is not a perfect square")
        y, x = torch.meshgrid(
            torch.arange(side, device=device),
            torch.arange(side, device=device),
            indexing='ij',
        )
        pos = torch.stack((y.flatten().float(), x.flatten().float()), dim=-1)
        return pos.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def _compute_rope_freqs(
        self,
        positions_2d: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.width // self.num_heads
        axis_inv_freq = self._get_axis_inv_freq(head_dim // 2, device)
        freqs_cos, freqs_sin = compute_2d_freqs_cis(
            positions_2d[..., 0],
            positions_2d[..., 1],
            dim=head_dim,
            theta=self.rope_theta,
            axis_inv_freq=axis_inv_freq,
        )
        return freqs_cos.float(), freqs_sin.float()

    def forward(self, dit_dict: dict) -> torch.Tensor:
        """
        Forward pass.

        Args:
            dit_dict: Dictionary containing:
                - z: [B, L, code_width] latent tokens
                - t: [B] timesteps (0 to num_train_timesteps)
                - context: [B] class labels
                - yidx, xidx: [B, L] spatial positions (optional)
                - attention_mask: [B, L, L] attention mask (optional)

        Returns:
            [B, L, code_width] predicted velocity
        """
        z = dit_dict['z']
        t = dit_dict['t']
        context = dit_dict['context']

        B, L, _ = z.shape
        device = z.device

        x = self.input_proj(z)

        if self.cls_token is not None or self.reg_token is not None:
            tokens = []
            if self.cls_token is not None:
                tokens.append(self.cls_token.expand(B, -1, -1))
            if self.reg_token is not None:
                tokens.append(self.reg_token.expand(B, -1, -1))
            tokens.append(x)
            x = torch.cat(tokens, dim=1)

        if 'yidx' in dit_dict and 'xidx' in dit_dict:
            positions_2d = self._prepare_2d_positions(dit_dict, B, device)
        else:
            positions_2d = self._make_square_positions(B, L, device)

        t_emb = timestep_embedding(t, self.freq_dim, dtype=torch.float32, device=device)
        t_emb = self.time_embedder(t_emb.to(x.dtype))

        class_emb = self.class_embedder(context)
        if class_emb.ndim == 3:
            class_emb = class_emb.squeeze(1)

        vec = t_emb + class_emb
        mod_params = self.mod_proj(vec).view(B, NUM_MOD_PARAMS, self.width)
        head_mod = self.head_mod_proj(vec).view(B, 2, self.width)
        if self.mod_tanh > 0:
            head_shift, head_scale = head_mod.unbind(dim=1)
            head_scale = torch.tanh(head_scale) * self.mod_tanh
            head_mod = torch.stack((head_shift, head_scale), dim=1)

        freqs_cos, freqs_sin = self._compute_rope_freqs(positions_2d, device)
        if self.num_special_tokens > 0:
            S = self.num_special_tokens
            head_dim = freqs_cos.shape[-1]
            ones = torch.ones(B, S, head_dim, device=device, dtype=freqs_cos.dtype)
            zeros = torch.zeros_like(ones)
            freqs_cos = torch.cat([ones, freqs_cos], dim=1)
            freqs_sin = torch.cat([zeros, freqs_sin], dim=1)
        freqs_cis = (freqs_cos, freqs_sin)

        attn_mask = dit_dict.get('attention_mask', None)

        for layer_idx, block in enumerate(self.blocks):
            if self._should_checkpoint(layer_idx):
                def run_block(_x):
                    return block(_x, mod_params, freqs_cis, attn_mask, self._block_mask)
                x = _checkpoint(run_block, x, use_reentrant=False)
            else:
                x = block(x, mod_params, freqs_cis, attn_mask, self._block_mask)

        if self.num_special_tokens > 0:
            x = x[:, self.num_special_tokens:]

        return self.final_layer(x, head_mod)

    def fsdp_unit_modules(self):
        """Return modules to shard individually for FSDP2."""
        return list(self.blocks)


def Model(**kw):
    """Factory function for DiT model."""
    return DiT(**kw)
