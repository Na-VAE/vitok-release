"""Simplified public API for ViTok Diffusion Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from safetensors.torch import load_file

from vitok.variant_parser import decode_dit_variant
from vitok.models.dit import DiT as _DiT
from vitok.utils import resolve_dtype


@dataclass(frozen=True)
class DiTConfig:
    """Configuration for Diffusion Transformer.

    Args:
        variant: Model variant string (e.g., "L/256", "Gd32/512")
        code_width: Latent token dimension (from AE encoder)
        num_classes: Number of class labels for conditioning
        checkpoint: Gradient checkpointing frequency (0 = disabled)
        float8: Enable float8 training (requires torchao)
        use_layer_scale: Enable LayerScale
        layer_scale_init: Initial LayerScale value
        sw: Sliding window size (None = full attention)
        class_token: Add learnable class token
        reg_tokens: Number of register tokens
        train_seq_len: Training sequence length for block mask precomputation
    """

    variant: str = "G/256"
    code_width: int = 32
    num_classes: int = 1000
    checkpoint: int = 0
    float8: bool = False
    use_layer_scale: bool = True
    layer_scale_init: float = 1e-4
    sw: Optional[int] = None
    class_token: bool = False
    reg_tokens: int = 0
    train_seq_len: Optional[int] = None


def create_dit(config: DiTConfig, **overrides) -> _DiT:
    """Create a DiT model from config.

    Args:
        config: DiTConfig instance
        **overrides: Additional kwargs to override config values

    Returns:
        DiT model instance
    """
    params = decode_dit_variant(config.variant)
    kwargs = {
        **params,
        "text_dim": config.num_classes,
        "code_width": config.code_width,
        "checkpoint": config.checkpoint,
        "float8": config.float8,
        "use_layer_scale": config.use_layer_scale,
        "layer_scale_init": config.layer_scale_init,
        "sw": config.sw,
        "class_token": config.class_token,
        "reg_tokens": config.reg_tokens,
        "train_seq_len": config.train_seq_len,
        **overrides,
    }
    return _DiT(**kwargs)


def load_dit(
    checkpoint: Optional[str],
    config: DiTConfig,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = "float32",
    strict: bool = True,
    **overrides,
) -> _DiT:
    """Create a DiT and optionally load weights.

    Args:
        checkpoint: Path to checkpoint file (or None to skip loading)
        config: DiTConfig instance
        device: Target device
        dtype: Target dtype ("float32", "bfloat16", "float16")
        strict: Whether to require exact checkpoint match
        **overrides: Additional kwargs to override config values

    Returns:
        DiT model instance (in eval mode if checkpoint loaded)
    """
    model = create_dit(config, **overrides)
    model.to(device=device, dtype=resolve_dtype(dtype))
    if checkpoint:
        model.load_state_dict(load_file(checkpoint), strict=strict)
    model.eval()
    return model


# Re-export for convenience
DiT = _DiT

__all__ = [
    "DiT",
    "DiTConfig",
    "create_dit",
    "load_dit",
]
