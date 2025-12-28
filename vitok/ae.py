"""Simplified public API for ViTok Autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from vitok.configs.variant_parser import decode_ae_variant
from vitok.datasets.io import preprocess_images, postprocess_images
from vitok.models.ae import AE as _AE
from vitok.utils.weights import load_weights


@dataclass(frozen=True)
class AEConfig:
    """Configuration for ViTok Autoencoder.

    Args:
        variant: Model variant string (e.g., "B/1x16x64", "Ld2-Ld22/1x16x64")
        variational: Whether to use variational encoding
        checkpoint: Gradient checkpointing frequency (0 = disabled)
        float8: Enable float8 training (requires torchao)
        use_layer_scale: Enable LayerScale
        layer_scale_init: Initial LayerScale value
        drop_path_rate: Stochastic depth rate for decoder
        sw: Sliding window size (None = full attention)
        class_token: Add learnable class token
        reg_tokens: Number of register tokens
        train_seq_len: Training sequence length for block mask precomputation
    """

    variant: str = "B/1x16x64"
    variational: bool = False
    checkpoint: int = 0
    float8: bool = False
    use_layer_scale: bool = True
    layer_scale_init: float = 1e-4
    drop_path_rate: float = 0.2
    sw: Optional[int] = None
    class_token: bool = False
    reg_tokens: int = 0
    train_seq_len: Optional[int] = None


def create_ae(config: AEConfig, **overrides) -> _AE:
    """Create an AE model from config.

    Args:
        config: AEConfig instance
        **overrides: Additional kwargs to override config values

    Returns:
        AE model instance
    """
    params = decode_ae_variant(config.variant)
    kwargs = {
        **params,
        "variational": config.variational,
        "checkpoint": config.checkpoint,
        "float8": config.float8,
        "use_layer_scale": config.use_layer_scale,
        "layer_scale_init": config.layer_scale_init,
        "drop_path_rate": config.drop_path_rate,
        "sw": config.sw,
        "class_token": config.class_token,
        "reg_tokens": config.reg_tokens,
        "train_seq_len": config.train_seq_len,
        **overrides,
    }
    return _AE(**kwargs)


def load_ae(
    checkpoint: Optional[str],
    config: AEConfig,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = "float32",
    strict: bool = True,
    **overrides,
) -> _AE:
    """Create an AE and optionally load weights.

    Args:
        checkpoint: Path to checkpoint file (or None to skip loading)
        config: AEConfig instance
        device: Target device
        dtype: Target dtype ("float32", "bfloat16", "float16")
        strict: Whether to require exact checkpoint match
        **overrides: Additional kwargs to override config values

    Returns:
        AE model instance (in eval mode if checkpoint loaded)
    """
    model = create_ae(config, **overrides)
    model.to(device=device, dtype=_resolve_dtype(dtype))
    if checkpoint:
        load_weights(model, checkpoint, strict=strict)
    model.eval()
    return model


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


# Re-export for convenience
AE = _AE

__all__ = [
    "AE",
    "AEConfig",
    "create_ae",
    "load_ae",
    "preprocess_images",
    "postprocess_images",
]
