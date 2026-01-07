"""Simplified public API for ViTok Autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from vitok.configs.variant_parser import decode_ae_variant
from vitok.models.ae import AE as _AE
from vitok.utils import load_weights, resolve_dtype, resolve_checkpoint, list_pretrained


@dataclass(frozen=True)
class AEConfig:
    """Configuration for ViTok Autoencoder.

    Args:
        variant: Model variant string (e.g., "B/1x16x64", "Ld4-L/1x16x64")
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
        encoder: Include encoder (set False for decoder-only)
        decoder: Include decoder (set False for encoder-only)
    """

    variant: str = "B/1x16x64"
    variational: bool = False
    checkpoint: int = 0
    float8: bool = False
    use_layer_scale: bool = True
    layer_scale_init: float = 1e-4
    drop_path_rate: float = 0.0
    sw: Optional[int] = None
    class_token: bool = False
    reg_tokens: int = 0
    train_seq_len: Optional[int] = None
    encoder: bool = True
    decoder: bool = True


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
        "encoder": config.encoder,
        "decoder": config.decoder,
        **overrides,
    }
    return _AE(**kwargs)


def load_ae(
    name_or_path: str,
    config: Optional[AEConfig] = None,
    *,
    pretrained: bool = False,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = "float32",
    strict: bool = True,
    cache_dir: Optional[str] = None,
    **overrides,
) -> _AE:
    """Load a ViTok Autoencoder.

    Can load from:
    1. Pretrained model name (e.g., "Ld4-L/1x16x64" with pretrained=True)
    2. Local checkpoint path with config
    3. Just create model from config (no checkpoint)

    Examples:
        # Load pretrained model
        ae = load_ae("Ld4-L/1x16x64", pretrained=True)
        ae = load_ae("L-64", pretrained=True)  # alias

        # Load from local checkpoint
        ae = load_ae("path/to/checkpoint", config=AEConfig(variant="Ld4-L/1x16x64"))

        # Create model without weights
        ae = load_ae(None, config=AEConfig(variant="B/1x16x64"))

    Args:
        name_or_path: Pretrained model name, local checkpoint path, or None
        config: AEConfig instance (optional if using pretrained)
        pretrained: If True, download pretrained weights from HuggingFace Hub
        device: Target device
        dtype: Target dtype ("float32", "bfloat16", "float16")
        strict: Whether to require exact checkpoint match
        cache_dir: Cache directory for downloaded models
        **overrides: Additional kwargs to override config values

    Returns:
        AE model instance (in eval mode if weights loaded)
    """
    # Resolve checkpoint path and variant
    checkpoint_path, variant_override = resolve_checkpoint(
        name_or_path,
        pretrained=pretrained,
        cache_dir=cache_dir,
    )

    # Determine config
    if config is None:
        if variant_override is None:
            raise ValueError(
                "Must provide config when not using pretrained model. "
                f"Available pretrained: {list_pretrained()}"
            )
        # Use variant from pretrained registry
        config = AEConfig(variant=variant_override)
    elif variant_override is not None:
        # Override variant from pretrained registry
        config = AEConfig(
            variant=variant_override,
            variational=config.variational,
            checkpoint=config.checkpoint,
            float8=config.float8,
            use_layer_scale=config.use_layer_scale,
            layer_scale_init=config.layer_scale_init,
            drop_path_rate=config.drop_path_rate,
            sw=config.sw,
            class_token=config.class_token,
            reg_tokens=config.reg_tokens,
            train_seq_len=config.train_seq_len,
            encoder=config.encoder,
            decoder=config.decoder,
        )

    # Create model
    model = create_ae(config, **overrides)
    model.to(device=device, dtype=resolve_dtype(dtype))

    # Load weights if checkpoint provided
    if checkpoint_path:
        load_weights(model, checkpoint_path, strict=strict)

    model.eval()
    return model


# Re-export for convenience
AE = _AE

__all__ = [
    "AE",
    "AEConfig",
    "create_ae",
    "load_ae",
    "list_pretrained",
]
