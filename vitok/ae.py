"""Public API for ViTok Autoencoder."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import torch

from vitok.models.ae import AE
from vitok.utils.weights import load_weights


# Base presets for model architectures
_BASE_WIDTHS = {"B": 768, "L": 1024, "G": 1728, "T": 3072, "E": 4096}
_BASE_DEPTHS = {"B": 12, "L": 24, "G": 32, "T": 40, "E": 48}
_BASE_HEADS = {"B": 12, "L": 16, "G": 24, "T": 24, "E": 32}
_BASE_MLP = 2.67


def _parse_variant_name(variant_name: str) -> Dict[str, Any]:
    """Parse a single variant name into config dict.

    Supports:
      - Base names: B, L, G, T, E
      - Inline modifiers: w{width} d{depth} h{heads} m{mlp_factor}
        e.g., "Lw1280", "Ld32h20", "Gm3.28"
      - Underscore format: w{width}_d{depth}_h{heads}[_m{mlp}]
        e.g., "w4096_d40_h32_m2.0"
    """
    # Custom underscore format: w{width}_d{depth}_h{heads}[_m{mlp}]
    if variant_name.startswith("w") and "_d" in variant_name and "_h" in variant_name:
        parts = variant_name.split("_")
        width = int(parts[0][1:])
        depth = int(parts[1][1:])
        heads = int(parts[2][1:])
        mlp_factor = float(parts[3][1:]) if len(parts) > 3 and parts[3].startswith("m") else _BASE_MLP
        return {"width": width, "depth": depth, "heads": heads, "mlp_factor": mlp_factor}

    # Inline modifiers
    width_match = re.search(r"w(\d+)", variant_name)
    depth_match = re.search(r"d(\d+)", variant_name)
    heads_match = re.search(r"h(\d+)", variant_name)
    mlp_match = re.search(r"m(\d+(?:\.\d+)?)", variant_name)

    # Remove modifiers to get base name
    base_name = re.sub(r"w\d+|d\d+|h\d+|m\d+(?:\.\d+)?", "", variant_name)

    if base_name and base_name not in _BASE_WIDTHS:
        raise ValueError(f"Unknown base variant: {base_name}. Available: {list(_BASE_WIDTHS.keys())}")

    width = int(width_match.group(1)) if width_match else _BASE_WIDTHS.get(base_name, 768)
    depth = int(depth_match.group(1)) if depth_match else _BASE_DEPTHS.get(base_name, 12)
    heads = int(heads_match.group(1)) if heads_match else _BASE_HEADS.get(base_name, 12)
    mlp_factor = float(mlp_match.group(1)) if mlp_match else _BASE_MLP

    return {"width": width, "depth": depth, "heads": heads, "mlp_factor": mlp_factor}


def decode_variant(variant: str) -> Dict[str, Any]:
    """Parse AE variant string like "B/1x16x64" or "Ld2-Ld22/1x16x64".

    Format: {encoder}[-{decoder}]/{temporal}x{spatial}x{channels}

    Examples:
        - "B/1x16x64": Base encoder/decoder, stride 16, 64 channels
        - "Ld2-Ld22/1x16x64": 2-layer Large encoder, 22-layer Large decoder
        - "w4096_d40_h32/1x16x64": Custom width/depth/heads

    Returns:
        Dict with encoder_width, decoder_width, encoder_depth, decoder_depth,
        encoder_heads, decoder_heads, mlp_factor, spatial_stride, temporal_stride,
        channels_per_token, pixels_per_token
    """
    v, rest = variant.split("/")
    enc_v, dec_v = v.split("-") if "-" in v else (v, v)

    parts = list(map(int, rest.split("x")))
    if len(parts) == 3:
        temporal_stride, spatial_stride, channel_size = parts
    elif len(parts) == 2:
        temporal_stride, spatial_stride, channel_size = 1, parts[0], parts[1]
    else:
        raise ValueError(f"Invalid variant format: {variant}")

    enc_config = _parse_variant_name(enc_v)
    dec_config = _parse_variant_name(dec_v)

    return {
        "encoder_width": enc_config["width"],
        "decoder_width": dec_config["width"],
        "encoder_depth": enc_config["depth"],
        "decoder_depth": dec_config["depth"],
        "encoder_heads": enc_config["heads"],
        "decoder_heads": dec_config["heads"],
        "mlp_factor": max(enc_config["mlp_factor"], dec_config["mlp_factor"]),
        "temporal_stride": temporal_stride,
        "spatial_stride": spatial_stride,
        "channels_per_token": channel_size,
        "pixels_per_token": spatial_stride * spatial_stride * temporal_stride * 3,
    }


def create_ae(variant: str, **kwargs) -> AE:
    """Create an AE model from variant string.

    Args:
        variant: Model variant string (e.g., "B/1x16x64", "Ld2-Ld22/1x16x64")
        **kwargs: Additional model kwargs (checkpoint, float8, etc.)

    Returns:
        AE model instance
    """
    params = decode_variant(variant)
    return AE(**params, **kwargs)


def load_ae(
    path: str,
    variant: str,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = "float32",
    strict: bool = True,
    **kwargs,
) -> AE:
    """Load an AE model from checkpoint.

    Args:
        path: Path to checkpoint file
        variant: Model variant string
        device: Target device
        dtype: Target dtype ("float32", "bfloat16", "float16")
        strict: Whether to require exact checkpoint match
        **kwargs: Additional model kwargs

    Returns:
        AE model instance in eval mode
    """
    model = create_ae(variant, **kwargs)
    model.to(device=device, dtype=_resolve_dtype(dtype))
    load_weights(model, path, strict=strict)
    model.eval()
    return model


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
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


# Add classmethods to AE for convenience
def _from_variant(cls, variant: str, **kwargs) -> "AE":
    """Create an AE model from variant string.

    Args:
        variant: Model variant string (e.g., "B/1x16x64", "Ld2-Ld22/1x16x64")
        **kwargs: Additional model kwargs

    Returns:
        AE model instance
    """
    return create_ae(variant, **kwargs)


def _load(
    cls,
    path: str,
    variant: str,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = "float32",
    strict: bool = True,
    **kwargs,
) -> "AE":
    """Load an AE model from checkpoint.

    Args:
        path: Path to checkpoint file
        variant: Model variant string
        device: Target device
        dtype: Target dtype
        strict: Whether to require exact checkpoint match
        **kwargs: Additional model kwargs

    Returns:
        AE model instance in eval mode
    """
    return load_ae(path, variant, device, dtype, strict, **kwargs)


# Attach classmethods to AE
AE.from_variant = classmethod(_from_variant)
AE.load = classmethod(_load)


__all__ = ["AE", "create_ae", "load_ae", "decode_variant"]
