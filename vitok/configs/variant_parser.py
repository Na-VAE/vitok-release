"""Variant string parsers for AE and DiT models."""

import re
from typing import Dict, Any, Optional


# Base presets for model architectures
BASE_WIDTHS = {"B": 768, "L": 1024, "G": 1728, "T": 3072, "E": 4096}
BASE_DEPTHS = {"B": 12, "L": 24, "G": 32, "T": 40, "E": 48}
BASE_HEADS = {"B": 12, "L": 16, "G": 24, "T": 24, "E": 32}
BASE_MLP = {key: 2.67 for key in BASE_WIDTHS}


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
    if variant_name.startswith('w') and '_d' in variant_name and '_h' in variant_name:
        try:
            parts = variant_name.split('_')
            width = int(parts[0][1:])
            depth = int(parts[1][1:])
            heads = int(parts[2][1:])
            mlp_factor = (
                float(parts[3][1:]) if len(parts) > 3 and parts[3].startswith('m')
                else 2.67
            )
            return {"width": width, "depth": depth, "heads": heads, "mlp_factor": mlp_factor}
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid custom variant format: {variant_name}. "
                f"Expected: w{{width}}_d{{depth}}_h{{heads}}[_m{{mlp}}]"
            )

    # Inline modifiers
    width_match = re.search(r'w(\d+)', variant_name)
    depth_match = re.search(r'd(\d+)', variant_name)
    heads_match = re.search(r'h(\d+)', variant_name)
    mlp_match = re.search(r'm(\d+(?:\.\d+)?)', variant_name)

    # Remove modifiers to get base name
    base_name = re.sub(r'w\d+|d\d+|h\d+|m\d+(?:\.\d+)?', '', variant_name)

    if base_name and base_name not in BASE_WIDTHS:
        raise ValueError(f"Unknown base variant: {base_name}. Available: {list(BASE_WIDTHS.keys())}")

    width = int(width_match.group(1)) if width_match else BASE_WIDTHS.get(base_name, 768)
    depth = int(depth_match.group(1)) if depth_match else BASE_DEPTHS.get(base_name, 12)
    heads = int(heads_match.group(1)) if heads_match else BASE_HEADS.get(base_name, 12)
    mlp_factor = float(mlp_match.group(1)) if mlp_match else BASE_MLP.get(base_name, 2.67)

    return {"width": width, "depth": depth, "heads": heads, "mlp_factor": mlp_factor}


def decode_ae_variant(variant: str) -> Dict[str, Any]:
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
    if variant is None:
        return {}

    v, rest = variant.split("/")
    enc_v, dec_v = v.split("-") if "-" in v else (v, v)

    # Parse stride and channel size
    parts = list(map(int, rest.split("x")))
    if len(parts) == 3:
        temporal_stride, spatial_stride, channel_size = parts
    elif len(parts) == 2:
        temporal_stride, spatial_stride, channel_size = 1, parts[0], parts[1]
    else:
        raise ValueError(f"Invalid variant format: {variant}")

    enc_config = _parse_variant_name(enc_v)
    dec_config = _parse_variant_name(dec_v)

    mlp_factor = max(enc_config.get("mlp_factor", 2.67), dec_config.get("mlp_factor", 2.67))

    return {
        "encoder_width": enc_config["width"],
        "decoder_width": dec_config["width"],
        "encoder_depth": enc_config["depth"],
        "decoder_depth": dec_config["depth"],
        "encoder_heads": enc_config["heads"],
        "decoder_heads": dec_config["heads"],
        "mlp_factor": mlp_factor,
        "temporal_stride": temporal_stride,
        "spatial_stride": spatial_stride,
        "channels_per_token": channel_size,
        "pixels_per_token": spatial_stride * spatial_stride * temporal_stride * 3,
    }


def decode_dit_variant(variant: str) -> Dict[str, Any]:
    """Parse DiT variant string like "L/256" or "Gd32/512".

    Format: {model}/{num_tokens}

    Examples:
        - "L/256": Large model, 256 tokens (16x16 grid)
        - "Gd32h20/512": Giant with 32 depth and 20 heads, 512 tokens
        - "w2048_d32_h32/256": Custom config

    Returns:
        Dict with width, depth, num_heads, num_tokens
    """
    if variant is None:
        return {}

    v, num_tokens = variant.split("/")
    num_tokens = int(num_tokens)

    # Custom underscore format
    if v.startswith('w') and '_d' in v and '_h' in v:
        try:
            parts = v.split('_')
            width = int(parts[0][1:])
            depth = int(parts[1][1:])
            heads = int(parts[2][1:])
            return {"width": width, "depth": depth, "num_heads": heads, "num_tokens": num_tokens}
        except (ValueError, IndexError):
            raise ValueError(f"Invalid custom variant: {v}. Expected: w{{width}}_d{{depth}}_h{{heads}}")

    # Inline modifiers
    width_match = re.search(r'w(\d+)', v)
    depth_match = re.search(r'd(\d+)', v)
    heads_match = re.search(r'h(\d+)', v)

    base_name = re.sub(r'w\d+|d\d+|h\d+', '', v)

    if base_name and base_name not in BASE_WIDTHS:
        raise ValueError(f"Unknown variant: {base_name}. Available: {list(BASE_WIDTHS.keys())}")

    width = int(width_match.group(1)) if width_match else BASE_WIDTHS.get(base_name, 1024)
    depth = int(depth_match.group(1)) if depth_match else BASE_DEPTHS.get(base_name, 24)
    heads = int(heads_match.group(1)) if heads_match else BASE_HEADS.get(base_name, 16)

    return {"width": width, "depth": depth, "num_heads": heads, "num_tokens": num_tokens}


def decode_variant(variant: str, model_type: str = "ae") -> Dict[str, Any]:
    """Generic variant decoder.

    Args:
        variant: Variant string
        model_type: "ae" or "dit"

    Returns:
        Config dict for the model
    """
    if model_type == "ae":
        return decode_ae_variant(variant)
    elif model_type == "dit":
        return decode_dit_variant(variant)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
