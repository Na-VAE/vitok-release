"""Pretrained model registry and HuggingFace Hub integration."""

from __future__ import annotations

import os
from typing import Optional

# Registry of pretrained models: name -> (hf_repo, filename, variant)
# The variant must match the checkpoint architecture exactly
PRETRAINED_MODELS = {
    # L models with different channel configs
    "Ld4-L/1x16x16": ("philippe-eecs/vitok", "Ld4-L_x16x16_test/model.safetensors", "Ld4-Ld24/1x16x16"),
    "Ld4-L/1x16x32": ("philippe-eecs/vitok", "Ld4-L_x16x32_test/model.safetensors", "Ld4-Ld24/1x16x32"),
    "Ld4-L/1x16x64": ("philippe-eecs/vitok", "Ld4-L_x16x64_test/model.safetensors", "Ld4-Ld24/1x16x64"),
    # T models (tiny) - decoder only
    "Td4-T/1x32x64": ("philippe-eecs/vitok", "Td4-T_x32x64/model.safetensors", "Td4-Td40/1x32x64"),
    "Td4-T/1x32x128": ("philippe-eecs/vitok", "Td4-T_x32x128/model.safetensors", "Td4-Td40/1x32x128"),
    "Td4-T/1x32x256": ("philippe-eecs/vitok", "Td4-T_x32x256/model.safetensors", "Td4-Td40/1x32x256"),
}

# Aliases for convenience
PRETRAINED_ALIASES = {
    "L-16": "Ld4-L/1x16x16",
    "L-32": "Ld4-L/1x16x32",
    "L-64": "Ld4-L/1x16x64",
    "T-64": "Td4-T/1x32x64",
    "T-128": "Td4-T/1x32x128",
    "T-256": "Td4-T/1x32x256",
}


def list_pretrained() -> list[str]:
    """List all available pretrained model names."""
    return list(PRETRAINED_MODELS.keys()) + list(PRETRAINED_ALIASES.keys())


def get_pretrained_info(name: str) -> tuple[str, str, str]:
    """Get pretrained model info (repo, filename, variant).

    Args:
        name: Model name or alias

    Returns:
        Tuple of (hf_repo, filename, variant)

    Raises:
        ValueError: If model name is not found
    """
    # Resolve alias
    if name in PRETRAINED_ALIASES:
        name = PRETRAINED_ALIASES[name]

    if name not in PRETRAINED_MODELS:
        available = ", ".join(list_pretrained())
        raise ValueError(f"Unknown pretrained model: {name}. Available: {available}")

    return PRETRAINED_MODELS[name]


def download_pretrained(name: str, cache_dir: Optional[str] = None) -> str:
    """Download a pretrained model from HuggingFace Hub.

    Args:
        name: Model name or alias
        cache_dir: Optional cache directory (defaults to HF cache)

    Returns:
        Local path to the downloaded checkpoint
    """
    from huggingface_hub import hf_hub_download

    repo_id, filename, _ = get_pretrained_info(name)

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
    )

    return local_path


def resolve_checkpoint(
    checkpoint: Optional[str],
    pretrained: bool = False,
    cache_dir: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve checkpoint path and variant.

    Args:
        checkpoint: Checkpoint path, pretrained name, or None
        pretrained: If True, treat checkpoint as pretrained model name
        cache_dir: Cache directory for downloads

    Returns:
        Tuple of (resolved_path, variant_override)
        - resolved_path: Local path to checkpoint or None
        - variant_override: Variant string if pretrained, else None
    """
    if checkpoint is None:
        return None, None

    # Check if it's a pretrained model name
    is_pretrained_name = (
        checkpoint in PRETRAINED_MODELS or
        checkpoint in PRETRAINED_ALIASES
    )

    if pretrained or is_pretrained_name:
        # Download from HuggingFace Hub
        _, _, variant = get_pretrained_info(checkpoint)
        local_path = download_pretrained(checkpoint, cache_dir=cache_dir)
        return local_path, variant

    # Local path - no variant override
    return checkpoint, None
