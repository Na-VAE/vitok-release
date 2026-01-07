"""Pretrained model registry and download utilities."""

from typing import Tuple
from huggingface_hub import hf_hub_download

# Registry of pretrained models: name -> (repo_id, filename, variant)
PRETRAINED_MODELS = {
    # Large models (1.1B decoder)
    "Ld4-Ld24/1x16x64": ("Na-VAE/ViTok-L-64", "model.safetensors", "Ld4-Ld24/1x16x64"),
    "Ld4-Ld24/1x32x64": ("Na-VAE/ViTok-L-32", "model.safetensors", "Ld4-Ld24/1x32x64"),
    "Ld4-Ld24/1x16x16": ("Na-VAE/ViTok-L-16", "model.safetensors", "Ld4-Ld24/1x16x16"),

    # Tiny models (for testing)
    "Td2-Td12/1x16x64": ("Na-VAE/ViTok-T-64", "model.safetensors", "Td2-Td12/1x16x64"),
    "Td2-Td12/1x16x128": ("Na-VAE/ViTok-T-128", "model.safetensors", "Td2-Td12/1x16x128"),
    "Td2-Td12/1x16x256": ("Na-VAE/ViTok-T-256", "model.safetensors", "Td2-Td12/1x16x256"),
}

# Short aliases for convenience
PRETRAINED_ALIASES = {
    "L-64": "Ld4-Ld24/1x16x64",
    "L-32": "Ld4-Ld24/1x32x64",
    "L-16": "Ld4-Ld24/1x16x16",
    "T-64": "Td2-Td12/1x16x64",
    "T-128": "Td2-Td12/1x16x128",
    "T-256": "Td2-Td12/1x16x256",
}


def resolve_model_name(name: str) -> str:
    """Resolve alias to full model name."""
    return PRETRAINED_ALIASES.get(name, name)


def get_pretrained_info(name: str) -> Tuple[str, str, str]:
    """Get pretrained model info.

    Args:
        name: Model name or alias (e.g., "L-64" or "Ld4-Ld24/1x16x64")

    Returns:
        Tuple of (repo_id, filename, variant)

    Raises:
        KeyError: If model not found
    """
    full_name = resolve_model_name(name)
    if full_name not in PRETRAINED_MODELS:
        available = list(PRETRAINED_ALIASES.keys()) + list(PRETRAINED_MODELS.keys())
        raise KeyError(f"Unknown model: {name}. Available: {available}")
    return PRETRAINED_MODELS[full_name]


def download_pretrained(name: str, cache_dir: str | None = None) -> str:
    """Download pretrained weights from HuggingFace Hub.

    Args:
        name: Model name or alias
        cache_dir: Optional cache directory (uses HF_HOME by default)

    Returns:
        Path to downloaded weights file
    """
    repo_id, filename, _ = get_pretrained_info(name)
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def list_pretrained() -> list[str]:
    """List all available pretrained models."""
    return list(PRETRAINED_ALIASES.keys()) + [
        k for k in PRETRAINED_MODELS.keys() if k not in PRETRAINED_ALIASES.values()
    ]


__all__ = [
    "PRETRAINED_MODELS",
    "PRETRAINED_ALIASES",
    "get_pretrained_info",
    "download_pretrained",
    "list_pretrained",
    "resolve_model_name",
]
