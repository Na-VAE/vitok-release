"""Pretrained model registry and download utilities."""

from typing import Tuple, List
from huggingface_hub import hf_hub_download

# Registry of pretrained models: name -> (repo_id, filenames, variant)
# filenames can be a single file or a list [encoder, decoder] for split weights
# Naming format: ViTok-v2-{size}-f{spatial}x{channel}
PRETRAINED_MODELS = {
    # 350M models (51M encoder + 303M decoder), patch size 16
    "350M-16": ("philippehansen/ViTok-v2-350M-f16x16", ["encoder.safetensors", "decoder.safetensors"], "Ld4-Ld24/1x16x16"),
    "350M-32": ("philippehansen/ViTok-v2-350M-f16x32", ["encoder.safetensors", "decoder.safetensors"], "Ld4-Ld24/1x16x32"),
    "350M-64": ("philippehansen/ViTok-v2-350M-f16x64", ["encoder.safetensors", "decoder.safetensors"], "Ld4-Ld24/1x16x64"),

    # 5B models (463M encoder + 4.5B decoder), patch size 32
    "5B-64": ("philippehansen/ViTok-v2-5B-f32x64", ["encoder.safetensors", "decoder.safetensors"], "Td4-T/1x32x64"),
    "5B-128": ("philippehansen/ViTok-v2-5B-f32x128", ["encoder.safetensors", "decoder.safetensors"], "Td4-T/1x32x128"),
    "5B-256": ("philippehansen/ViTok-v2-5B-f32x256", ["encoder.safetensors", "decoder.safetensors"], "Td4-T/1x32x256"),
}

# Short aliases for convenience (backward compatibility)
PRETRAINED_ALIASES = {
    # Legacy L aliases -> 350M models
    "L-64": "350M-64",
    "L-32": "350M-32",
    "L-16": "350M-16",
}


def resolve_model_name(name: str) -> str:
    """Resolve alias to full model name."""
    return PRETRAINED_ALIASES.get(name, name)


def get_pretrained_info(name: str) -> Tuple[str, str | List[str], str]:
    """Get pretrained model info.

    Args:
        name: Model name or alias (e.g., "L-64" or "Ld4-Ld24/1x16x64")

    Returns:
        Tuple of (repo_id, filenames, variant)
        filenames is either a single string or list of [encoder, decoder] files

    Raises:
        KeyError: If model not found
    """
    full_name = resolve_model_name(name)
    if full_name not in PRETRAINED_MODELS:
        available = list(PRETRAINED_ALIASES.keys()) + list(PRETRAINED_MODELS.keys())
        raise KeyError(f"Unknown model: {name}. Available: {available}")
    return PRETRAINED_MODELS[full_name]


def is_split_weights(name: str) -> bool:
    """Check if model uses split encoder/decoder weights."""
    _, filenames, _ = get_pretrained_info(name)
    return isinstance(filenames, list)


def download_pretrained(name: str, cache_dir: str | None = None) -> list[str]:
    """Download pretrained weights from HuggingFace Hub.

    Args:
        name: Model name or alias
        cache_dir: Optional cache directory (uses HF_HOME by default)

    Returns:
        List of paths to downloaded weight files
    """
    repo_id, filenames, _ = get_pretrained_info(name)

    if isinstance(filenames, list):
        # Split weights: download both encoder and decoder
        return [
            hf_hub_download(repo_id=repo_id, filename=f, cache_dir=cache_dir)
            for f in filenames
        ]
    else:
        # Single combined weights file - wrap in list for consistent API
        return [hf_hub_download(repo_id=repo_id, filename=filenames, cache_dir=cache_dir)]


def list_pretrained() -> list[str]:
    """List all available pretrained models."""
    return list(PRETRAINED_MODELS.keys()) + list(PRETRAINED_ALIASES.keys())


__all__ = [
    "PRETRAINED_MODELS",
    "PRETRAINED_ALIASES",
    "get_pretrained_info",
    "download_pretrained",
    "list_pretrained",
    "resolve_model_name",
    "is_split_weights",
]
