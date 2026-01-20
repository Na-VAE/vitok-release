"""Pretrained model registry and loading utilities."""

from huggingface_hub import hf_hub_download

# Registry: name -> (repo_id, filenames, variant)
# Format: {size}-f{spatial}x{channels}
_MODELS = {
    # 350M models (51M encoder + 303M decoder), patch size 16
    "350M-f16x16": ("philippehansen/ViTok-v2-350M-f16x16", ["encoder.safetensors", "decoder.safetensors"], "Ld4-Ld24/1x16x16"),
    "350M-f16x32": ("philippehansen/ViTok-v2-350M-f16x32", ["encoder.safetensors", "decoder.safetensors"], "Ld4-Ld24/1x16x32"),
    "350M-f16x64": ("philippehansen/ViTok-v2-350M-f16x64", ["encoder.safetensors", "decoder.safetensors"], "Ld4-Ld24/1x16x64"),
    # 5B models (463M encoder + 4.5B decoder), patch size 16
    "5B-f16x16": ("philippehansen/ViTok-v2-5B-f16x16", ["encoder.safetensors", "decoder.safetensors"], "Td4-T/1x16x16"),
    "5B-f16x32": ("philippehansen/ViTok-v2-5B-f16x32", ["encoder.safetensors", "decoder.safetensors"], "Td4-T/1x16x32"),
    "5B-f16x64": ("philippehansen/ViTok-v2-5B-f16x64", ["encoder.safetensors", "decoder.safetensors"], "Td4-T/1x16x64"),
    # 5B models (463M encoder + 4.5B decoder), patch size 32
    "5B-f32x64": ("philippehansen/ViTok-v2-5B-f32x64", ["encoder.safetensors", "decoder.safetensors"], "Td4-T/1x32x64"),
    "5B-f32x128": ("philippehansen/ViTok-v2-5B-f32x128", ["encoder.safetensors", "decoder.safetensors"], "Td4-T/1x32x128"),
    "5B-f32x256": ("philippehansen/ViTok-v2-5B-f32x256", ["encoder.safetensors", "decoder.safetensors"], "Td4-T/1x32x256"),
}

# Aliases for user-friendly model selection (1:1 mapping to model names)
PRETRAINED_ALIASES = {name: name for name in _MODELS}


def load_pretrained(name: str, component: str | None = None, cache_dir: str | None = None) -> dict:
    """Load pretrained weights from HuggingFace Hub.

    Args:
        name: Model name (e.g., "350M-f16x64", "5B-f32x128")
        component: 'encoder', 'decoder', or None for both
        cache_dir: Optional cache directory (uses HF_HOME by default)

    Returns:
        dict with 'variant' and 'encoder'/'decoder' keys
    """
    if name not in _MODELS:
        raise KeyError(f"Unknown model: {name}. Available: {list(_MODELS.keys())}")

    repo_id, filenames, variant = _MODELS[name]
    result = {'variant': variant}

    from safetensors.torch import load_file

    if component != 'decoder':
        path = hf_hub_download(repo_id=repo_id, filename=filenames[0], cache_dir=cache_dir)
        result['encoder'] = load_file(path)

    if component != 'encoder':
        path = hf_hub_download(repo_id=repo_id, filename=filenames[1], cache_dir=cache_dir)
        result['decoder'] = load_file(path)

    return result


def list_pretrained() -> list[str]:
    """List all available pretrained models."""
    return list(_MODELS.keys())


def get_pretrained_info(name: str) -> tuple[str, list[str], str]:
    """Get model info without downloading.

    Args:
        name: Model name (e.g., "5B-f16x64")

    Returns:
        Tuple of (repo_id, filenames, variant)
    """
    if name not in _MODELS:
        raise KeyError(f"Unknown model: {name}. Available: {list(_MODELS.keys())}")
    return _MODELS[name]


def download_pretrained(name: str, cache_dir: str | None = None) -> list[str]:
    """Download pretrained weights and return file paths.

    Args:
        name: Model name (e.g., "5B-f16x64")
        cache_dir: Optional cache directory (uses HF_HOME by default)

    Returns:
        List of paths to downloaded weight files
    """
    if name not in _MODELS:
        raise KeyError(f"Unknown model: {name}. Available: {list(_MODELS.keys())}")

    repo_id, filenames, _ = _MODELS[name]
    return [hf_hub_download(repo_id=repo_id, filename=f, cache_dir=cache_dir) for f in filenames]


__all__ = ["load_pretrained", "list_pretrained", "get_pretrained_info", "download_pretrained", "PRETRAINED_ALIASES"]
