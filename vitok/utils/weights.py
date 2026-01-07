"""Weight loading utilities."""

import os


def _remap_legacy_keys(state_dict: dict) -> dict:
    """Remap legacy checkpoint keys to current model keys.

    Handles:
    - torch.compile prefix: _orig_mod.X -> X
    - vitokv2 naming: encoder.X -> encoder_blocks.X, decoder.X -> decoder_blocks.X
    """
    remapped = {}
    for k, v in state_dict.items():
        new_key = k

        # Strip torch.compile prefix
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]

        # Remap encoder.X -> encoder_blocks.X (but not encoder_embed, etc.)
        if new_key.startswith("encoder.") and len(new_key) > 8 and new_key[8].isdigit():
            new_key = "encoder_blocks." + new_key[8:]

        # Remap decoder.X -> decoder_blocks.X (but not decoder_embed, etc.)
        if new_key.startswith("decoder.") and len(new_key) > 8 and new_key[8].isdigit():
            new_key = "decoder_blocks." + new_key[8:]

        remapped[new_key] = v

    return remapped


def load_weights(model, checkpoint_path: str, strict: bool = True):
    """Load model weights from a safetensors checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file or directory
        strict: Whether to require exact match of state dict keys

    Supports:
        - Direct .safetensors file path
        - Directory containing model.safetensors or ema.safetensors
        - Legacy vitokv2 checkpoints (automatic key remapping)
    """
    from safetensors.torch import load_file

    if checkpoint_path.endswith('.safetensors'):
        checkpoint_state = load_file(checkpoint_path)
        resolved_path = checkpoint_path
    elif os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        ema_path = os.path.join(checkpoint_path, "ema.safetensors")
        if os.path.exists(model_path):
            safepath = model_path
        elif os.path.exists(ema_path):
            print("Note: loading EMA weights (model.safetensors not found)")
            safepath = ema_path
        else:
            raise ValueError(f"No safetensors under {checkpoint_path}")
        checkpoint_state = load_file(safepath)
        resolved_path = safepath
    else:
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")

    print(f"Loading weights from: {resolved_path}")

    # Remap legacy keys (handles torch.compile prefix and vitokv2 naming)
    checkpoint_state = _remap_legacy_keys(checkpoint_state)

    # Check if model uses torch.compile prefix
    model_state = model.state_dict()
    has_orig_model = any(k.startswith("_orig_mod.") for k in model_state.keys())

    if has_orig_model:
        # Model is compiled, add prefix back
        checkpoint_state = {"_orig_mod." + k: v for k, v in checkpoint_state.items()}

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=strict)

    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    return model
