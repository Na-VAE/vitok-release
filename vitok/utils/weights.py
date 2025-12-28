"""Weight loading utilities."""

import os


def load_weights(model, checkpoint_path: str, strict: bool = True):
    """Load model weights from a safetensors checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file or directory
        strict: Whether to require exact match of state dict keys

    Supports:
        - Direct .safetensors file path
        - Directory containing model.safetensors or ema.safetensors
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

    model_state = model.state_dict()
    has_orig_ckpt = any(k.startswith("_orig_mod.") for k in checkpoint_state.keys())
    has_orig_model = any(k.startswith("_orig_mod.") for k in model_state.keys())

    # Handle torch.compile prefixes
    if has_orig_ckpt and not has_orig_model:
        checkpoint_state = {k[len("_orig_mod."):]: v for k, v in checkpoint_state.items()}
    elif has_orig_model and not has_orig_ckpt:
        checkpoint_state = {"_orig_mod." + k: v for k, v in checkpoint_state.items()}

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=strict)

    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    return model
