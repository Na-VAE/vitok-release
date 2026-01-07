"""Verify weight shapes match between checkpoint and model."""

import modal

app = modal.App("vitok-weight-check")

VITOK_PATH = "/root/vitok-release"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "packaging>=21.0",
        "webdataset>=0.2.86",
        "huggingface_hub>=0.23.0,<1.0",
    )
    .add_local_dir("vitok", remote_path=f"{VITOK_PATH}/vitok")
)

downloads_volume = modal.Volume.from_name("vitok-downloads")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/downloads": downloads_volume},
    timeout=300,
)
def check_weights():
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    from safetensors.torch import load_file
    from vitok.ae import AEConfig, create_ae

    # Load checkpoint
    ckpt_path = "/downloads/extracted/dir_for_transfer/Ld4-L_x16x64_test/model.safetensors"
    checkpoint_state = load_file(ckpt_path)
    print(f"Loaded {len(checkpoint_state)} keys from checkpoint")

    # Remap keys
    remapped_state = {}
    for k, v in checkpoint_state.items():
        new_key = k.replace("_orig_mod.", "")
        if new_key.startswith("encoder.") and len(new_key) > 8 and new_key[8].isdigit():
            new_key = "encoder_blocks." + new_key[8:]
        if new_key.startswith("decoder.") and len(new_key) > 8 and new_key[8].isdigit():
            new_key = "decoder_blocks." + new_key[8:]
        remapped_state[new_key] = v

    # Create model
    variant = "Ld4-Ld24/1x16x64"
    config = AEConfig(variant=variant, encoder=True, decoder=True)
    model = create_ae(config)

    # Get model state dict
    model_state = model.state_dict()

    print(f"\nModel has {len(model_state)} keys")
    print(f"Checkpoint (remapped) has {len(remapped_state)} keys")

    # Compare keys
    model_keys = set(model_state.keys())
    ckpt_keys = set(remapped_state.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    common = model_keys & ckpt_keys

    print(f"\nCommon keys: {len(common)}")
    print(f"Missing in checkpoint: {len(missing)}")
    print(f"Unexpected in checkpoint: {len(unexpected)}")

    if missing:
        print(f"\nMissing keys: {sorted(missing)}")
    if unexpected:
        print(f"\nUnexpected keys: {sorted(unexpected)}")

    # Check shape mismatches for common keys
    print("\n" + "="*60)
    print("Checking shape matches for common keys...")
    print("="*60)

    mismatches = []
    for key in sorted(common):
        model_shape = model_state[key].shape
        ckpt_shape = remapped_state[key].shape
        if model_shape != ckpt_shape:
            mismatches.append((key, model_shape, ckpt_shape))
            print(f"MISMATCH: {key}")
            print(f"  Model: {model_shape}")
            print(f"  Ckpt:  {ckpt_shape}")

    if not mismatches:
        print("All common keys have matching shapes!")

    # Print a few sample shapes for verification
    print("\n" + "="*60)
    print("Sample weight shapes:")
    print("="*60)

    sample_keys = [
        "patch_embed.weight",
        "to_code.weight",
        "decoder_embed.weight",
        "to_pixels.weight",
        "encoder_blocks.0.norm1.weight",
        "encoder_blocks.0.attn.qkv_proj.weight",
        "decoder_blocks.0.norm1.weight",
        "decoder_blocks.0.attn.qkv_proj.weight",
    ]

    for key in sample_keys:
        if key in model_state and key in remapped_state:
            print(f"{key}:")
            print(f"  Model: {model_state[key].shape}")
            print(f"  Ckpt:  {remapped_state[key].shape}")
        elif key in model_state:
            print(f"{key}: MISSING in checkpoint")
        elif key in remapped_state:
            print(f"{key}: MISSING in model")

    return len(mismatches), list(missing), list(unexpected)


@app.local_entrypoint()
def main():
    n_mismatches, missing, unexpected = check_weights.remote()
    print(f"\nSummary: {n_mismatches} shape mismatches, {len(missing)} missing, {len(unexpected)} unexpected")
