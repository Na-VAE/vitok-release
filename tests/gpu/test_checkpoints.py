"""Test loading safetensor checkpoints and running forward pass.

Run with:
    modal run tests/gpu/test_checkpoints.py --diagnose  # Check key mapping
    modal run tests/gpu/test_checkpoints.py --quick     # Test one L model
    modal run tests/gpu/test_checkpoints.py             # Test all models
"""

import modal

# Standalone app for this test
app = modal.App("vitok-safetensor-tests")

VITOK_PATH = "/root/vitok-release"

# Image with vitok code
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "webdataset>=0.2.86",
        "huggingface_hub>=0.23.0,<1.0",
    )
    .add_local_dir("vitok", remote_path=f"{VITOK_PATH}/vitok")
)

# Mount the volume with extracted safetensors
volume = modal.Volume.from_name("vitok-downloads")

# Model configs: (folder_name, variant_string)
L_MODELS = [
    ("Ld4-L_x16x16_test", "Ld4-L/1x16x16"),
    ("Ld4-L_x16x32_test", "Ld4-L/1x16x32"),
    ("Ld4-L_x16x64_test", "Ld4-L/1x16x64"),
]

T_MODELS = [
    ("Td4-T_x32x64", "Td4-T/1x32x64"),
    ("Td4-T_x32x128", "Td4-T/1x32x128"),
    ("Td4-T_x32x256", "Td4-T/1x32x256"),
]


@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=300)
def diagnose_checkpoint(folder_name: str, variant: str):
    """Compare checkpoint keys with model state dict keys."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    from safetensors import safe_open
    from vitok.ae import AEConfig, create_ae

    # Get checkpoint keys
    ckpt_path = f"/data/extracted/dir_for_transfer/{folder_name}/model.safetensors"
    print(f"Checkpoint: {ckpt_path}")

    with safe_open(ckpt_path, framework="pt") as f:
        ckpt_keys = list(f.keys())

    print(f"Checkpoint has {len(ckpt_keys)} keys")
    print(f"Sample keys: {ckpt_keys[:5]}")

    # Get model keys (decoder-only)
    config = AEConfig(variant=variant, encoder=False, decoder=True)
    model = create_ae(config)
    model_keys = list(model.state_dict().keys())

    print(f"\nModel has {len(model_keys)} keys")
    print(f"Sample keys: {model_keys[:5]}")

    # Compare (strip _orig_mod. prefix)
    ckpt_normalized = {k.replace("_orig_mod.", "") for k in ckpt_keys}
    model_keys_set = set(model_keys)

    common = ckpt_normalized & model_keys_set
    only_ckpt = ckpt_normalized - model_keys_set
    only_model = model_keys_set - ckpt_normalized

    print(f"\n=== Key Comparison ===")
    print(f"Common keys: {len(common)}")
    print(f"Only in checkpoint: {len(only_ckpt)}")
    print(f"Only in model: {len(only_model)}")

    if only_ckpt:
        print(f"\nKeys only in checkpoint (first 10):")
        for k in sorted(only_ckpt)[:10]:
            print(f"  {k}")

    if only_model:
        print(f"\nKeys only in model (first 10):")
        for k in sorted(only_model)[:10]:
            print(f"  {k}")

    # Try to detect the mapping pattern
    if only_ckpt and only_model:
        ckpt_sample = sorted(only_ckpt)[0]
        model_sample = sorted(only_model)[0]
        print(f"\n=== Potential Mapping ===")
        print(f"Checkpoint: {ckpt_sample}")
        print(f"Model:      {model_sample}")

    return {
        "checkpoint_keys": len(ckpt_keys),
        "model_keys": len(model_keys),
        "common": len(common),
        "only_checkpoint": len(only_ckpt),
        "only_model": len(only_model),
        "sample_ckpt": sorted(only_ckpt)[:5] if only_ckpt else [],
        "sample_model": sorted(only_model)[:5] if only_model else [],
    }


@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=600)
def test_load_l_model(folder_name: str, variant: str):
    """Test loading L model (~354M params) and forward pass."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from safetensors.torch import load_file
    from vitok.ae import AEConfig, create_ae

    print(f"Testing: {folder_name} with variant {variant}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Load checkpoint
    ckpt_path = f"/data/extracted/dir_for_transfer/{folder_name}/model.safetensors"
    checkpoint_state = load_file(ckpt_path)
    print(f"Loaded checkpoint with {len(checkpoint_state)} keys")

    # Remap keys: decoder.X -> decoder_blocks.X
    remapped_state = {}
    for k, v in checkpoint_state.items():
        new_key = k.replace("_orig_mod.", "")  # Strip torch.compile prefix
        new_key = new_key.replace("decoder.", "decoder_blocks.")  # Remap decoder
        remapped_state[new_key] = v
    print(f"Remapped to {len(remapped_state)} keys")

    # Create decoder-only model
    config = AEConfig(variant=variant, encoder=False, decoder=True)
    model = create_ae(config).cuda().eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Load weights
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # Parse variant to get channels
    parts = variant.split("/")[1].split("x")
    channels = int(parts[-1])
    spatial_stride = int(parts[-2])

    # Create test input for decode
    batch_size, seq_len = 2, 64
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    decode_input = {
        'z': torch.randn(batch_size, seq_len, channels).cuda(),
        'ptype': torch.ones(batch_size, seq_len, dtype=torch.bool).cuda(),
        'yidx': y.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'xidx': x.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'original_height': torch.full((batch_size,), grid_size * spatial_stride).cuda(),
        'original_width': torch.full((batch_size,), grid_size * spatial_stride).cuda(),
    }

    # Forward pass (decode)
    with torch.no_grad():
        output = model.decode(decode_input)

    # Check output
    assert 'patches' in output, f"Missing 'patches' in output: {output.keys()}"
    assert not torch.isnan(output['patches']).any(), "NaN in output"
    assert not torch.isinf(output['patches']).any(), "Inf in output"

    print(f"[PASS] Forward pass successful")
    print(f"  Input z shape: {decode_input['z'].shape}")
    print(f"  Output patches shape: {output['patches'].shape}")

    return {
        "status": "pass",
        "model": folder_name,
        "variant": variant,
        "output_shape": list(output['patches'].shape),
    }


@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume}, timeout=900)
def test_load_t_model(folder_name: str, variant: str):
    """Test loading T model (~5B params) - needs A100 with bfloat16."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from safetensors.torch import load_file
    from vitok.ae import AEConfig, create_ae

    print(f"Testing: {folder_name} with variant {variant}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load checkpoint
    ckpt_path = f"/data/extracted/dir_for_transfer/{folder_name}/model.safetensors"
    checkpoint_state = load_file(ckpt_path)
    print(f"Loaded checkpoint with {len(checkpoint_state)} keys")

    # Remap keys
    remapped_state = {}
    for k, v in checkpoint_state.items():
        new_key = k.replace("_orig_mod.", "")
        new_key = new_key.replace("decoder.", "decoder_blocks.")
        remapped_state[new_key] = v.to(torch.bfloat16)  # Convert to bfloat16
    print(f"Remapped to {len(remapped_state)} keys (bfloat16)")

    # Create decoder-only model in bfloat16
    config = AEConfig(variant=variant, encoder=False, decoder=True)
    model = create_ae(config).cuda().to(torch.bfloat16).eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Load weights
    missing, unexpected = model.load_state_dict(remapped_state, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # Parse variant to get channels
    parts = variant.split("/")[1].split("x")
    channels = int(parts[-1])
    spatial_stride = int(parts[-2])

    # Create smaller test input for big model
    batch_size, seq_len = 1, 16  # Smaller for 5B model
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    decode_input = {
        'z': torch.randn(batch_size, seq_len, channels, dtype=torch.bfloat16).cuda(),
        'ptype': torch.ones(batch_size, seq_len, dtype=torch.bool).cuda(),
        'yidx': y.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'xidx': x.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'original_height': torch.full((batch_size,), grid_size * spatial_stride).cuda(),
        'original_width': torch.full((batch_size,), grid_size * spatial_stride).cuda(),
    }

    # Forward pass (decode)
    with torch.no_grad():
        output = model.decode(decode_input)

    # Check output
    assert 'patches' in output, f"Missing 'patches' in output: {output.keys()}"
    assert not torch.isnan(output['patches']).any(), "NaN in output"
    assert not torch.isinf(output['patches']).any(), "Inf in output"

    print(f"[PASS] Forward pass successful")
    print(f"  Input z shape: {decode_input['z'].shape}")
    print(f"  Output patches shape: {output['patches'].shape}")

    return {
        "status": "pass",
        "model": folder_name,
        "variant": variant,
        "output_shape": list(output['patches'].shape),
    }


@app.local_entrypoint()
def main(diagnose: bool = False, quick: bool = False):
    """Run safetensor load tests.

    Args:
        diagnose: Only run diagnostic to check key mapping
        quick: Only test one L model
    """
    print("=" * 60)
    print("Safetensor Load Tests")
    print("=" * 60)

    if diagnose:
        print("\n=== Running Diagnostic ===")
        folder, variant = L_MODELS[0]
        result = diagnose_checkpoint.remote(folder, variant)
        print(f"\nResult: {result}")
        return

    # Test L models (T4)
    l_models_to_test = L_MODELS[:1] if quick else L_MODELS
    print(f"\n=== Testing {len(l_models_to_test)} L model(s) on T4 ===")

    for folder, variant in l_models_to_test:
        print(f"\nTesting {folder}...")
        try:
            result = test_load_l_model.remote(folder, variant)
            print(f"Result: {result}")
        except Exception as e:
            print(f"[FAIL] {folder}: {e}")

    if quick:
        print("\n[Quick mode] Skipping T models")
        return

    # Test T models (A100)
    print(f"\n=== Testing {len(T_MODELS)} T model(s) on A100 ===")

    for folder, variant in T_MODELS:
        print(f"\nTesting {folder}...")
        try:
            result = test_load_t_model.remote(folder, variant)
            print(f"Result: {result}")
        except Exception as e:
            print(f"[FAIL] {folder}: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
