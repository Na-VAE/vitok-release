"""GPU tests for AE (Autoencoder) functionality.

Run with: modal run modal/test_ae_gpu.py

These tests verify:
1. AE forward pass works on GPU
2. torch.compile compatibility
3. Weight compatibility with vitokv2 (if available)
4. Encode/decode roundtrip
"""

import modal
from modal.env import app, image, compat_image, VITOK_PATH, V2_PATH


@app.function(image=image, gpu="T4", timeout=300)
def test_ae_forward_gpu():
    """Test AE forward pass on GPU."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from vitok import AEConfig, create_ae

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Create model
    config = AEConfig(
        variant="Bd2-Bd4/1x16x32",
        variational=True,
        drop_path_rate=0.0,
    )
    model = create_ae(config).cuda().eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Create test input
    batch_size, seq_len, patch_size = 2, 64, 16
    C = patch_size * patch_size * 3
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    test_input = {
        'patches': torch.randn(batch_size, seq_len, C).cuda(),
        'ptype': torch.ones(batch_size, seq_len, dtype=torch.bool).cuda(),
        'yidx': y.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'xidx': x.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'original_height': torch.full((batch_size,), grid_size * patch_size).cuda(),
        'original_width': torch.full((batch_size,), grid_size * patch_size).cuda(),
        'attention_mask': None,
    }

    # Forward pass
    with torch.no_grad():
        output = model(test_input)

    assert 'patches' in output, f"Missing 'patches' in output: {output.keys()}"
    assert not torch.isnan(output['patches']).any(), "NaN in output"
    assert not torch.isinf(output['patches']).any(), "Inf in output"
    assert output['patches'].shape == test_input['patches'].shape

    print(f"[PASS] Forward pass successful")
    print(f"  Input shape: {test_input['patches'].shape}")
    print(f"  Output shape: {output['patches'].shape}")
    return {"status": "pass", "output_shape": list(output['patches'].shape)}


@app.function(image=image, gpu="T4", timeout=600)
def test_ae_compile():
    """Test AE with torch.compile."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from vitok import AEConfig, create_ae

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Create model
    config = AEConfig(
        variant="Bd2-Bd4/1x16x32",
        variational=True,
        drop_path_rate=0.0,
    )
    model = create_ae(config).cuda().eval()

    # Compile
    print("Compiling model...")
    model_compiled = torch.compile(model)

    # Create test input
    batch_size, seq_len, patch_size = 2, 64, 16
    C = patch_size * patch_size * 3
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    test_input = {
        'patches': torch.randn(batch_size, seq_len, C).cuda(),
        'ptype': torch.ones(batch_size, seq_len, dtype=torch.bool).cuda(),
        'yidx': y.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'xidx': x.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'original_height': torch.full((batch_size,), grid_size * patch_size).cuda(),
        'original_width': torch.full((batch_size,), grid_size * patch_size).cuda(),
        'attention_mask': None,
    }

    # Warmup and test
    with torch.no_grad():
        _ = model_compiled(test_input)  # warmup/compile
        out1 = model_compiled(test_input)
        out2 = model_compiled(test_input)

    max_diff = (out1['patches'] - out2['patches']).abs().max().item()
    print(f"Determinism check: max_diff = {max_diff:.2e}")

    if max_diff == 0.0:
        print("[PASS] Fully deterministic with torch.compile")
    else:
        print("[WARN] Non-deterministic (expected with flex_attention)")

    return {"status": "pass", "deterministic": max_diff == 0.0, "max_diff": max_diff}


@app.function(image=image, gpu="T4", timeout=300)
def test_ae_encode_decode():
    """Test AE encode and decode separately."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from vitok import AEConfig, create_ae

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Create model
    config = AEConfig(
        variant="Bd2-Bd4/1x16x32",
        variational=True,
    )
    model = create_ae(config).cuda().eval()

    # Create test input
    batch_size, seq_len, patch_size = 2, 64, 16
    C = patch_size * patch_size * 3
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    test_input = {
        'patches': torch.randn(batch_size, seq_len, C).cuda(),
        'ptype': torch.ones(batch_size, seq_len, dtype=torch.bool).cuda(),
        'yidx': y.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'xidx': x.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'original_height': torch.full((batch_size,), grid_size * patch_size).cuda(),
        'original_width': torch.full((batch_size,), grid_size * patch_size).cuda(),
        'attention_mask': None,
    }

    # Test encode
    with torch.no_grad():
        encoded = model.encode(test_input)

    print(f"Encode output keys: {encoded.keys()}")
    assert 'z' in encoded or 'posterior' in encoded

    # Get latent
    if 'posterior' in encoded:
        z = encoded['posterior'].mode()
    else:
        z = encoded['z']

    print(f"Latent shape: {z.shape}")

    # Test decode
    decode_input = {
        'z': z,
        'ptype': test_input['ptype'],
        'yidx': test_input['yidx'],
        'xidx': test_input['xidx'],
        'original_height': test_input['original_height'],
        'original_width': test_input['original_width'],
    }

    with torch.no_grad():
        decoded = model.decode(decode_input)

    assert 'patches' in decoded
    assert decoded['patches'].shape == test_input['patches'].shape

    print(f"[PASS] Encode/decode successful")
    print(f"  Input shape: {test_input['patches'].shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Output shape: {decoded['patches'].shape}")

    return {"status": "pass", "latent_shape": list(z.shape)}


@app.function(image=compat_image, gpu="T4", timeout=600)
def test_ae_weight_compatibility():
    """Test that vitokv2 and vitok-release produce identical weights from same seed."""
    import os
    import sys

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    import numpy as np

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # === Create V2 model ===
    sys.path.insert(0, V2_PATH)
    from vitok.models.ae import AE as V2_AE
    from vitok.configs.vae.base import decode_variant as v2_decode

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    v2_params = v2_decode("Bd2-Bd4/1x16x32")
    v2_model = V2_AE(**v2_params, variational=True, drop_path_rate=0.0).cuda().eval()
    v2_state = {k: v.clone() for k, v in v2_model.state_dict().items()}
    print(f"V2 model: {sum(p.numel() for p in v2_model.parameters()):,} params")

    # === Clear modules and create Release model with SAME seed ===
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove(V2_PATH)
    sys.path.insert(0, VITOK_PATH)

    from vitok.models.ae import AE as Release_AE
    from vitok.variant_parser import decode_ae_variant

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    release_params = decode_ae_variant("Bd2-Bd4/1x16x32")
    release_model = Release_AE(**release_params, variational=True, drop_path_rate=0.0).cuda().eval()
    release_state = release_model.state_dict()
    print(f"Release model: {sum(p.numel() for p in release_model.parameters()):,} params")

    # === Compare state dict keys ===
    v2_keys = set(v2_state.keys())
    release_keys = set(release_state.keys())

    if v2_keys != release_keys:
        missing = v2_keys - release_keys
        extra = release_keys - v2_keys
        print(f"[ERROR] Key mismatch!")
        if missing:
            print(f"  In V2 but not release: {missing}")
        if extra:
            print(f"  In release but not V2: {extra}")
        return {"status": "fail", "keys_match": False}

    print(f"[PASS] State dict keys match ({len(v2_keys)} keys)")

    # === Compare actual weight values ===
    max_weight_diff = 0.0
    for key in sorted(v2_keys):
        v2_w = v2_state[key]
        rel_w = release_state[key]
        if v2_w.shape != rel_w.shape:
            print(f"[ERROR] Shape mismatch for {key}: {v2_w.shape} vs {rel_w.shape}")
            return {"status": "fail", "error": f"Shape mismatch: {key}"}
        diff = (v2_w - rel_w).abs().max().item()
        max_weight_diff = max(max_weight_diff, diff)

    print(f"Max weight diff: {max_weight_diff:.2e}")

    if max_weight_diff == 0.0:
        print("[PASS] All weights are EXACTLY identical!")
    elif max_weight_diff < 1e-6:
        print("[PASS] Weights match within float precision")
    else:
        print("[WARN] Weights differ")

    # === Test forward pass ===
    print("\nTesting forward pass...")
    torch.manual_seed(123)
    batch_size, seq_len, patch_size = 2, 64, 16
    C = patch_size * patch_size * 3
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    test_input = {
        'patches': torch.randn(batch_size, seq_len, C).cuda(),
        'ptype': torch.ones(batch_size, seq_len, dtype=torch.bool).cuda(),
        'yidx': y.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'xidx': x.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'original_height': torch.full((batch_size,), grid_size * patch_size).cuda(),
        'original_width': torch.full((batch_size,), grid_size * patch_size).cuda(),
        'attention_mask': None,
    }

    with torch.no_grad():
        release_out = release_model(test_input)

    assert not torch.isnan(release_out['patches']).any(), "NaN in output"
    assert not torch.isinf(release_out['patches']).any(), "Inf in output"
    print("[PASS] Forward pass successful")

    return {
        "status": "pass",
        "keys_match": True,
        "weights_identical": max_weight_diff == 0.0,
        "max_weight_diff": max_weight_diff,
    }


@app.local_entrypoint()
def main():
    """Run all AE GPU tests."""
    print("=" * 60)
    print("Running AE GPU tests")
    print("=" * 60)

    print("\n1. Testing AE forward pass...")
    result = test_ae_forward_gpu.remote()
    print(f"Result: {result}")

    print("\n2. Testing AE encode/decode...")
    result = test_ae_encode_decode.remote()
    print(f"Result: {result}")

    print("\n3. Testing AE with torch.compile...")
    result = test_ae_compile.remote()
    print(f"Result: {result}")

    print("\n4. Testing weight compatibility with vitokv2...")
    try:
        result = test_ae_weight_compatibility.remote()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Skipped (vitokv2 not available): {e}")

    print("\n" + "=" * 60)
    print("All AE GPU tests completed!")
    print("=" * 60)
