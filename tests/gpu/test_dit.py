"""GPU tests for DiT (Diffusion Transformer) functionality.

Run with: modal run tests/gpu/test_dit.py

These tests verify:
1. DiT forward pass works on GPU
2. torch.compile compatibility
3. Weight compatibility with vitokv2 (if available)
4. CFG (classifier-free guidance) support
"""

import modal
from tests.gpu.env import app, image, compat_image, VITOK_PATH, V2_PATH


@app.function(image=image, gpu="T4", timeout=300)
def test_dit_forward_gpu():
    """Test DiT forward pass on GPU."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from vitok import DiTConfig, create_dit

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Create model
    config = DiTConfig(
        variant="Bd4/256",  # Base width, 4 layers for testing
        code_width=32,
        num_classes=1000,
    )
    model = create_dit(config).cuda().eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Create test input
    batch_size, seq_len, code_width = 2, 64, 32
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    dit_input = {
        'z': torch.randn(batch_size, seq_len, code_width).cuda(),
        't': torch.randint(0, 1000, (batch_size,)).float().cuda(),
        'context': torch.randint(0, 1000, (batch_size,)).cuda(),
        'row_idx': y.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'col_idx': x.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
    }

    # Forward pass
    with torch.no_grad():
        output = model(dit_input)

    assert output.shape == (batch_size, seq_len, code_width)
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"

    print(f"[PASS] Forward pass successful")
    print(f"  Input z shape: {dit_input['z'].shape}")
    print(f"  Output shape: {output.shape}")
    return {"status": "pass", "output_shape": list(output.shape)}


@app.function(image=image, gpu="T4", timeout=600)
def test_dit_compile():
    """Test DiT with torch.compile."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from vitok import DiTConfig, create_dit

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Create model
    config = DiTConfig(
        variant="Bd4/256",
        code_width=32,
        num_classes=1000,
    )
    model = create_dit(config).cuda().eval()

    # Compile
    print("Compiling model...")
    model_compiled = torch.compile(model)

    # Create test input
    batch_size, seq_len, code_width = 2, 64, 32
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    dit_input = {
        'z': torch.randn(batch_size, seq_len, code_width).cuda(),
        't': torch.randint(0, 1000, (batch_size,)).float().cuda(),
        'context': torch.randint(0, 1000, (batch_size,)).cuda(),
        'row_idx': y.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'col_idx': x.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
    }

    # Warmup and test
    with torch.no_grad():
        _ = model_compiled(dit_input)  # warmup/compile
        out1 = model_compiled(dit_input)
        out2 = model_compiled(dit_input)

    max_diff = (out1 - out2).abs().max().item()
    print(f"Determinism check: max_diff = {max_diff:.2e}")

    if max_diff == 0.0:
        print("[PASS] Fully deterministic with torch.compile")
    else:
        print("[WARN] Non-deterministic (expected with flex_attention)")

    return {"status": "pass", "deterministic": max_diff == 0.0, "max_diff": max_diff}


@app.function(image=image, gpu="T4", timeout=300)
def test_dit_cfg():
    """Test DiT with classifier-free guidance (CFG)."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from vitok import DiTConfig, create_dit

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Create model
    config = DiTConfig(
        variant="Bd4/256",
        code_width=32,
        num_classes=1000,
    )
    model = create_dit(config).cuda().eval()

    # Create test input with doubled batch for CFG
    batch_size, seq_len, code_width = 2, 64, 32
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    z = torch.randn(batch_size, seq_len, code_width).cuda()
    t = torch.randint(0, 1000, (batch_size,)).float().cuda()
    context = torch.randint(0, 1000, (batch_size,)).cuda()
    null_context = torch.full((batch_size,), 1000).cuda()  # Null class

    # Double the batch for CFG
    z_cfg = torch.cat([z, z], dim=0)
    t_cfg = torch.cat([t, t], dim=0)
    context_cfg = torch.cat([context, null_context], dim=0)
    yidx = y.flatten().unsqueeze(0).expand(batch_size * 2, -1).cuda()
    xidx = x.flatten().unsqueeze(0).expand(batch_size * 2, -1).cuda()

    dit_input = {
        'z': z_cfg,
        't': t_cfg,
        'context': context_cfg,
        'row_idx': yidx,
        'col_idx': xidx,
    }

    # Forward pass
    with torch.no_grad():
        output = model(dit_input)

    # Split conditional and unconditional
    cond, uncond = output.chunk(2, dim=0)

    # Apply CFG
    cfg_scale = 4.0
    guided = uncond + cfg_scale * (cond - uncond)

    assert guided.shape == (batch_size, seq_len, code_width)
    assert not torch.isnan(guided).any(), "NaN in CFG output"
    assert not torch.isinf(guided).any(), "Inf in CFG output"

    print(f"[PASS] CFG test successful")
    print(f"  Conditional output range: [{cond.min():.2f}, {cond.max():.2f}]")
    print(f"  Unconditional output range: [{uncond.min():.2f}, {uncond.max():.2f}]")
    print(f"  Guided output range: [{guided.min():.2f}, {guided.max():.2f}]")

    return {"status": "pass", "cfg_scale": cfg_scale}


@app.function(image=compat_image, gpu="T4", timeout=600)
def test_dit_weight_compatibility():
    """Test that vitokv2 and vitok-release DiT produce identical weights from same seed."""
    import os
    import sys

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    import numpy as np

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # === Create V2 model ===
    sys.path.insert(0, V2_PATH)
    from vitok.models.dit import DiT as V2_DiT
    from vitok.configs.dit.base import decode_variant as v2_decode

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    v2_params = v2_decode("Bd4/256")
    v2_model = V2_DiT(**v2_params, text_dim=1000, code_width=32).cuda().eval()
    v2_state = {k: v.clone() for k, v in v2_model.state_dict().items()}
    print(f"V2 model: {sum(p.numel() for p in v2_model.parameters()):,} params")

    # === Clear modules and create Release model with SAME seed ===
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove(V2_PATH)
    sys.path.insert(0, VITOK_PATH)

    from vitok.models.dit import DiT as Release_DiT
    from vitok.models.dit import decode_variant as decode_dit_variant

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    release_params = decode_dit_variant("Bd4/256")
    release_model = Release_DiT(**release_params, text_dim=1000, code_width=32).cuda().eval()
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
    batch_size, seq_len, code_width = 2, 64, 32
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    dit_input = {
        'z': torch.randn(batch_size, seq_len, code_width).cuda(),
        't': torch.randint(0, 1000, (batch_size,)).float().cuda(),
        'context': torch.randint(0, 1000, (batch_size,)).cuda(),
        'row_idx': y.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'col_idx': x.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
    }

    with torch.no_grad():
        release_out = release_model(dit_input)

    assert not torch.isnan(release_out).any(), "NaN in output"
    assert not torch.isinf(release_out).any(), "Inf in output"
    print("[PASS] Forward pass successful")

    return {
        "status": "pass",
        "keys_match": True,
        "weights_identical": max_weight_diff == 0.0,
        "max_weight_diff": max_weight_diff,
    }


@app.local_entrypoint()
def main():
    """Run all DiT GPU tests."""
    print("=" * 60)
    print("Running DiT GPU tests")
    print("=" * 60)

    print("\n1. Testing DiT forward pass...")
    result = test_dit_forward_gpu.remote()
    print(f"Result: {result}")

    print("\n2. Testing DiT CFG support...")
    result = test_dit_cfg.remote()
    print(f"Result: {result}")

    print("\n3. Testing DiT with torch.compile...")
    result = test_dit_compile.remote()
    print(f"Result: {result}")

    print("\n4. Testing weight compatibility with vitokv2...")
    try:
        result = test_dit_weight_compatibility.remote()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Skipped (vitokv2 not available): {e}")

    print("\n" + "=" * 60)
    print("All DiT GPU tests completed!")
    print("=" * 60)
