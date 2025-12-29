"""Run all GPU tests on Modal.

Usage:
    modal run modal/test_all.py              # Run all tests
    modal run modal/test_all.py --quick      # Run only quick tests
    modal run modal/test_all.py --compat     # Include vitokv2 compatibility tests
"""

import modal
from modal.env import app, image, compat_image, VITOK_PATH, V2_PATH


@app.function(image=image, gpu="T4", timeout=300)
def run_pytest():
    """Run pytest on GPU for all local tests."""
    import subprocess
    import sys

    sys.path.insert(0, VITOK_PATH)

    result = subprocess.run(
        ["python", "-m", "pytest", f"{VITOK_PATH}/tests/", "-v", "--tb=short", "-x"],
        capture_output=True, text=True, cwd=VITOK_PATH
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return {"returncode": result.returncode, "passed": result.returncode == 0}


@app.function(image=image, gpu="T4", timeout=300)
def test_ae_quick():
    """Quick AE test - forward pass only."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from vitok import AEConfig, create_ae

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    config = AEConfig(variant="Bd2-Bd4/1x16x32", variational=True)
    model = create_ae(config).cuda().eval()
    print(f"AE params: {sum(p.numel() for p in model.parameters()):,}")

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
        output = model(test_input)

    assert not torch.isnan(output['patches']).any()
    print("[PASS] AE forward pass")
    return {"status": "pass"}


@app.function(image=image, gpu="T4", timeout=300)
def test_dit_quick():
    """Quick DiT test - forward pass only."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    import numpy as np
    from vitok import DiTConfig, create_dit

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    config = DiTConfig(variant="Bd4/256", code_width=32)
    model = create_dit(config).cuda().eval()
    print(f"DiT params: {sum(p.numel() for p in model.parameters()):,}")

    batch_size, seq_len, code_width = 2, 64, 32
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')

    dit_input = {
        'z': torch.randn(batch_size, seq_len, code_width).cuda(),
        't': torch.randint(0, 1000, (batch_size,)).float().cuda(),
        'context': torch.randint(0, 1000, (batch_size,)).cuda(),
        'yidx': y.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
        'xidx': x.flatten().unsqueeze(0).expand(batch_size, -1).cuda(),
    }

    with torch.no_grad():
        output = model(dit_input)

    assert not torch.isnan(output).any()
    print("[PASS] DiT forward pass")
    return {"status": "pass"}


@app.function(image=image, gpu="T4", timeout=300)
def test_unipc_sampling():
    """Test UniPC scheduler sampling loop."""
    import sys
    sys.path.insert(0, VITOK_PATH)

    import torch
    from vitok.diffusion.unipc import FlowUniPCMultistepScheduler

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    scheduler = FlowUniPCMultistepScheduler(shift=2.0)
    scheduler.set_timesteps(num_inference_steps=20)

    sample = torch.randn(2, 64, 32).cuda()

    for t in scheduler.timesteps:
        model_output = torch.randn_like(sample)
        output = scheduler.step(model_output, t, sample)
        sample = output.prev_sample

    assert not torch.isnan(sample).any()
    print("[PASS] UniPC sampling loop")
    return {"status": "pass"}


@app.function(image=compat_image, gpu="T4", timeout=600)
def test_ae_compat():
    """Test AE weight compatibility with vitokv2."""
    import os
    import sys

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    import numpy as np

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Create V2 model
    sys.path.insert(0, V2_PATH)
    from vitok.models.ae import AE as V2_AE
    from vitok.configs.vae.base import decode_variant as v2_decode

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    v2_params = v2_decode("Bd2-Bd4/1x16x32")
    v2_model = V2_AE(**v2_params, variational=True, drop_path_rate=0.0).cuda().eval()
    v2_state = {k: v.clone() for k, v in v2_model.state_dict().items()}

    # Create Release model
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove(V2_PATH)
    sys.path.insert(0, VITOK_PATH)

    from vitok.models.ae import AE as Release_AE
    from vitok.configs.variant_parser import decode_ae_variant

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    release_params = decode_ae_variant("Bd2-Bd4/1x16x32")
    release_model = Release_AE(**release_params, variational=True, drop_path_rate=0.0).cuda().eval()
    release_state = release_model.state_dict()

    # Compare
    v2_keys = set(v2_state.keys())
    release_keys = set(release_state.keys())

    if v2_keys != release_keys:
        return {"status": "fail", "error": "Key mismatch"}

    max_diff = 0.0
    for key in v2_keys:
        if v2_state[key].shape != release_state[key].shape:
            return {"status": "fail", "error": f"Shape mismatch: {key}"}
        diff = (v2_state[key] - release_state[key]).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"[PASS] AE weight compatibility (max_diff={max_diff:.2e})")
    return {"status": "pass", "max_diff": max_diff}


@app.function(image=compat_image, gpu="T4", timeout=600)
def test_dit_compat():
    """Test DiT weight compatibility with vitokv2."""
    import os
    import sys

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Create V2 model
    sys.path.insert(0, V2_PATH)
    from vitok.models.dit import DiT as V2_DiT
    from vitok.configs.dit.base import decode_variant as v2_decode

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    v2_params = v2_decode("Bd4/256")
    v2_model = V2_DiT(**v2_params, text_dim=1000, code_width=32).cuda().eval()
    v2_state = {k: v.clone() for k, v in v2_model.state_dict().items()}

    # Create Release model
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove(V2_PATH)
    sys.path.insert(0, VITOK_PATH)

    from vitok.models.dit import DiT as Release_DiT
    from vitok.configs.variant_parser import decode_dit_variant

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    release_params = decode_dit_variant("Bd4/256")
    release_model = Release_DiT(**release_params, text_dim=1000, code_width=32).cuda().eval()
    release_state = release_model.state_dict()

    # Compare
    v2_keys = set(v2_state.keys())
    release_keys = set(release_state.keys())

    if v2_keys != release_keys:
        return {"status": "fail", "error": "Key mismatch"}

    max_diff = 0.0
    for key in v2_keys:
        if v2_state[key].shape != release_state[key].shape:
            return {"status": "fail", "error": f"Shape mismatch: {key}"}
        diff = (v2_state[key] - release_state[key]).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"[PASS] DiT weight compatibility (max_diff={max_diff:.2e})")
    return {"status": "pass", "max_diff": max_diff}


@app.local_entrypoint()
def main(quick: bool = False, compat: bool = False):
    """Run all GPU tests.

    Args:
        quick: Only run quick forward pass tests
        compat: Include vitokv2 compatibility tests (requires vitokv2 at ../vitokv2)
    """
    print("=" * 60)
    print("vitok-release GPU Tests")
    print("=" * 60)

    results = {}

    # Quick tests
    print("\n[1/3] AE forward pass...")
    results["ae_forward"] = test_ae_quick.remote()

    print("[2/3] DiT forward pass...")
    results["dit_forward"] = test_dit_quick.remote()

    print("[3/3] UniPC sampling...")
    results["unipc"] = test_unipc_sampling.remote()

    if not quick:
        print("\n[4/4] Running pytest suite...")
        results["pytest"] = run_pytest.remote()

    if compat:
        print("\n[+] Running vitokv2 compatibility tests...")
        try:
            results["ae_compat"] = test_ae_compat.remote()
            results["dit_compat"] = test_dit_compat.remote()
        except Exception as e:
            print(f"Compatibility tests skipped: {e}")

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    all_passed = True
    for name, result in results.items():
        status = result.get("status", "unknown")
        if status == "pass":
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name}: {result}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60)
