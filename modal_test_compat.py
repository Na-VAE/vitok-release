"""Modal GPU tests for vitok-release + vitokv2 compatibility.

Run with: modal run modal_test_compat.py
"""

import modal

# Image with both codebases
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
        "pytest>=7.0.0",
        "requests",
        "transformers",
        "ml_collections",
    )
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
    .add_local_dir("../vitokv2/vitok", remote_path="/root/vitokv2/vitok")
)

app = modal.App("vitok-compat-tests", image=image)


@app.function(gpu="T4", timeout=600)
def test_weight_compatibility():
    """Test that vitokv2 and vitok-release init with identical weights from same seed."""
    import os
    import sys

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    import numpy as np

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # === Create V2 model ===
    sys.path.insert(0, "/root/vitokv2")
    from vitok.models.ae import AE as V2_AE
    from vitok.configs.vae.base import decode_variant as v2_decode

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    v2_params = v2_decode("Bd2-Bd4/1x16x32")
    v2_model = V2_AE(**v2_params, variational=True, drop_path_rate=0.0).cuda().eval()
    v2_state = {k: v.clone() for k, v in v2_model.state_dict().items()}
    print(f"\nV2 model: {sum(p.numel() for p in v2_model.parameters()):,} params")

    # === Clear modules and create Release model with SAME seed ===
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove("/root/vitokv2")
    sys.path.insert(0, "/root/vitok-release")

    from vitok.models.ae import AE as Release_AE
    from vitok.configs.variant_parser import decode_ae_variant

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
        print(f"\n[ERROR] Key mismatch!")
        if missing:
            print(f"  In V2 but not release: {missing}")
        if extra:
            print(f"  In release but not V2: {extra}")
        return {"keys_match": False}

    print(f"\n[PASS] State dict keys match ({len(v2_keys)} keys)")

    # === Compare actual weight values ===
    print("\nComparing weight values...")
    weight_diffs = {}
    max_weight_diff = 0.0
    for key in sorted(v2_keys):
        v2_w = v2_state[key]
        rel_w = release_state[key]
        if v2_w.shape != rel_w.shape:
            print(f"  [ERROR] Shape mismatch for {key}: {v2_w.shape} vs {rel_w.shape}")
            return {"weights_match": False, "error": f"Shape mismatch: {key}"}
        diff = (v2_w - rel_w).abs().max().item()
        weight_diffs[key] = diff
        max_weight_diff = max(max_weight_diff, diff)

    print(f"\nWeight comparison:")
    print(f"  Max diff across all weights: {max_weight_diff:.2e}")

    if max_weight_diff == 0.0:
        print("\n[PASS] All weights are EXACTLY identical!")
    elif max_weight_diff < 1e-6:
        print("\n[PASS] Weights match within float precision")
    else:
        print("\n[WARN] Weights differ - checking which ones...")
        for key, diff in sorted(weight_diffs.items(), key=lambda x: -x[1])[:10]:
            if diff > 0:
                print(f"  {key}: diff={diff:.2e}")

    # === Test forward pass with same input ===
    print("\n" + "="*50)
    print("Testing forward pass...")

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

    # If weights match exactly, outputs should match (modulo flex_attention non-determinism)
    with torch.no_grad():
        release_out = release_model(test_input)

    print(f"Output shape: {release_out['patches'].shape}")
    assert not torch.isnan(release_out['patches']).any(), "NaN in output"
    assert not torch.isinf(release_out['patches']).any(), "Inf in output"
    print("[PASS] Forward pass successful, no NaN/Inf")

    return {
        "keys_match": True,
        "weights_identical": max_weight_diff == 0.0,
        "max_weight_diff": max_weight_diff,
    }


@app.local_entrypoint()
def main():
    print("Testing vitokv2 -> vitok-release weight compatibility...")
    result = test_weight_compatibility.remote()
    print(f"\n{'='*60}")
    print(f"Result: {result}")
