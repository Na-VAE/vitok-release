"""Modal GPU tests for vitok-release.

Run with: modal run modal_test_release.py
"""

import modal

# Image with vitok-release code baked in
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
    )
    .add_local_dir("vitok", remote_path="/root/vitok-release/vitok")
    .add_local_dir("tests", remote_path="/root/vitok-release/tests")
)

app = modal.App("vitok-release-tests", image=image)


@app.function(gpu="T4", timeout=300)
def test_model_determinism():
    """Test that the model produces deterministic outputs with torch.compile."""
    import sys
    sys.path.insert(0, "/root/vitok-release")

    import torch
    import numpy as np
    from vitok.models.ae import AE
    from vitok.configs.variant_parser import decode_ae_variant

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Create model
    torch.manual_seed(42)
    params = decode_ae_variant("Bd2-Bd4/1x16x32")
    model = AE(**params, variational=True, drop_path_rate=0.0).cuda().eval()
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")

    # Compile
    print("Compiling with torch.compile()...")
    model_compiled = torch.compile(model)

    # Create input
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

    # Run twice
    print("Running forward passes...")
    with torch.no_grad():
        _ = model_compiled(test_input)  # warmup
        out1 = model_compiled(test_input)
        out2 = model_compiled(test_input)

    max_diff = (out1['patches'] - out2['patches']).abs().max().item()
    print(f"\nDeterminism: max_diff = {max_diff:.2e}")

    if max_diff == 0.0:
        print("[PASS] Fully deterministic!")
    else:
        print("[WARN] Non-deterministic")

    return {"max_diff": max_diff, "deterministic": max_diff == 0.0}


@app.function(gpu="T4", timeout=300)
def run_pytest():
    """Run pytest on GPU."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "/root/vitok-release/tests/", "-v", "--tb=short"],
        capture_output=True, text=True, cwd="/root/vitok-release"
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode


@app.local_entrypoint()
def main():
    print("Testing model determinism...")
    result = test_model_determinism.remote()
    print(f"Result: {result}")
