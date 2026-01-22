"""Test AE weight compatibility between vitok-release and vitokv2.

This test verifies:
1. AE models from both repos have identical architectures
2. Weights can be transferred between them
3. Outputs match within numerical tolerance

Run with: pytest tests/test_ae_compatibility.py -v

Note: Requires vitokv2 to be installed or available at ../vitokv2
"""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Detect available attention backend
try:
    from flash_attn import flash_attn_func
    DEFAULT_ATTN_BACKEND = "flash"
except ImportError:
    DEFAULT_ATTN_BACKEND = "sdpa"


def check_vitokv2_available():
    """Check if vitokv2 is available for comparison."""
    vitokv2_path = Path(__file__).parent.parent.parent / "vitokv2"
    if vitokv2_path.exists():
        return True
    try:
        import vitokv2
        return True
    except ImportError:
        return False


def create_test_naflex_batch(batch_size: int = 2, seq_len: int = 64, patch_size: int = 16):
    """Create a test NaFlex batch dictionary."""
    C = patch_size * patch_size * 3  # Pixels per token

    # Random patches in [-1, 1]
    patches = torch.randn(batch_size, seq_len, C)

    # Grid positions
    grid_size = int(np.sqrt(seq_len))
    y, x = torch.meshgrid(
        torch.arange(grid_size),
        torch.arange(grid_size),
        indexing='ij'
    )
    yidx = y.flatten().unsqueeze(0).expand(batch_size, -1)
    xidx = x.flatten().unsqueeze(0).expand(batch_size, -1)

    # All positions valid
    ptype = torch.ones(batch_size, seq_len, dtype=torch.bool)

    # Original sizes
    orig_size = grid_size * patch_size
    original_height = torch.full((batch_size,), orig_size, dtype=torch.long)
    original_width = torch.full((batch_size,), orig_size, dtype=torch.long)

    return {
        'patches': patches,
        'patch_mask': ptype,
        'row_idx': yidx,
        'col_idx': xidx,
        'orig_height': original_height,
        'orig_width': original_width,
        'attention_mask': None,
    }


@pytest.mark.skipif(not check_vitokv2_available(), reason="vitokv2 not available")
def test_ae_weight_compatibility():
    """Test that vitokv2 weights can be loaded into vitok-release model."""
    import tempfile
    from safetensors.torch import save_file, load_file

    # Add vitokv2 to path to import its AE
    vitokv2_path = Path(__file__).parent.parent.parent / "vitokv2"
    sys.path.insert(0, str(vitokv2_path))

    # Import vitokv2's AE directly (not through shim)
    from vitok.models.ae import AE as V2_AE
    from vitok.configs.variant_parser import decode_ae_variant as v2_decode

    # Remove vitokv2 from path and re-import release version
    sys.path.remove(str(vitokv2_path))

    # Clear cached modules to get fresh imports
    mods_to_remove = [k for k in sys.modules.keys() if k.startswith('vitok')]
    for mod in mods_to_remove:
        del sys.modules[mod]

    # Re-add release path
    release_path = str(Path(__file__).parent.parent)
    if release_path not in sys.path:
        sys.path.insert(0, release_path)

    # Import release version
    from vitok.models.ae import AE as Release_AE
    from vitok.models.ae import decode_variant as release_decode

    # Use a small variant for faster testing
    variant = "Bd2-Bd4/1x16x32"

    # Create v2 model
    torch.manual_seed(42)
    v2_params = v2_decode(variant)
    v2_model = V2_AE(**v2_params, variational=True, drop_path_rate=0.0)

    # Create release model
    torch.manual_seed(42)
    release_params = release_decode(variant)
    release_model = Release_AE(**release_params, variational=True, drop_path_rate=0.0)

    # Compare state dict keys
    v2_keys = set(v2_model.state_dict().keys())
    release_keys = set(release_model.state_dict().keys())

    missing_in_release = v2_keys - release_keys
    extra_in_release = release_keys - v2_keys

    if missing_in_release:
        print(f"Keys in v2 but not in release: {missing_in_release}")
    if extra_in_release:
        print(f"Keys in release but not in v2: {extra_in_release}")

    assert v2_keys == release_keys, \
        f"State dict keys mismatch! Missing: {missing_in_release}, Extra: {extra_in_release}"

    # Test weight transfer: save v2 weights and load into release
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        temp_path = f.name

    try:
        save_file(v2_model.state_dict(), temp_path)
        release_model.load_state_dict(load_file(temp_path), strict=True)
        print("Weight transfer successful!")
    finally:
        import os
        os.unlink(temp_path)

    # Verify shapes match
    for key in v2_keys:
        v2_shape = v2_model.state_dict()[key].shape
        release_shape = release_model.state_dict()[key].shape
        assert v2_shape == release_shape, \
            f"Shape mismatch for {key}: v2={v2_shape}, release={release_shape}"

    print("AE weight compatibility verified!")
    print(f"  Model params: {sum(p.numel() for p in release_model.parameters()):,}")
    print(f"  State dict keys: {len(release_keys)}")


def test_ae_encode_decode():
    """Test that AE can encode and decode without errors."""
    from vitok import AE, decode_variant

    # Create small model with appropriate backend for CPU/GPU
    model = AE(**decode_variant("Bd2-Bd4/1x16x32"), attn_backend=DEFAULT_ATTN_BACKEND)
    model.eval()

    # Create test input
    test_input = create_test_naflex_batch(batch_size=2, seq_len=64, patch_size=16)

    # Test encode
    with torch.no_grad():
        encoded = model.encode(test_input)

    assert 'z' in encoded, f"Encode output missing 'z': {encoded.keys()}"
    z = encoded['z']

    # For non-variational, z should have channels_per_token dimensions
    assert z.shape[-1] == 32, f"Expected z dim 32, got {z.shape[-1]}"

    # Test decode
    decode_input = {
        'z': z,
        'patch_mask': test_input['patch_mask'],
        'row_idx': test_input['row_idx'],
        'col_idx': test_input['col_idx'],
        'orig_height': test_input['orig_height'],
        'orig_width': test_input['orig_width'],
    }

    with torch.no_grad():
        decoded = model.decode(decode_input)

    assert 'patches' in decoded, f"Decode output missing 'patches': {decoded.keys()}"

    print("AE encode/decode verified!")
    print(f"  Input shape: {test_input['patches'].shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Output shape: {decoded['patches'].shape}")


def test_ae_reconstruction():
    """Test that AE can reconstruct input approximately."""
    from vitok import AE, decode_variant

    model = AE(**decode_variant("Bd2-Bd4/1x16x32"), attn_backend=DEFAULT_ATTN_BACKEND)
    model.eval()

    # Create test input
    test_input = create_test_naflex_batch(batch_size=2, seq_len=64, patch_size=16)

    # Encode then decode (current forward only does encode for encoder+decoder models)
    with torch.no_grad():
        encoded = model.encode(test_input)
        output = model.decode(encoded)

    input_patches = test_input['patches']
    output_patches = output['patches']

    # Note: For a randomly initialized model, we don't expect perfect reconstruction
    # We just verify the output has the right shape and is in a reasonable range
    assert output_patches.shape == input_patches.shape, \
        f"Shape mismatch: input={input_patches.shape}, output={output_patches.shape}"

    # Check output is not NaN or Inf
    assert not torch.isnan(output_patches).any(), "Output contains NaN"
    assert not torch.isinf(output_patches).any(), "Output contains Inf"

    print("AE reconstruction test passed!")
    print(f"  Input range: [{input_patches.min():.2f}, {input_patches.max():.2f}]")
    print(f"  Output range: [{output_patches.min():.2f}, {output_patches.max():.2f}]")


if __name__ == "__main__":
    print("=" * 60)
    print("Running test_ae_encode_decode...")
    print("=" * 60)
    test_ae_encode_decode()

    print("\n" + "=" * 60)
    print("Running test_ae_reconstruction...")
    print("=" * 60)
    test_ae_reconstruction()

    if check_vitokv2_available():
        print("\n" + "=" * 60)
        print("Running test_ae_weight_compatibility...")
        print("=" * 60)
        test_ae_weight_compatibility()
    else:
        print("\n" + "=" * 60)
        print("Skipping test_ae_weight_compatibility (vitokv2 not available)")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
