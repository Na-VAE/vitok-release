"""Tests for DiT (Diffusion Transformer) functionality.

Tests verify:
1. DiT model instantiation
2. Forward pass with various configs
3. Weight compatibility with vitokv2
4. CFG (classifier-free guidance) support
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from vitok.models.dit import DiT, timestep_embedding, decode_variant as decode_dit_variant


class TestDiTInstantiation:
    """Test DiT model instantiation."""

    def test_basic_instantiation(self):
        """Test basic model creation."""
        dit_params = decode_dit_variant("Bd4/256")
        model = DiT(**dit_params, code_width=32, text_dim=1000)

        assert model is not None
        assert isinstance(model, DiT)

    def test_config_options(self):
        """Test various config options."""
        dit_params = decode_dit_variant("Bd4/256")
        model = DiT(
            **dit_params,
            code_width=64,
            text_dim=100,
            use_layer_scale=True,
            layer_scale_init=1e-5,
            class_token=True,
            reg_tokens=4,
        )

        assert model.code_width == 64
        assert model.text_dim == 100
        assert model.cls_token is not None
        assert model.reg_token is not None

    def test_parameter_count(self):
        """Test that model has expected number of parameters."""
        dit_params = decode_dit_variant("Bd4/256")
        model = DiT(**dit_params, code_width=32, text_dim=1000)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"DiT-Sd4/256 params: {n_params:,}")
        assert n_params > 1_000_000  # At least 1M params


class TestDiTForward:
    """Test DiT forward pass."""

    @pytest.fixture
    def model(self):
        dit_params = decode_dit_variant("Bd4/256")
        return DiT(**dit_params, code_width=32, text_dim=1000).eval()

    @pytest.fixture
    def dit_input(self):
        batch_size, seq_len, code_width = 2, 64, 32
        grid_size = int(np.sqrt(seq_len))
        y, x = torch.meshgrid(
            torch.arange(grid_size),
            torch.arange(grid_size),
            indexing='ij'
        )

        return {
            'z': torch.randn(batch_size, seq_len, code_width),
            't': torch.randint(0, 1000, (batch_size,)).float(),
            'context': torch.randint(0, 1000, (batch_size,)),
            'row_idx': y.flatten().unsqueeze(0).expand(batch_size, -1),
            'col_idx': x.flatten().unsqueeze(0).expand(batch_size, -1),
        }

    def test_forward_basic(self, model, dit_input):
        """Test basic forward pass."""
        with torch.no_grad():
            output = model(dit_input)

        assert output.shape == dit_input['z'].shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_without_positions(self, model):
        """Test forward pass without explicit positions (square input)."""
        batch_size, seq_len, code_width = 2, 64, 32

        dit_input = {
            'z': torch.randn(batch_size, seq_len, code_width),
            't': torch.randint(0, 1000, (batch_size,)).float(),
            'context': torch.randint(0, 1000, (batch_size,)),
        }

        with torch.no_grad():
            output = model(dit_input)

        assert output.shape == dit_input['z'].shape

    def test_forward_different_batch_sizes(self, model):
        """Test forward with different batch sizes."""
        for batch_size in [1, 2, 4]:
            seq_len, code_width = 64, 32
            grid_size = int(np.sqrt(seq_len))
            y, x = torch.meshgrid(
                torch.arange(grid_size),
                torch.arange(grid_size),
                indexing='ij'
            )

            dit_input = {
                'z': torch.randn(batch_size, seq_len, code_width),
                't': torch.randint(0, 1000, (batch_size,)).float(),
                'context': torch.randint(0, 1000, (batch_size,)),
                'row_idx': y.flatten().unsqueeze(0).expand(batch_size, -1),
                'col_idx': x.flatten().unsqueeze(0).expand(batch_size, -1),
            }

            with torch.no_grad():
                output = model(dit_input)

            assert output.shape == (batch_size, seq_len, code_width)

    def test_forward_different_seq_lengths(self, model):
        """Test forward with different sequence lengths."""
        for side in [4, 8, 16]:
            batch_size, code_width = 2, 32
            seq_len = side * side

            y, x = torch.meshgrid(
                torch.arange(side),
                torch.arange(side),
                indexing='ij'
            )

            dit_input = {
                'z': torch.randn(batch_size, seq_len, code_width),
                't': torch.randint(0, 1000, (batch_size,)).float(),
                'context': torch.randint(0, 1000, (batch_size,)),
                'row_idx': y.flatten().unsqueeze(0).expand(batch_size, -1),
                'col_idx': x.flatten().unsqueeze(0).expand(batch_size, -1),
            }

            with torch.no_grad():
                output = model(dit_input)

            assert output.shape == (batch_size, seq_len, code_width)


class TestTimestepEmbedding:
    """Test timestep embedding function."""

    def test_embedding_shape(self):
        """Test embedding has correct shape."""
        t = torch.tensor([0, 500, 999], dtype=torch.float32)
        emb = timestep_embedding(t, dim=256)

        assert emb.shape == (3, 256)

    def test_embedding_different_dims(self):
        """Test embedding with different dimensions."""
        t = torch.tensor([500.0])

        for dim in [64, 128, 256, 512]:
            emb = timestep_embedding(t, dim=dim)
            assert emb.shape == (1, dim)

    def test_embedding_values(self):
        """Test embedding values are reasonable."""
        t = torch.tensor([0.0, 500.0, 999.0])
        emb = timestep_embedding(t, dim=256)

        # Embeddings should be different for different timesteps
        assert not torch.allclose(emb[0], emb[1])
        assert not torch.allclose(emb[1], emb[2])

        # Values should be bounded
        assert emb.abs().max() <= 1.0


class TestDiTCFG:
    """Test classifier-free guidance support."""

    @pytest.fixture
    def model(self):
        dit_params = decode_dit_variant("Bd4/256")
        return DiT(**dit_params, code_width=32, text_dim=1000).eval()

    def test_cfg_batch_doubling(self, model):
        """Test CFG with doubled batch."""
        batch_size, seq_len, code_width = 2, 64, 32
        grid_size = int(np.sqrt(seq_len))
        y, x = torch.meshgrid(
            torch.arange(grid_size),
            torch.arange(grid_size),
            indexing='ij'
        )

        z = torch.randn(batch_size, seq_len, code_width)
        t = torch.randint(0, 1000, (batch_size,)).float()
        context = torch.randint(0, 1000, (batch_size,))
        null_context = torch.full((batch_size,), 1000)  # Null class

        # Double batch for CFG
        z_cfg = torch.cat([z, z], dim=0)
        t_cfg = torch.cat([t, t], dim=0)
        context_cfg = torch.cat([context, null_context], dim=0)

        dit_input = {
            'z': z_cfg,
            't': t_cfg,
            'context': context_cfg,
            'row_idx': y.flatten().unsqueeze(0).expand(batch_size * 2, -1),
            'col_idx': x.flatten().unsqueeze(0).expand(batch_size * 2, -1),
        }

        with torch.no_grad():
            output = model(dit_input)

        assert output.shape == (batch_size * 2, seq_len, code_width)

        # Split and apply CFG
        cond, uncond = output.chunk(2, dim=0)
        cfg_scale = 4.0
        guided = uncond + cfg_scale * (cond - uncond)

        assert guided.shape == (batch_size, seq_len, code_width)
        assert not torch.isnan(guided).any()

    def test_conditional_vs_unconditional(self, model):
        """Test that conditional and unconditional outputs differ."""
        batch_size, seq_len, code_width = 1, 64, 32

        dit_input_cond = {
            'z': torch.randn(batch_size, seq_len, code_width),
            't': torch.tensor([500.0]),
            'context': torch.tensor([42]),  # Real class
        }

        dit_input_uncond = {
            'z': dit_input_cond['z'].clone(),
            't': dit_input_cond['t'].clone(),
            'context': torch.tensor([1000]),  # Null class
        }

        with torch.no_grad():
            cond = model(dit_input_cond)
            uncond = model(dit_input_uncond)

        # Outputs should differ
        assert not torch.allclose(cond, uncond)


class TestDiTSpecialTokens:
    """Test class token and register tokens."""

    def test_class_token(self):
        """Test model with class token."""
        dit_params = decode_dit_variant("Bd4/256")
        model = DiT(**dit_params, code_width=32, text_dim=1000, class_token=True).eval()

        assert model.cls_token is not None
        assert model.num_special_tokens == 1

        batch_size, seq_len, code_width = 2, 64, 32
        dit_input = {
            'z': torch.randn(batch_size, seq_len, code_width),
            't': torch.randint(0, 1000, (batch_size,)).float(),
            'context': torch.randint(0, 1000, (batch_size,)),
        }

        with torch.no_grad():
            output = model(dit_input)

        # Output should not include class token
        assert output.shape == (batch_size, seq_len, code_width)

    def test_register_tokens(self):
        """Test model with register tokens."""
        n_reg = 4
        dit_params = decode_dit_variant("Bd4/256")
        model = DiT(**dit_params, code_width=32, text_dim=1000, reg_tokens=n_reg).eval()

        assert model.reg_token is not None
        assert model.num_special_tokens == n_reg

        batch_size, seq_len, code_width = 2, 64, 32
        dit_input = {
            'z': torch.randn(batch_size, seq_len, code_width),
            't': torch.randint(0, 1000, (batch_size,)).float(),
            'context': torch.randint(0, 1000, (batch_size,)),
        }

        with torch.no_grad():
            output = model(dit_input)

        assert output.shape == (batch_size, seq_len, code_width)

    def test_both_special_tokens(self):
        """Test model with both class and register tokens."""
        n_reg = 4
        dit_params = decode_dit_variant("Bd4/256")
        model = DiT(**dit_params, code_width=32, text_dim=1000, class_token=True, reg_tokens=n_reg).eval()

        assert model.cls_token is not None
        assert model.reg_token is not None
        assert model.num_special_tokens == 1 + n_reg

        batch_size, seq_len, code_width = 2, 64, 32
        dit_input = {
            'z': torch.randn(batch_size, seq_len, code_width),
            't': torch.randint(0, 1000, (batch_size,)).float(),
            'context': torch.randint(0, 1000, (batch_size,)),
        }

        with torch.no_grad():
            output = model(dit_input)

        assert output.shape == (batch_size, seq_len, code_width)


def check_vitokv2_available():
    """Check if vitokv2 is available for comparison."""
    vitokv2_path = Path(__file__).parent.parent.parent / "vitokv2"
    return vitokv2_path.exists()


@pytest.mark.skipif(not check_vitokv2_available(), reason="vitokv2 not available")
def test_dit_weight_compatibility():
    """Test that DiT weights match between vitokv2 and vitok-release.

    Note: This test compares state dict keys and shapes, not actual weight values,
    since the models are initialized randomly.
    """
    import tempfile

    vitokv2_path = Path(__file__).parent.parent.parent / "vitokv2"

    # Clear any existing vitok imports
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]

    sys.path.insert(0, str(vitokv2_path))

    try:
        # Import v2
        from vitok.models.dit import DiT as V2_DiT
        from vitok.configs.dit.base import decode_variant as v2_decode
    except ImportError as e:
        pytest.skip(f"Could not import from vitokv2: {e}")

    # Get v2 state dict
    torch.manual_seed(42)
    v2_params = v2_decode("Bd4/256")
    # Ensure float8 is disabled for testing (requires torchao)
    v2_model = V2_DiT(**v2_params, text_dim=1000, code_width=32, float8=False)
    v2_state = v2_model.state_dict()
    del v2_model

    # Clear modules
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove(str(vitokv2_path))

    # Import release
    from vitok.models.dit import DiT as Release_DiT
    from vitok.models.dit import decode_variant as decode_dit_variant

    # Create release model with same seed
    torch.manual_seed(42)
    rel_params = decode_dit_variant("Bd4/256")
    rel_model = Release_DiT(**rel_params, text_dim=1000, code_width=32)

    # Compare keys
    v2_keys = set(v2_state.keys())
    rel_keys = set(rel_model.state_dict().keys())

    missing = v2_keys - rel_keys
    extra = rel_keys - v2_keys

    if missing:
        print(f"In V2 but not release: {missing}")
    if extra:
        print(f"In release but not V2: {extra}")

    assert v2_keys == rel_keys, "State dict keys mismatch"

    # Compare shapes
    rel_state = rel_model.state_dict()
    for key in v2_keys:
        v2_shape = v2_state[key].shape
        rel_shape = rel_state[key].shape
        assert v2_shape == rel_shape, f"Shape mismatch for {key}"

    # Test weight transfer
    from safetensors.torch import save_file, load_file

    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        temp_path = f.name

    try:
        save_file(v2_state, temp_path)
        rel_model.load_state_dict(load_file(temp_path), strict=True)
    finally:
        import os
        os.unlink(temp_path)

    print("DiT weight compatibility verified!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
