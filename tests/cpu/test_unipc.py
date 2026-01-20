"""Tests for UniPC scheduler (FlowUniPCMultistepScheduler).

Tests verify:
1. Scheduler can be instantiated with default config
2. set_timesteps works correctly
3. step() produces expected output shapes
4. add_noise works correctly
5. Compatibility with vitokv2 scheduler (if available)
"""

import math
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from vitok.unipc import FlowUniPCMultistepScheduler


class TestUniPCSchedulerBasic:
    """Basic tests for UniPC scheduler."""

    def test_instantiation_default(self):
        """Test scheduler can be instantiated with defaults."""
        scheduler = FlowUniPCMultistepScheduler()
        assert scheduler is not None
        assert scheduler.config.num_train_timesteps == 1000
        assert scheduler.config.solver_order == 2
        assert scheduler.config.prediction_type == "flow_prediction"

    def test_instantiation_custom(self):
        """Test scheduler with custom config."""
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=500,
            solver_order=3,
            shift=2.0,
        )
        assert scheduler.config.num_train_timesteps == 500
        assert scheduler.config.solver_order == 3
        assert scheduler.config.shift == 2.0

    def test_set_timesteps(self):
        """Test set_timesteps configures scheduler correctly."""
        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(num_inference_steps=20)

        assert scheduler.num_inference_steps == 20
        assert len(scheduler.timesteps) == 20
        assert scheduler.sigmas is not None
        assert len(scheduler.sigmas) == 21  # One more for final sigma

    def test_set_timesteps_different_steps(self):
        """Test set_timesteps with various step counts."""
        scheduler = FlowUniPCMultistepScheduler()

        for n_steps in [5, 10, 20, 50]:
            scheduler.set_timesteps(num_inference_steps=n_steps)
            assert scheduler.num_inference_steps == n_steps
            assert len(scheduler.timesteps) == n_steps

    def test_timesteps_decrease(self):
        """Test that timesteps decrease monotonically."""
        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(num_inference_steps=20)

        timesteps = scheduler.timesteps.numpy()
        assert all(timesteps[i] > timesteps[i + 1] for i in range(len(timesteps) - 1))

    def test_sigmas_are_valid(self):
        """Test that sigmas are in valid range."""
        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(num_inference_steps=20)

        sigmas = scheduler.sigmas.numpy()
        # First sigma should be close to 1, last should be close to 0
        assert sigmas[0] > 0.9
        assert sigmas[-1] <= 0.01


class TestUniPCSchedulerStep:
    """Test the step function."""

    def test_step_basic(self):
        """Test basic step functionality."""
        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(num_inference_steps=20)

        # Create test data
        batch_size = 2
        seq_len = 64
        dim = 32
        sample = torch.randn(batch_size, seq_len, dim)
        model_output = torch.randn(batch_size, seq_len, dim)

        # Run first step
        timestep = scheduler.timesteps[0]
        output = scheduler.step(model_output, timestep, sample)

        assert hasattr(output, 'prev_sample')
        assert output.prev_sample.shape == sample.shape
        assert not torch.isnan(output.prev_sample).any()
        assert not torch.isinf(output.prev_sample).any()

    def test_step_all_timesteps(self):
        """Test stepping through all timesteps."""
        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(num_inference_steps=10)

        batch_size, seq_len, dim = 2, 64, 32
        sample = torch.randn(batch_size, seq_len, dim)

        for t in scheduler.timesteps:
            model_output = torch.randn_like(sample)
            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

            assert not torch.isnan(sample).any()
            assert not torch.isinf(sample).any()

    def test_step_return_dict_false(self):
        """Test step with return_dict=False."""
        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(num_inference_steps=10)

        sample = torch.randn(2, 64, 32)
        model_output = torch.randn_like(sample)
        timestep = scheduler.timesteps[0]

        output = scheduler.step(model_output, timestep, sample, return_dict=False)

        assert isinstance(output, tuple)
        assert len(output) == 1
        assert output[0].shape == sample.shape


class TestUniPCSchedulerNoise:
    """Test noise-related functions."""

    def test_add_noise(self):
        """Test add_noise function."""
        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(num_inference_steps=20)

        batch_size = 2
        seq_len = 64
        dim = 32
        original_samples = torch.randn(batch_size, seq_len, dim)
        noise = torch.randn_like(original_samples)
        timesteps = torch.tensor([scheduler.timesteps[5], scheduler.timesteps[10]])

        noisy = scheduler.add_noise(original_samples, noise, timesteps)

        assert noisy.shape == original_samples.shape
        assert not torch.isnan(noisy).any()
        # Noisy samples should be different from original
        assert not torch.allclose(noisy, original_samples)

    def test_add_noise_different_timesteps(self):
        """Test that add_noise produces different results for different timesteps."""
        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(num_inference_steps=20)

        original = torch.randn(1, 64, 32)
        noise = torch.randn_like(original)

        noisy_early = scheduler.add_noise(
            original, noise, torch.tensor([scheduler.timesteps[0]])
        )
        noisy_late = scheduler.add_noise(
            original, noise, torch.tensor([scheduler.timesteps[-1]])
        )

        # Early timesteps should have more noise
        diff_early = (noisy_early - original).abs().mean()
        diff_late = (noisy_late - original).abs().mean()
        assert diff_early > diff_late

    def test_scale_model_input(self):
        """Test scale_model_input is identity."""
        scheduler = FlowUniPCMultistepScheduler()
        sample = torch.randn(2, 64, 32)
        scaled = scheduler.scale_model_input(sample)
        assert torch.allclose(sample, scaled)


class TestUniPCSchedulerShift:
    """Test timestep shifting functionality."""

    def test_shift_affects_sigmas(self):
        """Test that shift parameter affects sigma schedule."""
        scheduler_unshifted = FlowUniPCMultistepScheduler(shift=1.0)
        scheduler_shifted = FlowUniPCMultistepScheduler(shift=3.0)

        scheduler_unshifted.set_timesteps(num_inference_steps=20)
        scheduler_shifted.set_timesteps(num_inference_steps=20)

        # Shifted scheduler should have different sigma values
        assert not torch.allclose(scheduler_unshifted.sigmas, scheduler_shifted.sigmas)

    def test_time_shift_function(self):
        """Test the time_shift function directly."""
        scheduler = FlowUniPCMultistepScheduler(use_dynamic_shifting=True)

        mu = 1.0
        sigma = 1.0
        t = torch.tensor([0.5])

        shifted = scheduler.time_shift(mu, sigma, t)
        assert shifted.shape == t.shape

        # With mu=0, sigma=1, shift should be identity
        t2 = scheduler.time_shift(0.0, 1.0, torch.tensor([0.5]))
        expected = 1.0 / (1.0 + (1.0 / 0.5 - 1.0) ** 1.0)
        assert abs(t2.item() - expected) < 1e-5


class TestUniPCSchedulerDeterminism:
    """Test determinism of scheduler."""

    def test_deterministic_steps(self):
        """Test that same inputs produce same outputs."""
        torch.manual_seed(42)

        scheduler1 = FlowUniPCMultistepScheduler()
        scheduler1.set_timesteps(num_inference_steps=20)

        scheduler2 = FlowUniPCMultistepScheduler()
        scheduler2.set_timesteps(num_inference_steps=20)

        sample = torch.randn(2, 64, 32)
        model_output = torch.randn_like(sample)
        timestep = scheduler1.timesteps[0]

        out1 = scheduler1.step(model_output.clone(), timestep, sample.clone())
        out2 = scheduler2.step(model_output.clone(), timestep, sample.clone())

        assert torch.allclose(out1.prev_sample, out2.prev_sample)

    def test_full_denoising_determinism(self):
        """Test full denoising loop is deterministic."""
        torch.manual_seed(42)

        for run in range(2):
            torch.manual_seed(42)
            scheduler = FlowUniPCMultistepScheduler()
            scheduler.set_timesteps(num_inference_steps=10)

            sample = torch.randn(1, 64, 32)
            initial = sample.clone()

            for t in scheduler.timesteps:
                model_output = torch.randn_like(sample) * 0.1
                output = scheduler.step(model_output, t, sample)
                sample = output.prev_sample

            if run == 0:
                first_result = sample.clone()
            else:
                assert torch.allclose(first_result, sample, atol=1e-6)


class TestUniPCSchedulerConfig:
    """Test scheduler configuration options."""

    def test_solver_types(self):
        """Test different solver types."""
        for solver_type in ["bh1", "bh2"]:
            scheduler = FlowUniPCMultistepScheduler(solver_type=solver_type)
            scheduler.set_timesteps(num_inference_steps=10)

            sample = torch.randn(2, 64, 32)
            model_output = torch.randn_like(sample)

            output = scheduler.step(model_output, scheduler.timesteps[0], sample)
            assert not torch.isnan(output.prev_sample).any()

    def test_invalid_solver_type(self):
        """Test that invalid solver type raises error."""
        with pytest.raises(NotImplementedError):
            FlowUniPCMultistepScheduler(solver_type="invalid")

    def test_lower_order_final(self):
        """Test lower_order_final option."""
        scheduler = FlowUniPCMultistepScheduler(lower_order_final=True)
        scheduler.set_timesteps(num_inference_steps=5)

        sample = torch.randn(2, 64, 32)
        for t in scheduler.timesteps:
            model_output = torch.randn_like(sample)
            output = scheduler.step(model_output, t, sample)
            sample = output.prev_sample

        assert not torch.isnan(sample).any()

    def test_predict_x0_options(self):
        """Test predict_x0 options."""
        for predict_x0 in [True, False]:
            scheduler = FlowUniPCMultistepScheduler(predict_x0=predict_x0)
            scheduler.set_timesteps(num_inference_steps=10)

            sample = torch.randn(2, 64, 32)
            model_output = torch.randn_like(sample)

            output = scheduler.step(model_output, scheduler.timesteps[0], sample)
            assert not torch.isnan(output.prev_sample).any()


def check_vitokv2_available():
    """Check if vitokv2 is available for comparison."""
    vitokv2_path = Path(__file__).parent.parent.parent / "vitokv2"
    return vitokv2_path.exists()


@pytest.mark.skipif(not check_vitokv2_available(), reason="vitokv2 not available")
def test_unipc_vitokv2_compatibility():
    """Test that scheduler matches vitokv2 implementation."""
    vitokv2_path = Path(__file__).parent.parent.parent / "vitokv2"
    sys.path.insert(0, str(vitokv2_path))

    from vitok.unipc import FlowUniPCMultistepScheduler as V2_Scheduler

    # Clear and re-import release version
    for mod in [k for k in list(sys.modules.keys()) if k.startswith('vitok')]:
        del sys.modules[mod]
    sys.path.remove(str(vitokv2_path))

    from vitok.unipc import FlowUniPCMultistepScheduler as Release_Scheduler

    # Create both schedulers with same config
    v2_sched = V2_Scheduler(shift=2.0, solver_order=2)
    rel_sched = Release_Scheduler(shift=2.0, solver_order=2)

    v2_sched.set_timesteps(num_inference_steps=20)
    rel_sched.set_timesteps(num_inference_steps=20)

    # Check timesteps match
    assert torch.allclose(v2_sched.timesteps.float(), rel_sched.timesteps.float())
    assert torch.allclose(v2_sched.sigmas, rel_sched.sigmas)

    # Run through sampling loop with same inputs
    torch.manual_seed(123)
    sample = torch.randn(2, 64, 32)
    v2_sample = sample.clone()
    rel_sample = sample.clone()

    for v2_t, rel_t in zip(v2_sched.timesteps, rel_sched.timesteps):
        model_output = torch.randn_like(sample)

        v2_out = v2_sched.step(model_output.clone(), v2_t, v2_sample)
        rel_out = rel_sched.step(model_output.clone(), rel_t, rel_sample)

        v2_sample = v2_out.prev_sample
        rel_sample = rel_out.prev_sample

    # Final samples should match
    max_diff = (v2_sample - rel_sample).abs().max().item()
    assert max_diff < 1e-5, f"Scheduler outputs differ by {max_diff}"

    print(f"[PASS] UniPC schedulers match (max_diff={max_diff:.2e})")


if __name__ == "__main__":
    print("=" * 60)
    print("Running UniPC scheduler tests")
    print("=" * 60)

    pytest.main([__file__, "-v", "--tb=short"])
