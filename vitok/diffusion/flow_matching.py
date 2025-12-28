"""Flow matching scheduler for DiT training and sampling."""

import math
from typing import Optional, Tuple

import torch


class FlowMatchingScheduler:
    """Simple flow matching scheduler for training and inference.

    Uses linear interpolation between data and noise:
        x_t = (1 - t) * x_0 + t * noise
        v = dx/dt = noise - x_0

    The model predicts velocity v, and we use it to estimate x_0:
        x_0 = x_t - t * v
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        # Precompute sigma schedule (t values from 0 to 1)
        alphas = torch.linspace(1, 1 / num_train_timesteps, num_train_timesteps)
        sigmas = 1.0 - alphas.flip(0)
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.num_inference_steps: Optional[int] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.inference_sigmas: Optional[torch.Tensor] = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """Set up timesteps for inference."""
        self.num_inference_steps = num_inference_steps

        # Linear spacing from max to min sigma
        sigmas = torch.linspace(
            self.sigmas[-1].item(),
            self.sigmas[0].item(),
            num_inference_steps + 1,
        )[:-1]

        # Apply shift
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        sigmas = torch.cat([sigmas, torch.zeros(1)])

        self.inference_sigmas = sigmas.to(device) if device else sigmas
        self.timesteps = (sigmas[:-1] * self.num_train_timesteps).long()
        if device:
            self.timesteps = self.timesteps.to(device)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples using linear interpolation.

        Args:
            original_samples: Clean samples x_0
            noise: Noise tensor
            timesteps: Discrete timestep indices (0 to num_train_timesteps)

        Returns:
            Noisy samples x_t = (1 - sigma) * x_0 + sigma * noise
        """
        sigmas = self.sigmas.to(
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        # Convert timesteps to sigma values
        sigma = sigmas[timesteps.long().clamp(0, len(sigmas) - 1)]
        while sigma.ndim < original_samples.ndim:
            sigma = sigma.unsqueeze(-1)

        noisy_samples = (1 - sigma) * original_samples + sigma * noise
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep_idx: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one denoising step.

        Args:
            model_output: Model velocity prediction
            timestep_idx: Current timestep index in inference schedule
            sample: Current noisy sample x_t

        Returns:
            Denoised sample x_{t-1}
        """
        if self.inference_sigmas is None:
            raise ValueError("Call set_timesteps() before step()")

        sigma = self.inference_sigmas[timestep_idx]
        sigma_next = self.inference_sigmas[timestep_idx + 1]

        # x_0 estimate: x_0 = x_t - sigma * v
        x0_pred = sample - sigma * model_output

        # Linear interpolation to next timestep
        sample_next = (1 - sigma_next) * x0_pred + sigma_next * model_output + x0_pred
        # Simplified: x_{t-1} = x_t + (sigma_next - sigma) * v
        # But more stable form:
        sample_next = x0_pred + sigma_next * model_output

        return sample_next

    def get_velocity_target(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the flow matching velocity target.

        Args:
            original_samples: Clean samples x_0
            noise: Noise tensor

        Returns:
            Velocity target v = noise - x_0
        """
        return noise - original_samples


def euler_sample(
    model,
    scheduler: FlowMatchingScheduler,
    latents: torch.Tensor,
    context: torch.Tensor,
    num_steps: int = 50,
    cfg_scale: float = 1.0,
    cfg_interval: Tuple[float, float] = (0.0, 1.0),
    device: torch.device = None,
    autocast_ctx=None,
) -> torch.Tensor:
    """Sample from the model using Euler method.

    Args:
        model: DiT model
        scheduler: Flow matching scheduler
        latents: Initial noise latents [B, L, C]
        context: Class labels [B]
        num_steps: Number of denoising steps
        cfg_scale: Classifier-free guidance scale
        cfg_interval: Sigma interval for applying CFG
        device: Target device
        autocast_ctx: Optional autocast context manager

    Returns:
        Denoised latents
    """
    if device is None:
        device = latents.device

    scheduler.set_timesteps(num_steps, device=device)

    # Null context for CFG
    num_classes = model.text_dim if hasattr(model, 'text_dim') else 1000
    context_null = torch.full_like(context, num_classes)

    autocast = autocast_ctx if autocast_ctx else lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)

    for i, t in enumerate(scheduler.timesteps):
        sigma = scheduler.inference_sigmas[i].item()

        # Check if CFG should be applied
        use_cfg = cfg_scale != 1.0 and cfg_interval[0] <= sigma <= cfg_interval[1]

        t_batch = t.expand(latents.shape[0])

        if use_cfg:
            # Batched CFG
            x_in = torch.cat([latents, latents], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)
            y_in = torch.cat([context_null, context], dim=0)

            with autocast():
                out = model({"z": x_in, "t": t_in, "context": y_in})

            uncond, cond = out.chunk(2, dim=0)
            v_pred = uncond + cfg_scale * (cond - uncond)
        else:
            with autocast():
                v_pred = model({"z": latents, "t": t_batch, "context": context})

        # Euler step
        latents = scheduler.step(v_pred.float(), i, latents)

    return latents
