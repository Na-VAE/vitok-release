"""UniPC: Unified Predictor-Corrector Sampler for Diffusion Models.

Based on the paper: "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models"
https://arxiv.org/abs/2302.04867

This is a multi-step sampler that achieves high-quality generation with few steps.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


class NoiseScheduleVP:
    """Noise schedule for variance-preserving (VP) diffusion models.

    Supports both discrete-time and continuous-time parameterizations.
    """

    def __init__(
        self,
        schedule: str = "discrete",
        betas: Optional[torch.Tensor] = None,
        alphas_cumprod: Optional[torch.Tensor] = None,
        continuous_beta_0: float = 0.1,
        continuous_beta_1: float = 20.0,
    ):
        """Initialize the noise schedule.

        Args:
            schedule: One of "discrete" (for discrete-time models) or
                     "linear" / "cosine" (for continuous-time).
            betas: Beta schedule for discrete-time models.
            alphas_cumprod: Cumulative product of alphas for discrete-time.
            continuous_beta_0: Starting beta for continuous linear schedule.
            continuous_beta_1: Ending beta for continuous linear schedule.
        """
        self.schedule = schedule

        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            elif alphas_cumprod is not None:
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            else:
                raise ValueError("For discrete schedule, provide betas or alphas_cumprod")

            self.total_N = len(log_alphas)
            self.T = 1.0
            # Convert to lambda = log(alpha/sigma) = log(alpha) - log(sigma) = 2*log_alpha - log(1-alpha^2)
            self.t_array = torch.linspace(0, 1, self.total_N + 1)[1:].reshape(1, -1)
            self.log_alpha_array = log_alphas.reshape(1, -1)
        else:
            # Continuous-time schedule
            self.total_N = 1000
            self.T = 1.0
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def marginal_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Compute log(alpha_t) for given t."""
        if self.schedule == "discrete":
            return self._interpolate_log_alpha(t)
        else:
            # Linear schedule: log(alpha_t) = -0.25 * t^2 * (beta_1 - beta_0) - 0.5 * t * beta_0
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha_t for given t."""
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sigma_t = sqrt(1 - alpha_t^2) for given t."""
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t: torch.Tensor) -> torch.Tensor:
        """Compute lambda_t = log(alpha_t / sigma_t)."""
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb: torch.Tensor) -> torch.Tensor:
        """Compute t from lambda (inverse of marginal_lambda)."""
        if self.schedule == "discrete":
            log_alpha = -0.5 * torch.logaddexp(torch.zeros_like(lamb), -2.0 * lamb)
            return self._inverse_interpolate_log_alpha(log_alpha)
        else:
            # For linear schedule, solve quadratic
            tmp = 2.0 * (self.beta_1 - self.beta_0) * torch.logaddexp(-2.0 * lamb, torch.zeros_like(lamb))
            delta = self.beta_0 ** 2 + tmp
            return 2.0 * tmp / (torch.sqrt(delta) + self.beta_0) / (self.beta_1 - self.beta_0)

    def _interpolate_log_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Interpolate log_alpha for discrete schedule."""
        shape = t.shape
        t = t.reshape(-1, 1)
        # Find indices for interpolation
        idx = torch.searchsorted(self.t_array.flatten(), t.flatten()).clamp(1, self.total_N - 1)
        t0 = self.t_array.flatten()[idx - 1]
        t1 = self.t_array.flatten()[idx]
        log_alpha_0 = self.log_alpha_array.flatten()[idx - 1]
        log_alpha_1 = self.log_alpha_array.flatten()[idx]
        # Linear interpolation
        weight = (t.flatten() - t0) / (t1 - t0 + 1e-8)
        log_alpha = log_alpha_0 + weight * (log_alpha_1 - log_alpha_0)
        return log_alpha.reshape(shape)

    def _inverse_interpolate_log_alpha(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Inverse interpolation: find t given log_alpha."""
        shape = log_alpha.shape
        log_alpha = log_alpha.reshape(-1)
        # Binary search style - find where log_alpha fits
        idx = torch.searchsorted(self.log_alpha_array.flatten().flip(0), log_alpha)
        idx = self.total_N - 1 - idx.clamp(0, self.total_N - 2)
        t0 = self.t_array.flatten()[idx]
        t1 = self.t_array.flatten()[idx + 1]
        log_alpha_0 = self.log_alpha_array.flatten()[idx]
        log_alpha_1 = self.log_alpha_array.flatten()[idx + 1]
        weight = (log_alpha - log_alpha_0) / (log_alpha_1 - log_alpha_0 + 1e-8)
        t = t0 + weight * (t1 - t0)
        return t.reshape(shape)


def model_wrapper(
    model,
    noise_schedule: NoiseScheduleVP,
    model_type: str = "noise",
    guidance_type: str = "uncond",
    guidance_scale: float = 1.0,
    classifier_fn=None,
):
    """Wrap a model to convert between different prediction types.

    Args:
        model: The diffusion model that predicts noise, x0, or v.
        noise_schedule: The noise schedule.
        model_type: One of "noise", "x_start", "v", "score".
        guidance_type: One of "uncond", "classifier", "classifier-free".
        guidance_scale: Scale for classifier or classifier-free guidance.
        classifier_fn: Optional classifier for classifier guidance.

    Returns:
        A wrapped model that outputs noise prediction.
    """

    def get_model_input_time(t_continuous):
        """Convert continuous time to model input format."""
        return t_continuous * 1000.0

    def noise_pred_fn(x, t_continuous, cond=None):
        """Predict noise from the model."""
        t_input = get_model_input_time(t_continuous)

        if cond is None:
            output = model(x, t_input)
        else:
            output = model(x, t_input, cond)

        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t = noise_schedule.marginal_alpha(t_continuous)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = [1] * (x.ndim - 1)
            return (x - alpha_t.view(-1, *dims) * output) / sigma_t.view(-1, *dims)
        elif model_type == "v":
            alpha_t = noise_schedule.marginal_alpha(t_continuous)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = [1] * (x.ndim - 1)
            return alpha_t.view(-1, *dims) * output + sigma_t.view(-1, *dims) * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = [1] * (x.ndim - 1)
            return -sigma_t.view(-1, *dims) * output
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def cond_grad_fn(x, t_continuous, cond):
        """Compute classifier gradient for classifier guidance."""
        with torch.enable_grad():
            x = x.requires_grad_(True)
            log_prob = classifier_fn(x, t_continuous, cond)
            return torch.autograd.grad(log_prob.sum(), x)[0]

    def model_fn(x, t_continuous, cond=None, uncond=None):
        """The wrapped model function."""
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous, cond)

        elif guidance_type == "classifier":
            assert classifier_fn is not None
            noise = noise_pred_fn(x, t_continuous, cond)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = [1] * (x.ndim - 1)
            return noise - guidance_scale * sigma_t.view(-1, *dims) * cond_grad_fn(x, t_continuous, cond)

        elif guidance_type == "classifier-free":
            # Compute unconditional and conditional predictions
            if uncond is None:
                # Model should handle None as unconditional
                noise_uncond = noise_pred_fn(x, t_continuous, None)
            else:
                noise_uncond = noise_pred_fn(x, t_continuous, uncond)
            noise_cond = noise_pred_fn(x, t_continuous, cond)
            return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        else:
            raise ValueError(f"Unknown guidance_type: {guidance_type}")

    return model_fn


class UniPC:
    """UniPC: Unified Predictor-Corrector sampler.

    This sampler combines predictor and corrector steps for high-quality
    fast sampling of diffusion models.
    """

    def __init__(
        self,
        model_fn,
        noise_schedule: NoiseScheduleVP,
        predict_x0: bool = True,
        thresholding: bool = False,
        max_val: float = 1.0,
        variant: str = "bh1",
    ):
        """Initialize UniPC sampler.

        Args:
            model_fn: Wrapped model function that predicts noise.
            noise_schedule: The noise schedule.
            predict_x0: If True, use x0 prediction; otherwise use epsilon prediction.
            thresholding: If True, apply dynamic thresholding.
            max_val: Maximum value for thresholding.
            variant: UniPC variant, one of "bh1", "bh2".
        """
        self.model_fn = model_fn
        self.noise_schedule = noise_schedule
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val
        self.variant = variant

    def dynamic_thresholding(self, x0: torch.Tensor) -> torch.Tensor:
        """Apply dynamic thresholding to x0 prediction."""
        dims = x0.ndim - 1
        p = 0.995
        s = torch.quantile(
            torch.abs(x0).reshape(x0.shape[0], -1),
            p,
            dim=1
        )
        s = torch.maximum(s, self.max_val * torch.ones_like(s))
        s = s.reshape(-1, *([1] * dims))
        return torch.clamp(x0, -s, s) / s * self.max_val

    def model_fn_wrapper(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond=None,
        uncond=None,
    ) -> torch.Tensor:
        """Wrapper that handles model output and optional thresholding."""
        noise = self.model_fn(x, t, cond, uncond)

        if self.predict_x0:
            alpha_t = self.noise_schedule.marginal_alpha(t)
            sigma_t = self.noise_schedule.marginal_std(t)
            dims = [1] * (x.ndim - 1)
            x0 = (x - sigma_t.view(-1, *dims) * noise) / alpha_t.view(-1, *dims)

            if self.thresholding:
                x0 = self.dynamic_thresholding(x0)

            return x0
        else:
            return noise

    def get_time_steps(
        self,
        skip_type: str,
        t_T: float,
        t_0: float,
        N: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate time steps for sampling.

        Args:
            skip_type: Type of time step spacing ("time_uniform", "logSNR", "time_quadratic").
            t_T: Starting time (typically 1.0 or close to it).
            t_0: Ending time (typically small, like 1e-3).
            N: Number of steps.
            device: Target device.

        Returns:
            Tensor of time steps from t_T to t_0.
        """
        if skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1, device=device)

        elif skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T, device=device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0, device=device))
            lambdas = torch.linspace(lambda_T.item(), lambda_0.item(), N + 1, device=device)
            return self.noise_schedule.inverse_lambda(lambdas)

        elif skip_type == "time_quadratic":
            t = torch.linspace(t_T ** 0.5, t_0 ** 0.5, N + 1, device=device) ** 2
            return t

        else:
            raise ValueError(f"Unknown skip_type: {skip_type}")

    def multistep_uni_pc_update(
        self,
        x: torch.Tensor,
        model_prev_list: List[torch.Tensor],
        t_prev_list: List[torch.Tensor],
        t: torch.Tensor,
        order: int,
        cond=None,
        uncond=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one UniPC update step.

        Args:
            x: Current sample.
            model_prev_list: List of previous model outputs.
            t_prev_list: List of previous time steps.
            t: Target time step.
            order: Order of the method.
            cond: Optional conditioning.
            uncond: Optional unconditional conditioning for CFG.

        Returns:
            Tuple of (updated sample, model output at current step).
        """
        ns = self.noise_schedule

        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)

        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)

        h = lambda_t - lambda_prev_0

        device = x.device

        if self.predict_x0:
            # x0-prediction mode
            x_prev = model_prev_list[-1]

            if order == 1:
                # First-order update (Euler)
                phi_1 = torch.expm1(h)
                dims = [1] * (x.ndim - 1)
                x_t = sigma_t.view(-1, *dims) / ns.marginal_std(t_prev_0).view(-1, *dims) * x - \
                      alpha_t.view(-1, *dims) * phi_1 * x_prev
            else:
                # Higher-order update
                phi_1 = torch.expm1(h)

                # Compute coefficients
                lambda_list = [ns.marginal_lambda(t_prev_list[-(i+1)]) for i in range(order)]
                lambda_list.append(lambda_t)

                # Compute B coefficients for polynomial interpolation
                # Using Lagrange interpolation
                rks = []
                for i in range(order):
                    rks.append((lambda_list[i] - lambda_prev_0) / h)

                # Compute R coefficients
                R = []
                for i in range(1, order):
                    r = 1.0
                    for j in range(1, order):
                        if j != i:
                            r = r * (rks[i] - rks[j]) / (i - j) if i != j else r
                    R.append(r)

                b_h = h

                dims = [1] * (x.ndim - 1)

                # Predictor step using interpolation
                x_t = sigma_t.view(-1, *dims) / ns.marginal_std(t_prev_0).view(-1, *dims) * x

                # Add contributions from previous predictions
                for i in range(order):
                    coeff = phi_1 * self._get_bh_coeff(i, order, rks)
                    x_t = x_t - alpha_t.view(-1, *dims) * coeff * model_prev_list[-(i+1)]

            # Corrector step
            if order > 1 and self.variant == "bh2":
                model_t = self.model_fn_wrapper(x_t, t, cond, uncond)

                # Apply correction
                dims = [1] * (x.ndim - 1)
                corr_coeff = phi_1 / (2 * order)
                x_t = x_t - alpha_t.view(-1, *dims) * corr_coeff * (model_t - x_prev)
            else:
                model_t = self.model_fn_wrapper(x_t, t, cond, uncond)

        else:
            # epsilon-prediction mode
            eps_prev = model_prev_list[-1]

            dims = [1] * (x.ndim - 1)

            if order == 1:
                phi_1 = torch.expm1(h)
                x_t = alpha_t.view(-1, *dims) / ns.marginal_alpha(t_prev_0).view(-1, *dims) * x + \
                      sigma_t.view(-1, *dims) * phi_1 * eps_prev
            else:
                phi_1 = torch.expm1(h)
                x_t = alpha_t.view(-1, *dims) / ns.marginal_alpha(t_prev_0).view(-1, *dims) * x + \
                      sigma_t.view(-1, *dims) * phi_1 * eps_prev

            model_t = self.model_fn_wrapper(x_t, t, cond, uncond)

        return x_t, model_t

    def _get_bh_coeff(self, i: int, order: int, rks: List[float]) -> float:
        """Compute B_h coefficient for UniPC."""
        if order == 1:
            return 1.0
        elif order == 2:
            if i == 0:
                return 1.0 + 1.0 / (2 * rks[1])
            else:
                return -1.0 / (2 * rks[1])
        elif order == 3:
            if i == 0:
                return 1.0 + (1.0 / (2 * rks[1])) + (1.0 / (3 * rks[1] * rks[2]))
            elif i == 1:
                return -(1.0 / (2 * rks[1])) * (1.0 + 2.0 / (3 * rks[2]))
            else:
                return 1.0 / (3 * rks[1] * rks[2])
        else:
            # For higher orders, use numerical integration or explicit formulas
            return 1.0 / order if i == 0 else 0.0

    def sample(
        self,
        x: torch.Tensor,
        steps: int = 20,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        order: int = 3,
        skip_type: str = "time_uniform",
        cond=None,
        uncond=None,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Sample from the diffusion model using UniPC.

        Args:
            x: Initial noise sample.
            steps: Number of sampling steps.
            t_start: Starting time (default: schedule maximum).
            t_end: Ending time (default: small epsilon).
            order: Order of the solver (1-3).
            skip_type: Time step spacing type.
            cond: Optional conditioning.
            uncond: Optional unconditional conditioning for CFG.
            return_intermediate: If True, return all intermediate samples.

        Returns:
            Final sample or list of intermediate samples.
        """
        device = x.device

        if t_start is None:
            t_start = self.noise_schedule.T
        if t_end is None:
            t_end = 1.0 / self.noise_schedule.total_N

        timesteps = self.get_time_steps(skip_type, t_start, t_end, steps, device)

        intermediates = []
        model_prev_list = []
        t_prev_list = []

        for i, t in enumerate(timesteps[:-1]):
            t_next = timesteps[i + 1]

            # Compute current order (lower at the beginning)
            current_order = min(order, i + 1)

            if i == 0:
                # First step: compute model output
                model_output = self.model_fn_wrapper(x, t, cond, uncond)
                model_prev_list.append(model_output)
                t_prev_list.append(t)

            # UniPC update
            x, model_output = self.multistep_uni_pc_update(
                x,
                model_prev_list,
                t_prev_list,
                t_next,
                current_order,
                cond,
                uncond,
            )

            # Update history
            model_prev_list.append(model_output)
            t_prev_list.append(t_next)

            # Keep only needed history
            if len(model_prev_list) > order:
                model_prev_list.pop(0)
                t_prev_list.pop(0)

            if return_intermediate:
                intermediates.append(x.clone())

        if return_intermediate:
            return intermediates
        return x


class UniPCScheduler:
    """A simplified scheduler interface for UniPC sampling.

    This provides a more user-friendly interface similar to diffusers schedulers.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        solver_order: int = 3,
        thresholding: bool = False,
        sample_max_value: float = 1.0,
    ):
        """Initialize the UniPC scheduler.

        Args:
            num_train_timesteps: Number of training timesteps.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            beta_schedule: Beta schedule type ("linear", "scaled_linear", "cosine").
            prediction_type: Model prediction type ("epsilon", "sample", "v_prediction").
            solver_order: Order of the UniPC solver (1-3).
            thresholding: Whether to use dynamic thresholding.
            sample_max_value: Maximum value for thresholding.
        """
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.solver_order = solver_order
        self.thresholding = thresholding
        self.sample_max_value = sample_max_value

        # Create beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        elif beta_schedule == "cosine":
            timesteps = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1)
            alphas_cumprod = torch.cos((timesteps / num_train_timesteps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Create noise schedule
        self.noise_schedule = NoiseScheduleVP(
            schedule="discrete",
            alphas_cumprod=self.alphas_cumprod,
        )

        self.num_inference_steps: Optional[int] = None
        self.timesteps: Optional[torch.Tensor] = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """Set up timesteps for inference."""
        self.num_inference_steps = num_inference_steps

        # Create evenly spaced timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps, device=device) * step_ratio
        timesteps = timesteps.flip(0)

        self.timesteps = timesteps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples for training.

        Args:
            original_samples: Clean samples.
            noise: Gaussian noise.
            timesteps: Timestep indices.

        Returns:
            Noisy samples.
        """
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        # Reshape for broadcasting
        while sqrt_alpha_prod.ndim < original_samples.ndim:
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def create_sampler(
        self,
        model,
        guidance_type: str = "uncond",
        guidance_scale: float = 1.0,
    ):
        """Create a UniPC sampler for the given model.

        Args:
            model: The diffusion model.
            guidance_type: One of "uncond", "classifier-free".
            guidance_scale: Scale for classifier-free guidance.

        Returns:
            UniPC sampler instance.
        """
        # Map prediction type to model_type
        model_type_map = {
            "epsilon": "noise",
            "sample": "x_start",
            "v_prediction": "v",
        }
        model_type = model_type_map.get(self.prediction_type, "noise")

        # Create wrapped model
        model_fn = model_wrapper(
            model,
            self.noise_schedule,
            model_type=model_type,
            guidance_type=guidance_type,
            guidance_scale=guidance_scale,
        )

        # Create UniPC sampler
        sampler = UniPC(
            model_fn,
            self.noise_schedule,
            predict_x0=True,
            thresholding=self.thresholding,
            max_val=self.sample_max_value,
        )

        return sampler


def unipc_sample(
    model,
    scheduler: UniPCScheduler,
    latents: torch.Tensor,
    num_steps: int = 20,
    order: int = 3,
    guidance_scale: float = 1.0,
    cond=None,
    uncond=None,
    device: torch.device = None,
) -> torch.Tensor:
    """Convenience function to sample using UniPC.

    Args:
        model: The diffusion model.
        scheduler: UniPC scheduler.
        latents: Initial noise latents.
        num_steps: Number of sampling steps.
        order: Solver order (1-3).
        guidance_scale: Classifier-free guidance scale.
        cond: Optional conditioning.
        uncond: Optional unconditional conditioning.
        device: Target device.

    Returns:
        Denoised latents.
    """
    if device is None:
        device = latents.device

    guidance_type = "classifier-free" if guidance_scale != 1.0 else "uncond"

    sampler = scheduler.create_sampler(
        model,
        guidance_type=guidance_type,
        guidance_scale=guidance_scale,
    )

    return sampler.sample(
        latents,
        steps=num_steps,
        order=order,
        cond=cond,
        uncond=uncond,
    )
