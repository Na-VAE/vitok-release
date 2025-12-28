"""Distributions for variational autoencoders."""

import torch
import numpy as np


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution for VAE latent space."""

    def __init__(self, parameters, deterministic=False, dim=2):
        self.parameters = parameters
        if deterministic:
            self.mean = parameters
            self.logvar = torch.zeros_like(parameters)
        else:
            full = parameters.shape[dim]
            if full % 2 != 0:
                raise ValueError(f"Expected even-sized last dim for stochastic posterior, got {full}")
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)

        if dim == 1:
            self.dims = [1, 2, 3]
        elif dim == 2:
            self.dims = [2]
        else:
            raise NotImplementedError

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        """Sample from the distribution."""
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl_per_token(self):
        """KL divergence per token against unit Gaussian."""
        if self.deterministic:
            return torch.zeros_like(self.mean).sum() * 0
        else:
            return 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=self.dims)

    def nll(self, sample):
        """Negative log-likelihood of sample."""
        if self.deterministic:
            return torch.zeros_like(self.mean).sum() * 0
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.mean(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=self.dims,
        )

    def mode(self):
        """Return the mode (mean) of the distribution."""
        return self.mean
