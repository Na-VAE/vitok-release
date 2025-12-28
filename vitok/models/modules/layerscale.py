"""LayerScale for training stability."""

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """LayerScale for training stability in deep transformers.

    Applies a learnable diagonal scaling to the residual connection.
    This helps with training stability in very deep networks.

    Args:
        dim: dimension of the input
        init_values: initial value for the scaling parameters
    """

    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma
