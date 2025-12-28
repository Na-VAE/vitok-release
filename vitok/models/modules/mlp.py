"""MLP layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation with gated linear unit."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        # Round to multiple of 16 for float8-friendly kernels
        hidden_dim = ((hidden_dim + 8) // 16) * 16

        self.fc1 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(p=float(dropout)) if (dropout and dropout > 0.0) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, g = self.fc1(x).chunk(2, dim=-1)
        out = self.fc2(F.silu(g) * v)
        return self.dropout(out)
