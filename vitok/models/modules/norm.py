"""Normalization layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Args:
            x: Shape [B, L, C] or [B, H, L, D]
        """
        x32 = x.float()
        w32 = self.weight.float()
        y = F.rms_norm(x32, (self.dim,), weight=w32, eps=self.eps)
        return y.type_as(x)


class LayerNorm(nn.LayerNorm):
    """LayerNorm with float32 computation for stability."""

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        """
        Args:
            x: Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)
