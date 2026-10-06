"""e3nn-free equivalents of e3nn modules restricted to scalar (``Nx0e``) irreps.

For purely scalar features e3nn's tensor product, linear layer and batch norm
reduce to simple closed forms. These modules reproduce them exactly — same
outputs, same parameter/buffer names and shapes — so a scalar-irreps
``UnifiedEquivariantMLIP`` checkpoint is interchangeable between environments
with and without e3nn.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ScalarTensorProduct(nn.Module):
    """``o3.FullyConnectedTensorProduct(Nx0e, sh, Mx0e, shared_weights=False)``.

    Only the ``0e x 0e -> 0e`` path exists; with component-normalised
    harmonics ``Y_00 = 1``, so the edge harmonics drop out and the result is
    ``einsum(x, W) / sqrt(N)`` with per-edge weights ``W`` of shape ``[N, M]``.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight_numel = dim_in * dim_out
        self.weight = nn.Parameter(torch.zeros(0))  # e3nn: no internal weights
        self.register_buffer("output_mask", torch.ones(dim_out))

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        w = weight.view(-1, self.dim_in, self.dim_out)
        return torch.einsum("eu,euw->ew", x, w) / math.sqrt(self.dim_in)


class ScalarLinear(nn.Module):
    """``o3.Linear(Nx0e, Mx0e)`` (no biases): ``x @ W / sqrt(N)``."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = nn.Parameter(torch.randn(dim_in * dim_out))
        self.bias = nn.Parameter(torch.zeros(0))
        self.register_buffer("output_mask", torch.ones(dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.view(self.dim_in, self.dim_out) / math.sqrt(self.dim_in)


class ScalarBatchNorm(nn.Module):
    """``e3nn.nn.BatchNorm(Nx0e)`` with its default settings.

    Training mode normalises with batch statistics (component normalisation,
    mean reduction) and updates the running averages; eval mode uses them.
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(0)
            centered = x - mean
            var = centered.pow(2).mean(0)
            with torch.no_grad():
                self.running_mean.copy_(
                    (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
                )
                self.running_var.copy_(
                    (1 - self.momentum) * self.running_var + self.momentum * var.detach()
                )
        else:
            centered = x - self.running_mean
            var = self.running_var
        return centered * ((var + self.eps).pow(-0.5) * self.weight) + self.bias


__all__ = ["ScalarBatchNorm", "ScalarLinear", "ScalarTensorProduct"]
