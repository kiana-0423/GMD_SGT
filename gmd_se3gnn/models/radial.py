"""Radial basis and cutoff layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class BesselBasis(nn.Module):
    """
    Bessel radial basis following the DimeNet / NequIP convention.
    """

    def __init__(self, cutoff: float, n_basis: int = 8):
        super().__init__()
        self.cutoff = cutoff
        self.n_basis = n_basis
        freq = torch.arange(1, n_basis + 1, dtype=torch.float32) * math.pi / cutoff
        self.register_buffer("freq", freq)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r_safe = r.unsqueeze(-1).clamp(min=1e-8)
        return (2.0 / self.cutoff) ** 0.5 * torch.sin(self.freq * r_safe) / r_safe


class PolynomialCutoff(nn.Module):
    """
    Smooth polynomial envelope that goes to zero at the cutoff radius.
    """

    def __init__(self, cutoff: float, p: int = 6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        u = r / self.cutoff
        mask = (u < 1.0).float()
        p = self.p
        env = (
            1.0
            - ((p + 1) * (p + 2) / 2) * u**p
            + p * (p + 2) * u ** (p + 1)
            - (p * (p + 1) / 2) * u ** (p + 2)
        )
        return env * mask


__all__ = ["BesselBasis", "PolynomialCutoff"]
