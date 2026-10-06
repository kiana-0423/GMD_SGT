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

    Mathematically this is the NequIP envelope

        f(u) = 1 - (p+1)(p+2)/2 u^p + p(p+2) u^(p+1) - p(p+1)/2 u^(p+2),

    evaluated in the exactly equivalent factorised form

        f(u) = (1 - u)^3 * sum_{k=0}^{p-1} C(k+2, 2) u^k,   u = r / r_c.

    The expanded form cancels catastrophically near the cutoff (in float32 it
    turns negative, e.g. -5.7e-6 at r = 2.99365 for r_c = 3). In the factorised
    form ``1 - u`` is computed as ``(r_c - r) / r_c`` and every factor is
    non-negative for ``r < r_c``, so the envelope stays accurate and
    non-negative while keeping its triple zero (value, first and second
    derivative vanish) at the cutoff.
    """

    def __init__(self, cutoff: float, p: int = 6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        u = r / self.cutoff
        inside = u < 1.0
        one_minus_u = torch.where(inside, (self.cutoff - r) / self.cutoff, torch.zeros_like(r))
        # Horner evaluation of sum_{k<p} (k+1)(k+2)/2 u^k (all coefficients > 0).
        series = torch.zeros_like(r)
        for k in range(self.p - 1, -1, -1):
            series = series * u + 0.5 * (k + 1) * (k + 2)
        env = one_minus_u * one_minus_u * one_minus_u * series
        return torch.where(inside, env, torch.zeros_like(env))


__all__ = ["BesselBasis", "PolynomialCutoff"]
