"""Local SE(3)-equivariant message passing layers."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .dependencies import E3NN_AVAILABLE, o3
from .geometry import scatter_sum
from .scalar_irreps import ScalarLinear, ScalarTensorProduct


def parse_scalar_irreps(irreps: str) -> Optional[int]:
    """Total multiplicity of an irreps string made only of ``0e`` terms.

    Returns ``None`` when any term is not an even scalar. Works without e3nn,
    e.g. ``"16x0e"`` -> 16, ``"8x0e + 0e"`` -> 9, ``"8x0e + 2x1o"`` -> None.
    """
    total = 0
    for term in str(irreps).replace(" ", "").split("+"):
        if not term:
            continue
        mul_str, sep, ir = term.partition("x")
        if not sep:
            mul_str, ir = "1", term
        if ir != "0e" or not mul_str.isdigit():
            return None
        total += int(mul_str)
    return total


def spherical_harmonics_irreps(l_max: int) -> str:
    """Irreps of real spherical harmonics up to ``l_max`` (parity ``(-1)^l``)."""
    if int(l_max) != l_max or l_max < 0:
        raise ValueError(f"l_max must be a non-negative integer, got {l_max!r}")
    return "+".join(f"1x{ell}{'e' if ell % 2 == 0 else 'o'}" for ell in range(int(l_max) + 1))


class SE3EquivariantMessagePassing(nn.Module):
    """
    One round of SE(3)-equivariant message passing.

    Messages are tensor products of sender features with the edge spherical
    harmonics, weighted by a learned radial filter of the distance. Without
    e3nn only scalar (``Nx0e``) irreps are supported; the tensor product then
    reduces to a distance-dependent continuous-filter convolution, computed
    by e3nn-free modules that match e3nn exactly (same parameters/outputs).

    Messages are multiplied by ``edge_weight`` (the smooth cutoff envelope)
    so they vanish continuously as edges leave the neighbor graph.
    """

    def __init__(
        self,
        irreps_in: str,
        irreps_out: str,
        irreps_sh: str,
        n_basis: int = 8,
        hidden_radial: int = 64,
        avg_neighbors: float = 10.0,
    ):
        super().__init__()
        self.avg_neighbors = avg_neighbors
        self._irreps_in = irreps_in
        self._irreps_out = irreps_out
        self._irreps_sh = irreps_sh

        if E3NN_AVAILABLE:
            irr_in = o3.Irreps(irreps_in)
            irr_out = o3.Irreps(irreps_out)
            irr_sh = o3.Irreps(irreps_sh)

            self.tp = o3.FullyConnectedTensorProduct(
                irr_in,
                irr_sh,
                irr_out,
                internal_weights=False,
                shared_weights=False,
            )
            n_tp_weights = self.tp.weight_numel
            self.self_interaction = o3.Linear(irr_in, irr_out)
        else:
            dim_in = parse_scalar_irreps(irreps_in)
            dim_out = parse_scalar_irreps(irreps_out)
            if dim_in is None or dim_out is None:
                raise ImportError(
                    "Non-scalar irreps "
                    f"({irreps_in!r} -> {irreps_out!r}) require e3nn. Install with "
                    "pip install 'gmd-sgt[e3nn]' or use scalar-only irreps such as '64x0e'."
                )
            self.tp = ScalarTensorProduct(dim_in, dim_out)
            n_tp_weights = self.tp.weight_numel
            self.self_interaction = ScalarLinear(dim_in, dim_out)

        self.radial_net = nn.Sequential(
            nn.Linear(n_basis, hidden_radial),
            nn.SiLU(),
            nn.Linear(hidden_radial, hidden_radial),
            nn.SiLU(),
            nn.Linear(hidden_radial, n_tp_weights),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_radial: torch.Tensor,
        n_atoms: int,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        tp_weights = self.radial_net(edge_radial)

        messages = self.tp(h[src], edge_sh, tp_weights)
        if edge_weight is not None:
            messages = messages * edge_weight.unsqueeze(-1)
        agg = scatter_sum(messages, dst, n_atoms) / math.sqrt(self.avg_neighbors)
        return self.self_interaction(h) + agg


__all__ = ["SE3EquivariantMessagePassing", "parse_scalar_irreps", "spherical_harmonics_irreps"]
