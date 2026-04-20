"""Local SE(3)-equivariant message passing layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .dependencies import E3NN_AVAILABLE, SCATTER_AVAILABLE, o3, scatter_add


class SE3EquivariantMessagePassing(nn.Module):
    """
    One round of SE(3)-equivariant message passing.
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
            n_tp_weights = hidden_radial
            scalar_dim = int(irreps_in.split("x0")[0])
            self.self_interaction = nn.Linear(scalar_dim, scalar_dim)

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
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        tp_weights = self.radial_net(edge_radial)

        if E3NN_AVAILABLE:
            messages = self.tp(h[src], edge_sh, tp_weights)

            if SCATTER_AVAILABLE:
                agg = scatter_add(messages, dst, dim=0, dim_size=n_atoms)
            else:
                agg = torch.zeros(
                    n_atoms,
                    messages.shape[-1],
                    dtype=messages.dtype,
                    device=messages.device,
                )
                agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
            agg = agg / math.sqrt(self.avg_neighbors)

            h_self = self.self_interaction(h)
            return h_self + agg

        return self.self_interaction(h)


__all__ = ["SE3EquivariantMessagePassing"]
