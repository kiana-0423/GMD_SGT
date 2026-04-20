"""Composite interaction blocks for the unified model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .dependencies import E3NN_AVAILABLE, IrrepsBatchNorm, o3
from .long_range import (
    ElectrostaticCorrection,
    EquivariantLongRangeAttention,
    InvariantScalarAttention,
)
from .message_passing import SE3EquivariantMessagePassing


class EquivariantLongRangeBlock(nn.Module):
    """One unified local-plus-long-range interaction block."""

    def __init__(
        self,
        irreps: str,
        scalar_dim: int,
        n_basis: int = 8,
        n_heads: int = 4,
        long_range_type: str = "invariant_attention",
        hidden_radial: int = 64,
        avg_neighbors: float = 10.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.long_range_type = long_range_type
        self.scalar_dim = scalar_dim

        self.local_mp = SE3EquivariantMessagePassing(
            irreps_in=irreps,
            irreps_out=irreps,
            irreps_sh="1x0e + 1x1o + 1x2e",
            n_basis=n_basis,
            hidden_radial=hidden_radial,
            avg_neighbors=avg_neighbors,
        )

        if long_range_type == "invariant_attention":
            self.long_range: Optional[nn.Module] = InvariantScalarAttention(
                scalar_dim=scalar_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
        elif long_range_type == "equivariant_attention":
            self.long_range = EquivariantLongRangeAttention(
                irreps=irreps,
                scalar_dim=scalar_dim,
                n_heads=n_heads,
            )
        elif long_range_type == "electrostatic":
            self.long_range = ElectrostaticCorrection(scalar_dim=scalar_dim)
        else:
            self.long_range = None

        if self.long_range is not None and long_range_type != "electrostatic":
            self.gate = nn.Sequential(
                nn.Linear(scalar_dim * 2, scalar_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        self.norm_scalar = nn.LayerNorm(scalar_dim)
        if E3NN_AVAILABLE:
            self.norm_equivariant = IrrepsBatchNorm(o3.Irreps(irreps))
        else:
            self.norm_equivariant = None

        self.feed_forward = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim * 2),
            nn.SiLU(),
            nn.Linear(scalar_dim * 2, scalar_dim),
        )
        self._elec_energy: Optional[torch.Tensor] = None

    def forward(
        self,
        h: torch.Tensor,
        h_scalar: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_radial: torch.Tensor,
        batch: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        n_atoms: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if n_atoms is None:
            n_atoms = h.shape[0]

        h_local = self.local_mp(h, edge_index, edge_sh, edge_radial, n_atoms)
        s_local = h_local[:, : self.scalar_dim]
        elec_energy: Optional[torch.Tensor] = None

        if self.long_range is None or self.long_range_type == "none":
            s_fused = s_local
            h_local_out = h_local
        elif self.long_range_type == "invariant_attention":
            s_lr = self.long_range(h_scalar, batch)
            gate_input = torch.cat([s_local, s_lr], dim=-1)
            g = self.gate(gate_input)
            s_fused = g * s_local + (1.0 - g) * s_lr
            h_local_out = h_local
        elif self.long_range_type == "equivariant_attention":
            h_lr = self.long_range(h, h_scalar, batch)
            s_lr = h_lr[:, : self.scalar_dim]
            gate_input = torch.cat([s_local, s_lr], dim=-1)
            g = self.gate(gate_input)
            s_fused = g * s_local + (1.0 - g) * s_lr
            h_local_out = h_local + h_lr
        elif self.long_range_type == "electrostatic":
            assert positions is not None, "positions required for electrostatic module"
            elec_energy, _ = self.long_range(h_scalar, positions, batch)
            s_fused = s_local
            h_local_out = h_local
        else:
            s_fused = s_local
            h_local_out = h_local

        h_scalar_new = self.norm_scalar(h_scalar + self.feed_forward(s_fused))

        if self.norm_equivariant is not None and E3NN_AVAILABLE:
            h_new = self.norm_equivariant(h + h_local_out)
        else:
            h_new = h + h_local_out

        self._elec_energy = elec_energy
        return h_new, h_scalar_new


__all__ = ["EquivariantLongRangeBlock"]
