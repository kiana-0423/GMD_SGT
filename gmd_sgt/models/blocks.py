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
from .message_passing import (
    SE3EquivariantMessagePassing,
    parse_scalar_irreps,
    spherical_harmonics_irreps,
)
from .scalar_irreps import ScalarBatchNorm

LONG_RANGE_TYPES = ("none", "invariant_attention", "equivariant_attention", "electrostatic")


class EquivariantLongRangeBlock(nn.Module):
    """One unified local-plus-long-range interaction block.

    The spherical-harmonic irreps used by the local tensor product are derived
    from ``l_max`` and must match the edge harmonics computed by the model.
    """

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
        l_max: int = 2,
    ):
        super().__init__()
        long_range_type = "none" if long_range_type is None else str(long_range_type)
        if long_range_type not in LONG_RANGE_TYPES:
            raise ValueError(
                f"Unknown long_range_type {long_range_type!r}; expected one of {LONG_RANGE_TYPES}"
            )
        self.long_range_type = long_range_type
        self.scalar_dim = scalar_dim
        self.l_max = int(l_max)

        self.local_mp = SE3EquivariantMessagePassing(
            irreps_in=irreps,
            irreps_out=irreps,
            irreps_sh=spherical_harmonics_irreps(l_max),
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
            dim = parse_scalar_irreps(irreps)
            if dim is None:
                raise ImportError(f"Non-scalar irreps {irreps!r} require e3nn")
            self.norm_equivariant = ScalarBatchNorm(dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim * 2),
            nn.SiLU(),
            nn.Linear(scalar_dim * 2, scalar_dim),
        )
        self._elec_energy: Optional[torch.Tensor] = None

    def forward_with_energy(
        self,
        h: torch.Tensor,
        h_scalar: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_radial: torch.Tensor,
        batch: torch.Tensor,
        positions: Optional[torch.Tensor],
        n_atoms: int,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return updated ``(h, h_scalar)`` and the optional electrostatic energy."""
        h_local = self.local_mp(h, edge_index, edge_sh, edge_radial, n_atoms, edge_weight)
        s_local = h_local[:, : self.scalar_dim]
        s_fused = s_local
        h_local_out = h_local
        elec_energy: Optional[torch.Tensor] = None

        if self.long_range is not None:
            s_lr, h_lr, elec_energy = self.long_range.block_forward(h, h_scalar, positions, batch)
            if s_lr is not None and self.gate is not None:
                g = self.gate(torch.cat([s_local, s_lr], dim=-1))
                s_fused = g * s_local + (1.0 - g) * s_lr
            if h_lr is not None:
                h_local_out = h_local + h_lr

        h_scalar_new = self.norm_scalar(h_scalar + self.feed_forward(s_fused))

        if self.norm_equivariant is not None:
            h_new = self.norm_equivariant(h + h_local_out)
        else:
            h_new = h + h_local_out
        return h_new, h_scalar_new, elec_energy

    @torch.jit.unused
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
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if n_atoms is None:
            n_atoms = h.shape[0]
        h_new, h_scalar_new, elec_energy = self.forward_with_energy(
            h,
            h_scalar,
            edge_index,
            edge_sh,
            edge_radial,
            batch,
            positions,
            n_atoms,
            edge_weight,
        )
        self._elec_energy = elec_energy
        return h_new, h_scalar_new


__all__ = ["EquivariantLongRangeBlock", "LONG_RANGE_TYPES"]
