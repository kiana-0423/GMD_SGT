"""Long-range interaction modules.

Semantics
---------
The attention modules are *global within each structure*: every atom attends
to every other atom of the same graph, independent of distance and without
periodic images. They act on invariant (or equivariant) features that the
local, cutoff-based message passing has already made PBC-aware, so the
resulting energy is smooth and invariant to wrapping atoms into the cell.
No long-range module uses a distance cutoff; ``lr_cutoff`` is therefore not
consumed by any of them.

:class:`ElectrostaticCorrection` sums a screened Coulomb kernel over all
pairs of the same structure using bare positions. It is only valid for
non-periodic structures (no Ewald summation), and the unified model rejects
periodic inputs when it is enabled.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dependencies import E3NN_AVAILABLE, o3
from .geometry import scatter_sum
from .message_passing import parse_scalar_irreps
from .scalar_irreps import ScalarLinear


class InvariantScalarAttention(nn.Module):
    """Multi-head attention over invariant scalar features."""

    def __init__(self, scalar_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert scalar_dim % n_heads == 0, "scalar_dim must be divisible by n_heads"
        self.scalar_dim = scalar_dim
        self.n_heads = n_heads
        self.head_dim = scalar_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(scalar_dim, scalar_dim)
        self.k_proj = nn.Linear(scalar_dim, scalar_dim)
        self.v_proj = nn.Linear(scalar_dim, scalar_dim)
        self.out_proj = nn.Linear(scalar_dim, scalar_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_scalar: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        n_atoms = h_scalar.shape[0]
        n_heads, head_dim = self.n_heads, self.head_dim

        q = self.q_proj(h_scalar).view(n_atoms, n_heads, head_dim)
        k = self.k_proj(h_scalar).view(n_atoms, n_heads, head_dim)
        v = self.v_proj(h_scalar).view(n_atoms, n_heads, head_dim)

        same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
        attn = torch.einsum("ihd,jhd->ijh", q, k) * self.scale
        attn = attn.masked_fill(~same_graph.unsqueeze(-1), float("-inf"))
        attn = torch.softmax(attn, dim=1)
        attn = self.dropout(attn)

        out = torch.einsum("ijh,jhd->ihd", attn, v).reshape(n_atoms, self.scalar_dim)
        return self.out_proj(out)

    def block_forward(
        self,
        h: torch.Tensor,
        h_scalar: torch.Tensor,
        positions: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Uniform interface used by the interaction block: (s_lr, h_lr, energy)."""
        return self.forward(h_scalar, batch), None, None


class EquivariantLongRangeAttention(nn.Module):
    """Attention over full equivariant features with invariant weights."""

    def __init__(self, irreps: str, scalar_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.scalar_dim = scalar_dim
        self.scale = (scalar_dim // n_heads) ** -0.5

        self.q_proj = nn.Linear(scalar_dim, n_heads)
        self.k_proj = nn.Linear(scalar_dim, n_heads)

        if E3NN_AVAILABLE:
            self.v_proj = o3.Linear(o3.Irreps(irreps), o3.Irreps(irreps))
        else:
            # Without e3nn the features are scalars (``Nx0e``).
            dim = parse_scalar_irreps(irreps)
            if dim is None:
                raise ImportError(f"Non-scalar irreps {irreps!r} require e3nn")
            self.v_proj = ScalarLinear(dim, dim)

    def forward(
        self,
        h: torch.Tensor,
        h_scalar: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(h_scalar)
        k = self.k_proj(h_scalar)

        same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
        attn = (q.unsqueeze(1) * k.unsqueeze(0)).sum(-1) * self.scale
        attn = attn.masked_fill(~same_graph, float("-inf"))
        attn = torch.softmax(attn, dim=1)

        v = self.v_proj(h)
        return torch.einsum("ij,jd->id", attn, v)

    def block_forward(
        self,
        h: torch.Tensor,
        h_scalar: torch.Tensor,
        positions: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Uniform interface used by the interaction block: (s_lr, h_lr, energy)."""
        h_lr = self.forward(h, h_scalar, batch)
        return h_lr[:, : self.scalar_dim], h_lr, None


class ElectrostaticCorrection(nn.Module):
    """Physics-inspired screened Coulomb correction (non-periodic only)."""

    def __init__(self, scalar_dim: int, damping: float = 2.0):
        super().__init__()
        self.damping = damping
        self.charge_net = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim // 2),
            nn.SiLU(),
            nn.Linear(scalar_dim // 2, 1),
        )
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        h_scalar: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        charges = self.charge_net(h_scalar).squeeze(-1)

        n_graphs = int(batch.max().item()) + 1
        q_sum = scatter_sum(charges.unsqueeze(-1), batch, n_graphs).squeeze(-1)
        n_per_graph = scatter_sum(torch.ones_like(charges).unsqueeze(-1), batch, n_graphs).squeeze(-1)
        charges = charges - (q_sum / n_per_graph)[batch]

        sigma = F.softplus(self.log_sigma) + 1e-4

        same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
        eye_mask = torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device)
        mask = same_graph & ~eye_mask
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        # Safe distance: masked pairs (including i == j) get a dummy value so
        # the gradient of sqrt never sees zero.
        dist_sq = torch.where(mask, (diff * diff).sum(-1), torch.ones_like(diff[..., 0]))
        dist = dist_sq.sqrt()

        kernel = torch.where(
            mask,
            torch.erfc(dist / sigma) / dist,
            torch.zeros_like(dist),
        )
        q_prod = charges.unsqueeze(0) * charges.unsqueeze(1)
        e_pair = 0.5 * q_prod * kernel

        e_per_atom = e_pair.sum(dim=1)
        e_elec = scatter_sum(e_per_atom.unsqueeze(-1), batch, n_graphs).squeeze(-1)
        return e_elec, charges

    def block_forward(
        self,
        h: torch.Tensor,
        h_scalar: torch.Tensor,
        positions: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Uniform interface used by the interaction block: (s_lr, h_lr, energy)."""
        if positions is None:
            raise ValueError("positions required for electrostatic module")
        energy, _ = self.forward(h_scalar, positions, batch)
        return None, None, energy


__all__ = [
    "ElectrostaticCorrection",
    "EquivariantLongRangeAttention",
    "InvariantScalarAttention",
]
