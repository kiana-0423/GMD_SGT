"""Long-range interaction modules."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dependencies import E3NN_AVAILABLE, SCATTER_AVAILABLE, o3, scatter_mean


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
            irr_scalar = int(irreps.split("x0")[0])
            self.v_proj = nn.Linear(irr_scalar, irr_scalar)

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


class ElectrostaticCorrection(nn.Module):
    """Physics-inspired screened Coulomb correction."""

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

        if SCATTER_AVAILABLE:
            q_mean = scatter_mean(charges, batch, dim=0)[batch]
        else:
            n_graphs = int(batch.max().item()) + 1
            q_sum = torch.zeros(n_graphs, device=charges.device).scatter_add_(
                0, batch, charges
            )
            n_per_graph = torch.zeros(n_graphs, device=charges.device).scatter_add_(
                0, batch, torch.ones_like(charges)
            )
            q_mean = (q_sum / n_per_graph)[batch]
        charges = charges - q_mean

        sigma = F.softplus(self.log_sigma) + 1e-4
        n_graphs = int(batch.max().item()) + 1
        e_elec = torch.zeros(n_graphs, device=positions.device)

        same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist = diff.norm(dim=-1)
        eye_mask = torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)
        mask = same_graph & ~eye_mask

        kernel = torch.where(
            mask,
            torch.erfc(dist / sigma) / (dist + 1e-8),
            torch.zeros_like(dist),
        )
        q_prod = charges.unsqueeze(0) * charges.unsqueeze(1)
        e_pair = 0.5 * q_prod * kernel

        e_per_atom = e_pair.sum(dim=1)
        e_elec.scatter_add_(0, batch, e_per_atom)
        return e_elec, charges


__all__ = [
    "ElectrostaticCorrection",
    "EquivariantLongRangeAttention",
    "InvariantScalarAttention",
]
