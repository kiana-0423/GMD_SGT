"""Shared readout heads for conservative MLIP models."""

from __future__ import annotations

import torch
import torch.nn as nn


class AtomicEnergyReadout(nn.Module):
    """Small MLP mapping invariant node features to atomic energies."""

    def __init__(self, input_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim // 2, 8)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.net(node_features).squeeze(-1)
