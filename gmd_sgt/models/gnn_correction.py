"""Invariant GNN residual correction branch for staged MLIPs."""

from __future__ import annotations

import torch
import torch.nn as nn

from .geometry import scatter_sum
from .readout import AtomicEnergyReadout


class _ResidualMessageLayer(nn.Module):
    """Simple invariant message-passing block for residual atomic energies."""

    def __init__(self, hidden_channels: int, edge_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + edge_dim + 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        coordination: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes = node_features.shape[0]
        src, dst = edge_index[0], edge_index[1]

        edge_input = torch.cat(
            [
                node_features[src],
                node_features[dst],
                edge_features,
                coordination[src].unsqueeze(-1),
                coordination[dst].unsqueeze(-1),
            ],
            dim=-1,
        )
        messages = self.message_mlp(edge_input)
        aggregated = scatter_sum(messages, dst, n_nodes)
        update_input = torch.cat(
            [
                node_features,
                aggregated,
                coordination.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.norm(node_features + self.update_mlp(update_input))


class GNNCorrection(nn.Module):
    """Residual GNN branch predicting atomic energy corrections only."""

    def __init__(
        self,
        n_species: int,
        input_channels: int,
        hidden_channels: int,
        edge_dim: int,
        num_layers: int = 2,
        species_embedding_dim: int | None = None,
    ):
        super().__init__()
        species_embedding_dim = species_embedding_dim or hidden_channels
        self.species_embedding = nn.Embedding(n_species, species_embedding_dim, padding_idx=0)
        self.input_proj = nn.Linear(
            input_channels + species_embedding_dim + 1,
            hidden_channels,
        )
        self.layers = nn.ModuleList(
            [
                _ResidualMessageLayer(hidden_channels=hidden_channels, edge_dim=edge_dim)
                for _ in range(num_layers)
            ]
        )
        self.readout = AtomicEnergyReadout(hidden_channels)

    def forward(
        self,
        node_features: torch.Tensor,
        species: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        coordination: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                node_features,
                self.species_embedding(species),
                coordination.unsqueeze(-1),
            ],
            dim=-1,
        )
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(
                node_features=x,
                edge_index=edge_index,
                edge_features=edge_features,
                coordination=coordination,
            )
        return self.readout(x)
