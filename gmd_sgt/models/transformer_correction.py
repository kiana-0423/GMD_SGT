"""Sparse local-attention residual correction branch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .geometry import scatter_sum
from .readout import AtomicEnergyReadout


def _segment_softmax(
    scores: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Softmax over edge groups with the same destination node."""
    if scores.numel() == 0:
        return scores

    expanded_index = index.unsqueeze(-1).expand(-1, scores.shape[-1])
    max_scores = torch.full(
        (dim_size, scores.shape[-1]),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    max_scores.scatter_reduce_(
        0,
        expanded_index,
        scores,
        reduce="amax",
        include_self=True,
    )
    stabilized = scores - max_scores[index]
    exp_scores = stabilized.exp()
    normalizer = scores.new_zeros((dim_size, scores.shape[-1]))
    normalizer.scatter_add_(0, expanded_index, exp_scores)
    return exp_scores / normalizer[index].clamp_min(1e-12)


class _SparseAttentionLayer(nn.Module):
    """Attention restricted to the local neighbor graph."""

    def __init__(
        self,
        hidden_channels: int,
        edge_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_channels % n_heads != 0:
            raise ValueError("hidden_channels must be divisible by n_heads")
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.head_dim = hidden_channels // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.bias_mlp = nn.Sequential(
            nn.Linear(edge_dim, n_heads),
            nn.SiLU(),
            nn.Linear(n_heads, n_heads),
        )
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes = node_features.shape[0]
        src, dst = edge_index[0], edge_index[1]

        q = self.q_proj(node_features).view(n_nodes, self.n_heads, self.head_dim)
        k = self.k_proj(node_features).view(n_nodes, self.n_heads, self.head_dim)
        v = self.v_proj(node_features).view(n_nodes, self.n_heads, self.head_dim)

        logits = (q[dst] * k[src]).sum(dim=-1) * self.scale + self.bias_mlp(edge_features)
        attn = _segment_softmax(logits, dst, n_nodes)
        weighted_values = attn.unsqueeze(-1) * v[src]
        aggregated = scatter_sum(weighted_values, dst, n_nodes).reshape(
            n_nodes,
            self.hidden_channels,
        )

        x = self.norm1(node_features + self.dropout(self.out_proj(aggregated)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class TransformerCorrection(nn.Module):
    """Sparse local Transformer branch predicting atomic energy corrections."""

    def __init__(
        self,
        n_species: int,
        input_channels: int,
        hidden_channels: int,
        edge_dim: int,
        num_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.species_embedding = nn.Embedding(n_species, hidden_channels, padding_idx=0)
        self.input_proj = nn.Linear(input_channels + hidden_channels + 1, hidden_channels)
        self.layers = nn.ModuleList(
            [
                _SparseAttentionLayer(
                    hidden_channels=hidden_channels,
                    edge_dim=edge_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                )
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
            )
        return self.readout(x)
