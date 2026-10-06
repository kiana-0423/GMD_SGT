"""Sparse local-attention residual correction branch."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .geometry import scatter_sum
from .readout import AtomicEnergyReadout


def _segment_softmax(
    scores: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Softmax over edge groups with the same destination node.

    Without ``edge_weight`` this is the ordinary per-node softmax. With an
    edge weight ``w_e >= 0`` (the smooth cutoff envelope) the attention is

        a_e = w_e exp(s_e) / (1 + sum_{e' -> i} w_e' exp(s_e'))

    i.e. each edge's contribution is scaled by its envelope and every node
    attends to an implicit "null" key with logit 0. Both the numerator and
    the normaliser are continuous when an edge crosses the cutoff
    (``w_e -> 0``), including when a node loses its last neighbour, and an
    edge with ``w_e = 0`` is exactly equivalent to removing it.

    Evaluation: with ``t_e = s_e + log w_e`` (``-inf`` for ``w_e = 0``) and the
    detached stabiliser ``m_i = max(0, max_e t_e)``,
    ``a_e = w_e exp(s_e - m_i) / (exp(-m_i) + sum_e' w_e' exp(s_e' - m_i))``.
    Every factor ``exp(t_e - m_i)`` is at most 1 and the denominator contains a
    term equal to 1 (the null key or the maximising edge), so no clamp is
    needed. Zero-weight edges contribute exactly zero value and gradient
    (their weight derivative is multiplied by the envelope derivative, which
    also vanishes at and beyond the cutoff).
    """
    # Branch-free for empty inputs (see ``scatter_sum``).
    expanded_index = index.unsqueeze(-1).expand(-1, scores.shape[-1])
    if edge_weight is None:
        max_scores = torch.full(
            (dim_size, scores.shape[-1]),
            float("-inf"),
            dtype=scores.dtype,
            device=scores.device,
        ).scatter_reduce(0, expanded_index, scores.detach(), reduce="amax", include_self=True)
        exp_scores = (scores - max_scores[index]).exp()
        normalizer = scores.new_zeros((dim_size, scores.shape[-1]))
        normalizer = normalizer.scatter_add(0, expanded_index, exp_scores)
        return exp_scores / normalizer[index]

    weight = edge_weight.unsqueeze(-1).expand_as(scores)
    positive = weight > 0
    # Weights below 1e-30 use 1e-30 only for the stabiliser's log; the result
    # still uses the exact weight (literal: TorchScript cannot read globals).
    weight_floor = weight.detach().clamp_min(1e-30)
    log_weight = weight_floor.log()
    shifted = scores.detach() + log_weight
    # Stabiliser over positive-weight edges and the null key (logit 0).
    max_shifted = torch.zeros(
        (dim_size, scores.shape[-1]), dtype=scores.dtype, device=scores.device
    ).scatter_reduce(
        0,
        expanded_index,
        torch.where(positive, shifted, torch.full_like(shifted, float("-inf"))),
        reduce="amax",
        include_self=True,
    )
    # exp(s + log w_floor - m) <= 1 on positive edges; masked edges use 0.
    exponent = torch.where(
        positive, scores + log_weight - max_shifted[index], torch.zeros_like(scores)
    )
    numerator = torch.where(
        positive, exponent.exp() * (weight / weight_floor), torch.zeros_like(scores)
    )
    normalizer = (-max_shifted).exp().scatter_add(0, expanded_index, numerator)
    return numerator / normalizer[index]


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
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_nodes = node_features.shape[0]
        src, dst = edge_index[0], edge_index[1]

        q = self.q_proj(node_features).view(n_nodes, self.n_heads, self.head_dim)
        k = self.k_proj(node_features).view(n_nodes, self.n_heads, self.head_dim)
        v = self.v_proj(node_features).view(n_nodes, self.n_heads, self.head_dim)

        logits = (q[dst] * k[src]).sum(dim=-1) * self.scale + self.bias_mlp(edge_features)
        attn = _segment_softmax(logits, dst, n_nodes, edge_weight)
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
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Residual atomic energies; see :func:`_segment_softmax` for ``edge_weight``."""
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
            x = layer(x, edge_index, edge_features, edge_weight)
        return self.readout(x)
