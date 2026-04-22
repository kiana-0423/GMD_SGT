"""Lightweight Allegro-style local backbone for staged residual MLIPs."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .geometry import (
    build_neighbor_graph,
    compute_edge_geometry,
    directional_basis,
    scatter_sum,
)
from .radial import BesselBasis, PolynomialCutoff
from .readout import AtomicEnergyReadout


class _LocalTensorProductLayer(nn.Module):
    """Approximate local tensor-product interaction layer.

    The layer builds edge filters from species-conditioned node features and
    radial bases, mixes them with directional bases up to ``l_max``, and then
    reduces the directional channels back to invariant node features.
    """

    def __init__(
        self,
        hidden_channels: int,
        n_basis: int,
        l_max: int,
        avg_neighbors: float = 12.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.l_max = l_max
        self.avg_neighbors = max(avg_neighbors, 1.0)

        self.src_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dst_proj = nn.Linear(hidden_channels, hidden_channels)
        self.filter_net = nn.Sequential(
            nn.Linear(hidden_channels * 2 + n_basis, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Update input layout:
        #   [node_features, scalar_agg, l=1 invariant, l=2 invariant, ...]
        update_dim = hidden_channels * (2 + l_max)
        self.update = nn.Sequential(
            nn.Linear(update_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
        basis_per_l: list[torch.Tensor],
    ) -> torch.Tensor:
        n_nodes = node_features.shape[0]
        src, dst = edge_index[0], edge_index[1]

        edge_input = torch.cat(
            [
                self.src_proj(node_features[src]),
                self.dst_proj(node_features[dst]),
                edge_rbf,
            ],
            dim=-1,
        )
        edge_message = self.filter_net(edge_input)

        scalar_agg = scatter_sum(edge_message, dst, n_nodes) / self.avg_neighbors

        invariant_parts = []
        for basis in basis_per_l[1:]:
            # edge_message[:, :, None] * basis[:, None, :] -> [E, H, 2l+1]
            # reduce over incoming edges, then take squared norm over m to get
            # invariant local environment descriptors [N, H].
            tensor_field = scatter_sum(
                edge_message.unsqueeze(-1) * basis.unsqueeze(1),
                dst,
                n_nodes,
            )
            invariant_parts.append(tensor_field.pow(2).mean(dim=-1))

        update_input = torch.cat([node_features, scalar_agg] + invariant_parts, dim=-1)
        return self.norm(node_features + self.update(update_input))


class AllegroStyleBackbone(nn.Module):
    """Local conservative backbone with Allegro-style directional interactions.

    This is a lightweight, dependency-friendly implementation intended as the
    first staged backbone. When ``e3nn`` is available it uses spherical
    harmonics for directional encoding; otherwise it falls back to low-order
    cartesian directional tensors while preserving the same interface.
    """

    def __init__(
        self,
        n_species: int = 100,
        hidden_channels: int = 64,
        num_layers: int = 2,
        n_basis: int = 8,
        cutoff: float = 5.0,
        l_max: int = 2,
        avg_neighbors: float = 12.0,
        atomic_energies: Optional[Dict[int, float]] = None,
    ):
        super().__init__()
        self.n_species = n_species
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.n_basis = n_basis
        self.local_cutoff = cutoff
        self.lr_cutoff = cutoff
        self.l_max = l_max
        self.avg_neighbors = avg_neighbors

        self.species_embedding = nn.Embedding(n_species, hidden_channels, padding_idx=0)
        self.radial_basis = BesselBasis(cutoff, n_basis)
        self.cutoff_env = PolynomialCutoff(cutoff)
        self.layers = nn.ModuleList(
            [
                _LocalTensorProductLayer(
                    hidden_channels=hidden_channels,
                    n_basis=n_basis,
                    l_max=l_max,
                    avg_neighbors=avg_neighbors,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = AtomicEnergyReadout(hidden_channels)

        energy_ref = torch.zeros(n_species, dtype=torch.float32)
        if atomic_energies is not None:
            for atomic_number, energy in atomic_energies.items():
                energy_ref[int(atomic_number)] = float(energy)
        self.register_buffer("atomic_energies_ref", energy_ref)

    def build_neighbor_graph(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cutoff: float,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ):
        """Expose graph construction for inference/export parity."""
        return build_neighbor_graph(
            positions=positions,
            batch=batch,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
        )

    def forward(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_shift: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        neighbor_list: Optional[dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]] = None,
        compute_forces: bool = True,
        compute_stress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del compute_stress

        if compute_forces and not positions.requires_grad:
            positions = positions.requires_grad_(True)

        if neighbor_list is not None:
            edge_index, edge_shift = self._resolve_neighbor_list(
                neighbor_list=neighbor_list,
                edge_index=edge_index,
                edge_shift=edge_shift,
                positions=positions,
            )

        if edge_index is None:
            edge_index, edge_shift = self.build_neighbor_graph(
                positions=positions,
                batch=batch,
                cutoff=self.local_cutoff,
                cell=cell,
            )

        _, distances, unit_vec = compute_edge_geometry(
            positions=positions,
            edge_index=edge_index,
            edge_shift=edge_shift,
        )
        envelope = self.cutoff_env(distances)
        edge_rbf = self.radial_basis(distances) * envelope.unsqueeze(-1)
        basis_per_l = directional_basis(unit_vec, self.l_max)
        coordination = scatter_sum(
            envelope.unsqueeze(-1),
            edge_index[1],
            positions.shape[0],
        ).squeeze(-1)

        node_features = self.species_embedding(species)
        for layer in self.layers:
            node_features = layer(
                node_features=node_features,
                edge_index=edge_index,
                edge_rbf=edge_rbf,
                basis_per_l=basis_per_l,
            )

        atomic_energy = self.readout(node_features) + self.atomic_energies_ref[species]
        n_graphs = int(batch.max().item()) + 1
        total_energy = scatter_sum(atomic_energy.unsqueeze(-1), batch, n_graphs).squeeze(-1)

        output: Dict[str, torch.Tensor] = {
            "energy": total_energy,
            "atomic_energies": atomic_energy,
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_shift": (
                edge_shift
                if edge_shift is not None
                else positions.new_zeros((edge_index.shape[1], 3))
            ),
            "distances": distances,
            "edge_rbf": edge_rbf,
            "coordination": coordination,
        }

        if compute_forces:
            grad = torch.autograd.grad(
                outputs=[total_energy.sum()],
                inputs=[positions],
                create_graph=self.training,
                retain_graph=False,
                allow_unused=True,
            )[0]
            output["forces"] = -grad if grad is not None else torch.zeros_like(positions)

        return output

    @staticmethod
    def _resolve_neighbor_list(
        neighbor_list: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        edge_index: Optional[torch.Tensor],
        edge_shift: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize explicit neighbor-list inputs to ``edge_index`` / ``edge_shift``.

        Accepted forms:
          - ``{"edge_index": ..., "edge_shift": ...}``
          - ``{"index": ..., "shift": ...}``
          - ``(edge_index, edge_shift)``
        """
        if edge_index is not None:
            raise ValueError("Pass either neighbor_list or edge_index/edge_shift, not both")

        if isinstance(neighbor_list, tuple):
            resolved_edge_index, resolved_edge_shift = neighbor_list
        else:
            resolved_edge_index = neighbor_list.get("edge_index", neighbor_list.get("index"))
            resolved_edge_shift = neighbor_list.get("edge_shift", neighbor_list.get("shift"))

        if resolved_edge_index is None:
            raise ValueError("neighbor_list must provide edge_index/index")
        if resolved_edge_shift is None:
            resolved_edge_shift = positions.new_zeros((resolved_edge_index.shape[1], 3))

        return resolved_edge_index, resolved_edge_shift
