"""Top-level unified equivariant MLIP model."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .blocks import EquivariantLongRangeBlock
from .dependencies import CLUSTER_AVAILABLE, E3NN_AVAILABLE, o3, radius_graph
from .radial import BesselBasis, PolynomialCutoff


class UnifiedEquivariantMLIP(nn.Module):
    """Unified Equivariant Machine Learning Interatomic Potential."""

    def __init__(
        self,
        n_species: int = 100,
        n_blocks: int = 4,
        scalar_dim: int = 128,
        irreps: str = "128x0e + 64x1o + 32x2e",
        n_basis: int = 8,
        local_cutoff: float = 5.0,
        lr_cutoff: float = 12.0,
        l_max: int = 2,
        long_range_type: str = "invariant_attention",
        n_heads: int = 4,
        avg_neighbors: float = 10.0,
        dropout: float = 0.0,
        atomic_energies: Optional[Dict[int, float]] = None,
    ):
        super().__init__()

        self.scalar_dim = scalar_dim
        self.local_cutoff = local_cutoff
        self.lr_cutoff = lr_cutoff
        self.long_range_type = long_range_type
        self.l_max = l_max

        self.species_embedding = nn.Embedding(n_species, scalar_dim, padding_idx=0)

        self.radial_basis = BesselBasis(local_cutoff, n_basis)
        self.cutoff_env = PolynomialCutoff(local_cutoff)

        if long_range_type not in ("none", None, "electrostatic"):
            self.radial_basis_lr = BesselBasis(lr_cutoff, n_basis)
            self.cutoff_env_lr = PolynomialCutoff(lr_cutoff)
        else:
            self.radial_basis_lr = None
            self.cutoff_env_lr = None

        self.blocks = nn.ModuleList(
            [
                EquivariantLongRangeBlock(
                    irreps=irreps,
                    scalar_dim=scalar_dim,
                    n_basis=n_basis,
                    n_heads=n_heads,
                    long_range_type=long_range_type,
                    hidden_radial=64,
                    avg_neighbors=avg_neighbors,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        self.energy_head = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim // 2),
            nn.SiLU(),
            nn.Linear(scalar_dim // 2, 1),
        )

        e_ref = torch.zeros(n_species)
        if atomic_energies is not None:
            for atomic_number, energy in atomic_energies.items():
                e_ref[atomic_number] = energy
        self.register_buffer("atomic_energies_ref", e_ref)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.energy_head[-1].weight)
        nn.init.zeros_(self.energy_head[-1].bias)

    def build_neighbor_graph(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cutoff: float,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del pbc

        if cell is None:
            if CLUSTER_AVAILABLE:
                edge_index = radius_graph(positions, r=cutoff, batch=batch, loop=False)
                return edge_index, None

            diff = positions.unsqueeze(0) - positions.unsqueeze(1)
            dist = diff.norm(dim=-1)
            same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
            edge_mask = (dist < cutoff) & (dist > 0) & same_graph
            src, dst = edge_mask.nonzero(as_tuple=True)
            edge_index = torch.stack([src, dst], dim=0)
            return edge_index, None

        # ── PBC path: delegate to ASE neighbor list ──────────────────────────
        if cell.dim() == 2:
            # Single structure (e.g. from MLIPCalculator.compute())
            return self._build_neighbor_graph_pbc(positions, cell, cutoff)

        # Batched structures from collate_fn: cell is [n_graphs, 3, 3].
        # Build per-graph neighbor tables and concatenate with offset indices.
        n_graphs = cell.shape[0]
        expected_graphs = int(batch.max().item()) + 1
        if n_graphs != expected_graphs:
            raise ValueError(
                f"cell has {n_graphs} graphs but batch encodes {expected_graphs} graphs"
            )

        all_ei = []
        all_es = []
        offset = 0
        for g in range(n_graphs):
            mask = batch == g
            if not mask.any():
                continue
            pos_g = positions[mask]
            ei_g, es_g = self._build_neighbor_graph_pbc(pos_g, cell[g], cutoff)
            all_ei.append(ei_g + offset)
            all_es.append(es_g)
            offset += int(mask.sum().item())

        if not all_ei:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=positions.device),
                torch.zeros((0, 3), dtype=positions.dtype, device=positions.device),
            )

        return torch.cat(all_ei, dim=1), torch.cat(all_es, dim=0)

    def _build_neighbor_graph_pbc(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        cutoff: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build PBC neighbor graph using ASE.

        Returns
        -------
        edge_index : LongTensor [2, E]
        edge_shift : FloatTensor [E, 3]  Cartesian shift vectors (Å)
        """
        try:
            import numpy as np
            from ase.neighborlist import neighbor_list as ase_nl
        except ImportError as exc:
            raise ImportError(
                "ASE is required for PBC neighbor lists. "
                "Install with: pip install ase"
            ) from exc

        device = positions.device
        dtype = positions.dtype
        pos_np = positions.detach().cpu().numpy()
        cell_np = cell.detach().cpu().numpy()

        # ase_nl returns (i, j, S) where S[k] is integer lattice-vector indices
        i_idx, j_idx, S = ase_nl(
            "ijS",
            pbc=[True, True, True],
            cell=cell_np,
            positions=pos_np,
            cutoff=cutoff,
        )

        edge_index = torch.tensor(
            np.stack([i_idx, j_idx], axis=0), dtype=torch.long, device=device
        )
        # S @ cell_np converts fractional shifts → Cartesian (Å)
        edge_shift = torch.tensor(
            S @ cell_np, dtype=dtype, device=device
        )
        return edge_index, edge_shift

    def compute_edge_features(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        r_vec = positions[dst] - positions[src]
        if edge_shift is not None:
            r_vec = r_vec + edge_shift
        r = r_vec.norm(dim=-1)
        r_hat = r_vec / (r.unsqueeze(-1) + 1e-8)

        if E3NN_AVAILABLE:
            edge_sh = o3.spherical_harmonics(
                list(range(self.l_max + 1)),
                r_hat,
                normalize=True,
                normalization="component",
            )
        else:
            edge_sh = r_hat

        rbf = self.radial_basis(r)
        envelope = self.cutoff_env(r)
        edge_rbf = rbf * envelope.unsqueeze(-1)
        return r, edge_sh, edge_rbf

    def forward(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_shift: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        compute_forces: bool = True,
        compute_stress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if compute_forces and not positions.requires_grad:
            positions = positions.requires_grad_(True)

        n_atoms = positions.shape[0]
        n_graphs = int(batch.max().item()) + 1

        if edge_index is None:
            edge_index, edge_shift = self.build_neighbor_graph(
                positions,
                batch,
                self.local_cutoff,
                cell=cell,
            )

        _, edge_sh, edge_rbf = self.compute_edge_features(
            positions,
            edge_index,
            edge_shift,
        )

        h_scalar = self.species_embedding(species)
        h = h_scalar.clone()

        total_elec_energy: Optional[torch.Tensor] = None
        for block in self.blocks:
            h, h_scalar = block(
                h=h,
                h_scalar=h_scalar,
                edge_index=edge_index,
                edge_sh=edge_sh,
                edge_radial=edge_rbf,
                batch=batch,
                positions=positions,
                n_atoms=n_atoms,
            )
            if block._elec_energy is not None:
                if total_elec_energy is None:
                    total_elec_energy = block._elec_energy
                else:
                    total_elec_energy = total_elec_energy + block._elec_energy

        e_atomic = self.energy_head(h_scalar).squeeze(-1)
        e_atomic = e_atomic + self.atomic_energies_ref[species]

        e_total = torch.zeros(n_graphs, device=positions.device, dtype=e_atomic.dtype)
        e_total.scatter_add_(0, batch, e_atomic)

        if total_elec_energy is not None:
            e_total = e_total + total_elec_energy

        results: Dict[str, torch.Tensor] = {"energy": e_total}

        if compute_forces:
            grads = torch.autograd.grad(
                outputs=[e_total.sum()],
                inputs=[positions],
                create_graph=self.training,
                retain_graph=compute_stress,
                allow_unused=True,
            )
            if grads[0] is not None:
                results["forces"] = -grads[0]

        if compute_stress:
            results["stress"] = None

        return results


__all__ = ["UnifiedEquivariantMLIP"]
