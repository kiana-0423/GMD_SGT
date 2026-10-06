"""Top-level unified equivariant MLIP model."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .blocks import LONG_RANGE_TYPES, EquivariantLongRangeBlock
from .dependencies import E3NN_AVAILABLE, o3
from .derivatives import conservative_outputs, make_strain
from .geometry import (
    build_neighbor_graph,
    build_neighbor_graph_pbc,
    compute_edge_geometry,
    normalize_pbc,
)
from .message_passing import parse_scalar_irreps, spherical_harmonics_irreps
from .radial import BesselBasis, PolynomialCutoff

# Buffers of radial modules that older versions created but never used.
_LEGACY_UNUSED_KEYS = ("radial_basis_lr.freq",)


def _leading_scalar_count(irreps) -> int:
    """Number of ``0e`` channels at the start of an e3nn irreps sequence."""
    count = 0
    for mul, ir in irreps:
        if ir.l != 0 or ir.p != 1:
            break
        count += mul
    return count


def _unreachable_irreps(irreps, irreps_sh, n_blocks: int) -> List[str]:
    """Irreps in ``irreps`` that message passing can never populate.

    Features start as scalars only (``0e``); each block can reach
    ``ir_in x ir_sh`` for any reachable ``ir_in``. Channels never reached
    would stay identically zero.
    """
    targets = {ir for _, ir in irreps}
    reachable = {o3.Irrep("0e")}
    for _ in range(n_blocks):
        new = set(reachable)
        for ir_in in reachable:
            for _, ir_sh in irreps_sh:
                new.update(ir for ir in ir_in * ir_sh if ir in targets)
        reachable = new
    return sorted(str(ir) for ir in targets - reachable)


class UnifiedEquivariantMLIP(nn.Module):
    """Unified Equivariant Machine Learning Interatomic Potential.

    Requirements and semantics
    --------------------------
    * ``irreps`` must start with at least ``scalar_dim`` copies of ``0e``; the
      energy is read out from those invariant channels.
    * Initial node features are species-dependent scalars only; all ``l > 0``
      channels start at zero and are populated by tensor products with the
      edge spherical harmonics (``l <= l_max``). Configurations in which some
      declared irreps can never be populated are rejected.
    * Without e3nn only a single scalar term ``"Nx0e"`` (``N >= scalar_dim``)
      is supported; message passing is then an invariant, distance-dependent
      continuous-filter convolution computed by e3nn-free modules that match
      e3nn exactly, so such checkpoints load in either environment.
      Non-scalar irreps raise ``ImportError``.
    * Long-range attention is global within each structure and does not use
      ``lr_cutoff`` (kept only for configuration/checkpoint compatibility).
      ``"electrostatic"`` is non-periodic only.
    """

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
        long_range_type = "none" if long_range_type is None else str(long_range_type)
        if long_range_type not in LONG_RANGE_TYPES:
            raise ValueError(
                f"Unknown long_range_type {long_range_type!r}; expected one of {LONG_RANGE_TYPES}"
            )
        if int(l_max) != l_max or l_max < 0:
            raise ValueError(f"l_max must be a non-negative integer, got {l_max!r}")
        if local_cutoff <= 0:
            raise ValueError(f"local_cutoff must be positive, got {local_cutoff}")
        if lr_cutoff <= 0:
            raise ValueError(f"lr_cutoff must be positive, got {lr_cutoff}")

        self.scalar_dim = scalar_dim
        self.local_cutoff = float(local_cutoff)
        # Informational only: no long-range module consumes it (see class doc).
        self.lr_cutoff = float(lr_cutoff)
        self.long_range_type = long_range_type
        self.l_max = int(l_max)
        self._use_e3nn = E3NN_AVAILABLE

        self.species_embedding = nn.Embedding(n_species, scalar_dim, padding_idx=0)
        if self._use_e3nn:
            irr = o3.Irreps(irreps)
            if _leading_scalar_count(irr) < scalar_dim:
                raise ValueError(
                    f"irreps {irreps!r} must start with at least scalar_dim={scalar_dim} "
                    "even scalars (0e) for the invariant energy readout"
                )
            unreachable = _unreachable_irreps(
                irr, o3.Irreps(spherical_harmonics_irreps(self.l_max)), n_blocks
            )
            if unreachable:
                raise ValueError(
                    f"irreps {irreps!r} contain {unreachable}, which can never be populated "
                    f"from scalar inputs with l_max={self.l_max} and n_blocks={n_blocks}; "
                    "increase l_max/n_blocks or remove those irreps"
                )
            # Species information enters only through the 0e channels; a plain
            # linear map into l > 0 components would not transform correctly.
            self.input_proj = nn.Linear(scalar_dim, irr.dim)
            mask = torch.zeros(irr.dim)
            for sl, (_, ir) in zip(irr.slices(), irr):
                if ir.l == 0 and ir.p == 1:
                    mask[sl] = 1.0
            self.sh = o3.SphericalHarmonics(
                o3.Irreps(spherical_harmonics_irreps(self.l_max)),
                normalize=True,
                normalization="component",
            )
        else:
            n_scalars = parse_scalar_irreps(irreps)
            if n_scalars is None:
                raise ImportError(
                    f"UnifiedEquivariantMLIP with irreps {irreps!r} requires e3nn. Install "
                    "with pip install 'gmd-sgt[e3nn]', or use scalar-only irreps "
                    f"(e.g. irreps='{scalar_dim}x0e') for the invariant fallback model."
                )
            n_terms = len([t for t in str(irreps).replace(" ", "").split("+") if t])
            if n_terms != 1 or n_scalars < scalar_dim:
                raise ValueError(
                    "Without e3nn, irreps must be a single scalar term 'Nx0e' with "
                    f"N >= scalar_dim={scalar_dim}; got {irreps!r}"
                )
            self.input_proj = nn.Linear(scalar_dim, n_scalars)
            mask = torch.ones(n_scalars)
            self.sh = None
        self.register_buffer("input_irreps_mask", mask, persistent=False)

        self.radial_basis = BesselBasis(local_cutoff, n_basis)
        self.cutoff_env = PolynomialCutoff(local_cutoff)
        # Kept as attributes for backward compatibility; attention is global
        # and never used these modules.
        self.radial_basis_lr = None
        self.cutoff_env_lr = None
        self._register_load_state_dict_pre_hook(self._drop_legacy_keys)

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
                    l_max=self.l_max,
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
                e_ref[int(atomic_number)] = energy
        self.register_buffer("atomic_energies_ref", e_ref)

        self._init_weights()

    @staticmethod
    def _drop_legacy_keys(state_dict, prefix, *args, **kwargs):
        for key in _LEGACY_UNUSED_KEYS:
            state_dict.pop(prefix + key, None)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.energy_head[-1].weight)
        nn.init.zeros_(self.energy_head[-1].bias)

    @torch.jit.unused
    def build_neighbor_graph(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor,
        cutoff: float,
        cell: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Neighbor graph for non-periodic, fully or partially periodic batches."""
        return build_neighbor_graph(
            positions,
            batch,
            cutoff,
            cell=cell,
            pbc=pbc,
            pbc_builder=self._build_neighbor_graph_pbc,
        )

    @torch.jit.unused
    def _build_neighbor_graph_pbc(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        cutoff: float,
        pbc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a periodic neighbor graph for one structure using ASE.

        Returns
        -------
        edge_index : LongTensor [2, E]
        edge_shift : FloatTensor [E, 3]  Cartesian shift vectors (Å)
        """
        return build_neighbor_graph_pbc(positions, cell, cutoff, pbc)

    def _edge_features(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: Optional[torch.Tensor],
        strain: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, r, r_hat = compute_edge_geometry(positions, edge_index, edge_shift, strain, batch)
        if self.sh is not None:
            edge_sh = self.sh(r_hat)
        else:
            edge_sh = r_hat
        envelope = self.cutoff_env(r)
        edge_rbf = self.radial_basis(r) * envelope.unsqueeze(-1)
        return r, edge_sh, edge_rbf, envelope

    def compute_edge_features(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r, edge_sh, edge_rbf, _ = self._edge_features(positions, edge_index, edge_shift)
        return r, edge_sh, edge_rbf

    def compute_energy(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: Optional[torch.Tensor],
        n_graphs: int,
        strain: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Energy for an explicit graph (TorchScript-compatible)."""
        n_atoms = positions.shape[0]
        _, edge_sh, edge_rbf, envelope = self._edge_features(
            positions, edge_index, edge_shift, strain, batch
        )

        h_scalar = self.species_embedding(species)
        h = self.input_proj(h_scalar) * self.input_irreps_mask

        total_elec_energy: Optional[torch.Tensor] = None
        for block in self.blocks:
            h, h_scalar, elec_energy = block.forward_with_energy(
                h,
                h_scalar,
                edge_index,
                edge_sh,
                edge_rbf,
                batch,
                positions,
                n_atoms,
                envelope,
            )
            if elec_energy is not None:
                if total_elec_energy is None:
                    total_elec_energy = elec_energy
                else:
                    total_elec_energy = total_elec_energy + elec_energy

        e_atomic = self.energy_head(h_scalar).squeeze(-1)
        e_atomic = e_atomic + self.atomic_energies_ref[species]

        e_total = torch.zeros(n_graphs, device=positions.device, dtype=e_atomic.dtype)
        e_total = e_total.scatter_add(0, batch, e_atomic)

        if total_elec_energy is not None:
            e_total = e_total + total_elec_energy

        return {"energy": e_total, "atomic_energies": e_atomic, "node_features": h_scalar}

    @torch.jit.unused
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
        pbc: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if compute_forces and not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)

        n_graphs = int(batch.max().item()) + 1
        periodic = cell is not None and bool(normalize_pbc(pbc, n_graphs).any())
        if edge_shift is not None and bool((edge_shift != 0).any()):
            periodic = True
        if self.long_range_type == "electrostatic" and periodic:
            raise ValueError(
                "long_range_type='electrostatic' sums bare Coulomb pairs without periodic "
                "images and is only valid for non-periodic structures"
            )

        if edge_index is None:
            edge_index, edge_shift = self.build_neighbor_graph(
                positions,
                batch,
                self.local_cutoff,
                cell=cell,
                pbc=pbc,
            )

        strain = make_strain(compute_stress, n_graphs, positions, cell, pbc)
        results = self.compute_energy(
            species, positions, batch, edge_index, edge_shift, n_graphs, strain
        )
        results.update(
            conservative_outputs(
                energy=results["energy"],
                positions=positions,
                strain=strain,
                cell=cell,
                compute_forces=compute_forces,
                create_graph=self.training,
            )
        )
        return results


__all__ = ["UnifiedEquivariantMLIP"]
