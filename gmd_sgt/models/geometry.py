"""Shared geometry utilities for local MLIP models."""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

from .dependencies import CLUSTER_AVAILABLE, E3NN_AVAILABLE, o3, radius_graph

PBCLike = Union[None, bool, Sequence[bool], torch.Tensor]

# Largest spherical-harmonic order implemented without e3nn.
MAX_CARTESIAN_L = 2


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Sum features into segments defined by ``index`` (TorchScript friendly).

    Deliberately branch-free: ``index_add`` handles empty inputs, and a
    data-dependent early return for ``E == 0`` makes TorchScript's profiling
    executor produce wrong gradients on later non-empty calls.
    """
    out_shape = [dim_size] + list(src.shape[1:])
    return src.new_zeros(out_shape).index_add(0, index, src)


# ── Periodic boundary conditions ─────────────────────────────────────────────


def normalize_pbc(
    pbc: PBCLike,
    n_graphs: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return per-graph, per-axis PBC flags with shape ``[n_graphs, 3]``.

    Accepts ``None`` (fully periodic, the historical meaning of passing a
    cell), a scalar boolean, a length-3 sequence, or an ``[n_graphs, 3]``
    tensor.
    """
    if pbc is None:
        flags = torch.ones((n_graphs, 3), dtype=torch.bool)
    else:
        flags = torch.as_tensor(pbc).detach().cpu().to(torch.bool)
        if flags.dim() == 0 or flags.numel() == 1:
            flags = flags.reshape(1, 1).expand(n_graphs, 3)
        elif flags.dim() == 1:
            if flags.shape[0] != 3:
                raise ValueError(f"pbc must have 3 entries, got shape {tuple(flags.shape)}")
            flags = flags.reshape(1, 3).expand(n_graphs, 3)
        elif flags.dim() == 2 and flags.shape[1] == 3:
            if flags.shape[0] != n_graphs:
                raise ValueError(
                    f"pbc has {flags.shape[0]} rows but the batch encodes {n_graphs} graphs"
                )
        else:
            raise ValueError(
                f"pbc must be a bool, a length-3 sequence or [n_graphs, 3]; "
                f"got shape {tuple(flags.shape)}"
            )
        flags = flags.clone()
    if device is not None:
        flags = flags.to(device)
    return flags


def cell_volume(cell: torch.Tensor) -> torch.Tensor:
    """Absolute volume of one ``[3, 3]`` or batched ``[G, 3, 3]`` cell."""
    return torch.linalg.det(cell).abs()


# ── Neighbor graph construction ──────────────────────────────────────────────


def _dense_radius_graph(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    """All ordered pairs ``(src, dst)`` with ``0 < |r_dst - r_src| < cutoff``."""
    pos = positions.detach()
    dist = torch.cdist(pos, pos, compute_mode="donot_use_mm_for_euclid_dist")
    mask = (dist < cutoff) & ~torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
    src, dst = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)


def _nonperiodic_graph(
    positions: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    if CLUSTER_AVAILABLE:
        # torch_cluster truncates to 32 neighbours by default; never truncate.
        return radius_graph(
            positions.detach(),
            r=cutoff,
            batch=batch,
            loop=False,
            max_num_neighbors=max(int(positions.shape[0]), 1),
        )

    pos = positions.detach()
    dist = torch.cdist(pos, pos, compute_mode="donot_use_mm_for_euclid_dist")
    same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
    eye = torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
    src, dst = ((dist < cutoff) & same_graph & ~eye).nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)


def build_neighbor_graph(
    positions: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    cell: Optional[torch.Tensor] = None,
    pbc: PBCLike = None,
    pbc_builder: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build a neighbor graph for batched atomistic structures.

    Parameters
    ----------
    cell:
        ``None`` for non-periodic input, a single ``[3, 3]`` cell shared by all
        graphs, or one ``[n_graphs, 3, 3]`` cell per graph.
    pbc:
        Per-axis periodicity (see :func:`normalize_pbc`). Ignored when ``cell``
        is ``None``. Graphs whose flags are all ``False`` are treated as
        non-periodic.
    pbc_builder:
        Override for the per-graph periodic builder; must accept
        ``(positions, cell, cutoff, pbc)``.

    Returns
    -------
    edge_index : ``[2, E]`` with ``r_ij = pos[edge_index[1]] - pos[edge_index[0]] + shift``
    edge_shift : ``[E, 3]`` Cartesian shifts, or ``None`` for non-periodic input
    """
    if cell is None:
        return _nonperiodic_graph(positions, batch, cutoff), None

    builder = pbc_builder or build_neighbor_graph_pbc
    n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if cell.dim() == 2:
        cells = cell.unsqueeze(0).expand(n_graphs, 3, 3)
    elif cell.dim() == 3:
        if cell.shape[0] != n_graphs:
            raise ValueError(
                f"cell has {cell.shape[0]} graphs but batch encodes {n_graphs} graphs"
            )
        cells = cell
    else:
        raise ValueError(f"cell must be [3, 3] or [n_graphs, 3, 3], got {tuple(cell.shape)}")
    pbc_flags = normalize_pbc(pbc, n_graphs)

    edge_indices: List[torch.Tensor] = []
    edge_shifts: List[torch.Tensor] = []
    offset = 0
    for graph_idx in range(n_graphs):
        mask = batch == graph_idx
        n_in_graph = int(mask.sum().item())
        if n_in_graph == 0:
            continue
        pos_g = positions[mask]
        if bool(pbc_flags[graph_idx].any()):
            edge_index_g, edge_shift_g = builder(
                pos_g, cells[graph_idx], cutoff, pbc_flags[graph_idx]
            )
        else:
            edge_index_g = _dense_radius_graph(pos_g, cutoff)
            edge_shift_g = positions.new_zeros((edge_index_g.shape[1], 3))
        edge_indices.append(edge_index_g.to(positions.device) + offset)
        edge_shifts.append(edge_shift_g.to(device=positions.device, dtype=positions.dtype))
        offset += n_in_graph

    if not edge_indices:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=positions.device),
            torch.zeros((0, 3), dtype=positions.dtype, device=positions.device),
        )
    return torch.cat(edge_indices, dim=1), torch.cat(edge_shifts, dim=0)


def build_neighbor_graph_pbc(
    positions: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    pbc: PBCLike = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a periodic neighbor graph for one structure with ASE.

    Uses :func:`ase.neighborlist.primitive_neighbor_list`, which supports
    per-axis periodicity, unwrapped positions and cells smaller than the
    cutoff (multiple periodic images).
    """
    try:
        import numpy as np
        from ase.neighborlist import primitive_neighbor_list
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError("ASE is required for PBC neighbor graph construction") from exc

    pbc_np = normalize_pbc(pbc, 1)[0].numpy()
    pos_np = positions.detach().cpu().to(torch.float64).numpy()
    cell_np = cell.detach().cpu().to(torch.float64).numpy()
    src, dst, shifts = primitive_neighbor_list(
        "ijS",
        pbc_np,
        cell_np,
        pos_np,
        float(cutoff),
        self_interaction=False,
    )
    edge_index = torch.as_tensor(
        np.stack([src, dst], axis=0), dtype=torch.long, device=positions.device
    ).reshape(2, -1)
    edge_shift = torch.as_tensor(
        shifts.astype(np.float64) @ cell_np, dtype=positions.dtype, device=positions.device
    ).reshape(-1, 3)
    return edge_index, edge_shift


# ── Edge geometry ────────────────────────────────────────────────────────────


def apply_strain(
    rel_vec: torch.Tensor,
    strain: Optional[torch.Tensor],
    edge_batch: torch.Tensor,
) -> torch.Tensor:
    """Apply a homogeneous per-graph deformation ``r -> r (I + sym(strain))``.

    Used to obtain the stress as ``(1/V) dE/d(strain)`` at ``strain = 0``.
    """
    if strain is None:
        return rel_vec
    sym = 0.5 * (strain + strain.transpose(1, 2))
    return rel_vec + torch.bmm(rel_vec.unsqueeze(1), sym[edge_batch]).squeeze(1)


def compute_edge_geometry(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_shift: Optional[torch.Tensor] = None,
    strain: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return relative vectors, distances, and safe unit vectors."""
    src, dst = edge_index[0], edge_index[1]
    rel_vec = positions[dst] - positions[src]
    if edge_shift is not None:
        rel_vec = rel_vec + edge_shift
    if strain is not None:
        if batch is None:
            raise ValueError("batch is required when strain is given")
        rel_vec = apply_strain(rel_vec, strain, batch[src])
    distances = torch.linalg.norm(rel_vec, dim=-1)
    unit_vec = rel_vec / distances.unsqueeze(-1).clamp_min(1e-8)
    return rel_vec, distances, unit_vec


def cartesian_spherical_harmonics(unit_vec: torch.Tensor, l_max: int) -> torch.Tensor:
    """Real spherical harmonics for ``l <= 2`` without e3nn.

    The layout, signs and ``"component"`` normalisation (``|Y_l|^2 = 2l+1``)
    match ``e3nn.o3.spherical_harmonics(..., normalize=True,
    normalization="component")`` exactly, so the blocks are true irreducible
    representations and squared norms per ``l`` are rotation invariant.
    Returns ``[E, (l_max + 1) ** 2]``.
    """
    if l_max < 0 or l_max > 2:  # MAX_CARTESIAN_L (literal for TorchScript)
        raise ValueError("cartesian spherical harmonics support 0 <= l_max <= 2")
    x = unit_vec[:, 0]
    y = unit_vec[:, 1]
    z = unit_vec[:, 2]
    parts = [torch.ones_like(x).unsqueeze(-1)]
    if l_max >= 1:
        parts.append(math.sqrt(3.0) * unit_vec)
    if l_max >= 2:
        s15 = math.sqrt(15.0)
        parts.append(
            torch.stack(
                [
                    s15 * x * z,
                    s15 * x * y,
                    math.sqrt(5.0) * (y * y - 0.5 * (x * x + z * z)),
                    s15 * y * z,
                    0.5 * s15 * (z * z - x * x),
                ],
                dim=-1,
            )
        )
    return torch.cat(parts, dim=-1)


def split_spherical_harmonics(sh: torch.Tensor, l_max: int) -> List[torch.Tensor]:
    """Split concatenated ``[E, (l_max+1)^2]`` harmonics into per-l blocks."""
    bases: List[torch.Tensor] = []
    cursor = 0
    for ell in range(l_max + 1):
        dim = 2 * ell + 1
        bases.append(sh[:, cursor : cursor + dim])
        cursor += dim
    return bases


def directional_basis(unit_vec: torch.Tensor, l_max: int) -> list[torch.Tensor]:
    """Return per-l real spherical-harmonic blocks ``[E, 2l+1]`` for ``l <= l_max``.

    Orders up to 2 are computed in closed form (identical to e3nn); higher
    orders require e3nn.
    """
    if l_max <= MAX_CARTESIAN_L:
        return split_spherical_harmonics(cartesian_spherical_harmonics(unit_vec, l_max), l_max)
    if not E3NN_AVAILABLE:
        raise ImportError(
            f"l_max={l_max} requires e3nn (only l_max <= {MAX_CARTESIAN_L} is available "
            "without it). Install with: pip install 'gmd-sgt[e3nn]'"
        )
    sh = o3.spherical_harmonics(
        list(range(l_max + 1)),
        unit_vec,
        normalize=True,
        normalization="component",
    )
    return split_spherical_harmonics(sh, l_max)


__all__ = [
    "MAX_CARTESIAN_L",
    "apply_strain",
    "build_neighbor_graph",
    "build_neighbor_graph_pbc",
    "cartesian_spherical_harmonics",
    "cell_volume",
    "compute_edge_geometry",
    "directional_basis",
    "normalize_pbc",
    "scatter_sum",
    "split_spherical_harmonics",
]
