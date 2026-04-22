"""Shared geometry utilities for local MLIP models."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .dependencies import CLUSTER_AVAILABLE, E3NN_AVAILABLE, o3, radius_graph


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Sum features into segments defined by ``index``."""
    out_shape = (dim_size,) + tuple(src.shape[1:])
    out = src.new_zeros(out_shape)
    if src.numel() == 0:
        return out
    out.index_add_(0, index, src)
    return out


def build_neighbor_graph(
    positions: torch.Tensor,
    batch: torch.Tensor,
    cutoff: float,
    cell: Optional[torch.Tensor] = None,
    pbc: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build a neighbor graph for batched atomistic structures."""
    del pbc

    if cell is None:
        if CLUSTER_AVAILABLE:
            edge_index = radius_graph(positions, r=cutoff, batch=batch, loop=False)
            return edge_index, None

        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist = diff.norm(dim=-1)
        same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
        edge_mask = (dist < cutoff) & (dist > 0.0) & same_graph
        src, dst = edge_mask.nonzero(as_tuple=True)
        return torch.stack([src, dst], dim=0), None

    if cell.dim() == 2:
        return build_neighbor_graph_pbc(positions, cell, cutoff)

    n_graphs = cell.shape[0]
    expected_graphs = int(batch.max().item()) + 1
    if n_graphs != expected_graphs:
        raise ValueError(
            f"cell has {n_graphs} graphs but batch encodes {expected_graphs} graphs"
        )

    edge_indices = []
    edge_shifts = []
    offset = 0
    for graph_idx in range(n_graphs):
        mask = batch == graph_idx
        if not mask.any():
            continue
        edge_index_g, edge_shift_g = build_neighbor_graph_pbc(
            positions[mask],
            cell[graph_idx],
            cutoff,
        )
        edge_indices.append(edge_index_g + offset)
        edge_shifts.append(edge_shift_g)
        offset += int(mask.sum().item())

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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a PBC neighbor graph using ASE."""
    try:
        import numpy as np
        from ase.neighborlist import neighbor_list as ase_neighbor_list
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "ASE is required for PBC neighbor graph construction"
        ) from exc

    device = positions.device
    dtype = positions.dtype
    pos_np = positions.detach().cpu().numpy()
    cell_np = cell.detach().cpu().numpy()
    src, dst, shifts = ase_neighbor_list(
        "ijS",
        pbc=[True, True, True],
        cell=cell_np,
        positions=pos_np,
        cutoff=cutoff,
    )
    edge_index = torch.tensor(
        np.stack([src, dst], axis=0),
        dtype=torch.long,
        device=device,
    )
    edge_shift = torch.tensor(shifts @ cell_np, dtype=dtype, device=device)
    return edge_index, edge_shift


def compute_edge_geometry(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_shift: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return relative vectors, distances, and safe unit vectors."""
    src, dst = edge_index[0], edge_index[1]
    rel_vec = positions[dst] - positions[src]
    if edge_shift is not None:
        rel_vec = rel_vec + edge_shift
    distances = rel_vec.norm(dim=-1)
    unit_vec = rel_vec / distances.unsqueeze(-1).clamp_min(1e-8)
    return rel_vec, distances, unit_vec


def directional_basis(unit_vec: torch.Tensor, l_max: int) -> list[torch.Tensor]:
    """Return per-l directional bases used by the lightweight backbone.

    With ``e3nn`` installed this uses spherical harmonics blocks. Without it,
    the fallback uses low-order cartesian directional tensors that preserve the
    same input/output interface.
    """
    n_edges = unit_vec.shape[0]
    scalar = torch.ones((n_edges, 1), dtype=unit_vec.dtype, device=unit_vec.device)

    if n_edges == 0:
        bases = [scalar]
        if l_max >= 1:
            bases.append(unit_vec.new_zeros((0, 3)))
        if l_max >= 2:
            bases.append(unit_vec.new_zeros((0, 5)))
        return bases

    if E3NN_AVAILABLE:
        sh = o3.spherical_harmonics(
            list(range(l_max + 1)),
            unit_vec,
            normalize=True,
            normalization="component",
        )
        bases = []
        cursor = 0
        for ell in range(l_max + 1):
            dim = 2 * ell + 1
            bases.append(sh[:, cursor : cursor + dim])
            cursor += dim
        return bases

    bases = [scalar]
    if l_max >= 1:
        bases.append(unit_vec)
    if l_max >= 2:
        x, y, z = unit_vec.unbind(dim=-1)
        bases.append(
            torch.stack(
                [
                    x * x - y * y,
                    2.0 * x * y,
                    2.0 * x * z,
                    2.0 * y * z,
                    3.0 * z * z - 1.0,
                ],
                dim=-1,
            )
        )
    return bases
