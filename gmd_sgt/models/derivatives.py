"""Conservative force / stress evaluation shared by all energy models."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .geometry import cell_volume, normalize_pbc


def make_strain(
    compute_stress: bool,
    n_graphs: int,
    like: torch.Tensor,
    cell: Optional[torch.Tensor] = None,
    pbc: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Zero strain tensor ``[n_graphs, 3, 3]`` to differentiate against, or None.

    Stress is only defined for periodic structures, so a cell with at least
    one periodic axis is required when ``compute_stress`` is set.
    """
    if not compute_stress:
        return None
    if cell is None or not bool(normalize_pbc(pbc, n_graphs).any()):
        raise ValueError(
            "compute_stress=True requires a periodic cell; stress is undefined "
            "for non-periodic structures"
        )
    stress_volumes(cell, n_graphs)
    return torch.zeros((n_graphs, 3, 3), dtype=like.dtype, device=like.device, requires_grad=True)


def stress_volumes(cell: Optional[torch.Tensor], n_graphs: int) -> torch.Tensor:
    """Per-graph cell volumes used to normalise the stress."""
    if cell is None:
        raise ValueError(
            "compute_stress=True requires a periodic cell; stress is undefined "
            "for non-periodic structures"
        )
    cells = cell.unsqueeze(0).expand(n_graphs, 3, 3) if cell.dim() == 2 else cell
    if cells.shape[0] != n_graphs:
        raise ValueError(f"cell has {cells.shape[0]} graphs but batch encodes {n_graphs}")
    volumes = cell_volume(cells.detach())
    if bool((volumes <= 1e-10).any()):
        raise ValueError("compute_stress=True requires cells with non-zero volume")
    return volumes


def conservative_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    strain: Optional[torch.Tensor],
    cell: Optional[torch.Tensor],
    compute_forces: bool,
    create_graph: bool,
) -> Dict[str, torch.Tensor]:
    """Return ``forces = -dE/dr`` and/or ``stress = (1/V) dE/d(strain)``.

    Both derivatives come from one ``autograd.grad`` call. The graph is kept
    alive exactly when ``create_graph`` is set (training), so a later
    ``loss.backward()`` through energy *and* forces works, while inference
    frees it immediately.
    """
    inputs: List[torch.Tensor] = []
    if compute_forces:
        inputs.append(positions)
    if strain is not None:
        volumes = stress_volumes(cell, strain.shape[0])
        inputs.append(strain)
    if not inputs:
        return {}

    if energy.requires_grad:
        grads = torch.autograd.grad(
            outputs=[energy.sum()],
            inputs=inputs,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )
    else:
        grads = [None] * len(inputs)

    results: Dict[str, torch.Tensor] = {}
    cursor = 0
    if compute_forces:
        grad = grads[cursor]
        results["forces"] = -grad if grad is not None else torch.zeros_like(positions)
        cursor += 1
    if strain is not None:
        grad = grads[cursor]
        if grad is None:
            grad = torch.zeros_like(strain)
        results["stress"] = grad / volumes.to(grad.dtype).view(-1, 1, 1)
    return results


__all__ = ["conservative_outputs", "make_strain", "stress_volumes"]
