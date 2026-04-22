"""Validation helpers for atomic-structure datasets."""

from __future__ import annotations

from typing import Dict

import torch


def validate_structure_item(item: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Validate one structure dictionary and return it unchanged on success.

    The checks are intentionally minimal and format-oriented:
      - energy exists and is scalar-like
      - species is 1D and positions is [N, 3]
      - forces, if present, is [N, 3]
      - cell, if present, is [3, 3]
      - pbc/cell do not contradict each other
    """
    required = {"species", "positions", "energy"}
    missing = required.difference(item)
    if missing:
        raise KeyError(f"Structure item is missing required keys: {sorted(missing)}")

    species = item["species"]
    positions = item["positions"]
    energy = item["energy"]

    if species.ndim != 1:
        raise ValueError(f"species must have shape [N], got {tuple(species.shape)}")
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(
            f"positions must have shape [N, 3], got {tuple(positions.shape)}"
        )
    if positions.shape[0] != species.shape[0]:
        raise ValueError(
            "positions and species must describe the same number of atoms"
        )
    if energy.numel() != 1:
        raise ValueError(f"energy must be scalar-like, got shape {tuple(energy.shape)}")
    if species.numel() == 0:
        raise ValueError("Empty structures are not supported")
    if not torch.is_floating_point(positions):
        raise TypeError("positions must be a floating-point tensor")

    forces = item.get("forces")
    if forces is not None:
        if forces.ndim != 2 or forces.shape != positions.shape:
            raise ValueError(
                f"forces must have shape {tuple(positions.shape)}, got {tuple(forces.shape)}"
            )

    cell = item.get("cell")
    pbc = item.get("pbc")

    pbc_enabled = False
    if pbc is not None:
        if isinstance(pbc, torch.Tensor):
            pbc_enabled = bool(pbc.detach().cpu().to(dtype=torch.bool).any().item())
        else:
            try:
                # Support sequence-like pbc values such as [True, False, True].
                pbc_enabled = bool(any(bool(v) for v in pbc))
            except TypeError:
                pbc_enabled = bool(pbc)

    if cell is not None and tuple(cell.shape) != (3, 3):
        raise ValueError(f"cell must have shape [3, 3], got {tuple(cell.shape)}")
    if pbc_enabled and cell is None:
        raise ValueError("pbc=True requires a cell tensor")
    if cell is not None and pbc is None:
        raise ValueError("cell is present but pbc flag is missing")

    n_atoms = item.get("n_atoms")
    if n_atoms is not None and int(n_atoms) != positions.shape[0]:
        raise ValueError(
            f"n_atoms={int(n_atoms)} does not match structure size {positions.shape[0]}"
        )

    return item
