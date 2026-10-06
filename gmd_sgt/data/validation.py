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
        # Accept a scalar flag or per-axis flags such as [True, True, False].
        pbc_tensor = torch.as_tensor(pbc).detach().cpu().to(dtype=torch.bool)
        if pbc_tensor.numel() not in (1, 3):
            raise ValueError(
                f"pbc must be a bool or have 3 entries, got shape {tuple(pbc_tensor.shape)}"
            )
        pbc_enabled = bool(pbc_tensor.any().item())

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


def check_stress_labels(dataset, w_stress: float) -> None:
    """Reject stress training when labels or periodic cells are missing."""
    if w_stress <= 0:
        return
    for idx, item in enumerate(dataset):
        if "stress" not in item:
            raise ValueError(
                f"w_stress={w_stress} > 0 but structure {idx} has no stress label; "
                "set w_stress=0 or provide stress for every structure"
            )
        pbc = item.get("pbc")
        if "cell" not in item or pbc is None or not bool(torch.as_tensor(pbc).any()):
            raise ValueError(
                f"w_stress={w_stress} > 0 but structure {idx} is not periodic; "
                "stress is only defined for structures with a periodic cell"
            )
