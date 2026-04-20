"""
Per-species energy reference calculation.

The MLIP model learns residual energies after subtracting a linear baseline:
    E_DFT(structure) ≈ Σ_i  a[Z_i]   (solved by linear regression on training set)

This baseline is passed to UnifiedEquivariantMLIP as `atomic_energies`.
Subtracting it makes training significantly more stable and faster.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .dataset import AtomicDataset


def compute_per_species_energy_shift(
    dataset: AtomicDataset,
    species_list: List[int],
) -> Dict[int, float]:
    """
    Fit per-species energy shift by linear regression:
        E = Σ_i  a[Z_i]  +  residual

    Minimises ||X a - E||² where
      X[n, k] = number of atoms of species k in structure n
      E[n]    = total energy of structure n

    Parameters
    ----------
    dataset      : AtomicDataset (training set only, not val/test)
    species_list : list of atomic numbers to fit  e.g. [1, 6, 8]

    Returns
    -------
    Dict mapping atomic_number → energy_shift (eV)
    Suitable for passing as `atomic_energies` to UnifiedEquivariantMLIP.

    Notes
    -----
    - If a species has very few structures, the fit may be poor.
      Consider using isolated-atom DFT energies as a fallback.
    - The residual energies should have zero mean and small std after subtraction.
    """
    species_list = sorted(species_list)
    k = len(species_list)
    sp_idx = {Z: i for i, Z in enumerate(species_list)}

    n_structs = len(dataset)
    X = np.zeros((n_structs, k), dtype=np.float64)
    E = np.zeros(n_structs, dtype=np.float64)

    for n, item in enumerate(dataset):
        Z_arr = item["species"].numpy()
        for Z in Z_arr:
            if Z in sp_idx:
                X[n, sp_idx[Z]] += 1
        E[n] = float(item["energy"].item())

    # Least-squares: a = (X^T X)^{-1} X^T E
    # Use numpy lstsq for numerical stability
    a, residuals, rank, sv = np.linalg.lstsq(X, E, rcond=None)

    result = {Z: float(a[i]) for i, Z in enumerate(species_list)}

    # Diagnostics
    E_pred = X @ a
    mae = np.abs(E_pred - E).mean()
    print(f"[statistics] Per-species energy shift fitted on {n_structs} structures")
    print(f"[statistics] Fit MAE: {mae*1000:.2f} meV/structure")
    for Z, val in result.items():
        print(f"  Z={Z:3d}: {val:.4f} eV")

    return result


def compute_dataset_statistics(
    dataset: AtomicDataset,
    atomic_energies: Dict[int, float],
) -> Dict[str, float]:
    """
    Compute mean and std of residual energies per atom and force components.
    Useful for verifying that the energy shift is working correctly,
    and for setting loss weights.

    Returns dict with keys:
      e_mean, e_std      : residual energy per atom mean/std  [eV/atom]
      f_mean, f_std      : force component mean/std  [eV/Å]
    """
    e_residuals, f_components = [], []

    for item in dataset:
        Z_arr = item["species"].numpy()
        E = float(item["energy"].item())
        E_ref = sum(atomic_energies.get(int(Z), 0.0) for Z in Z_arr)
        n = len(Z_arr)
        e_residuals.append((E - E_ref) / n)
        f_components.extend(item["forces"].numpy().ravel().tolist())

    e = np.array(e_residuals)
    f = np.array(f_components)
    return {
        "e_mean": float(e.mean()),
        "e_std":  float(e.std()),
        "f_mean": float(f.mean()),
        "f_std":  float(f.std()),
    }
