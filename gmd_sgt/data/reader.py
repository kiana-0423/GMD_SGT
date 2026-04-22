"""
Data readers for common DFT output formats.

Supported formats
-----------------
  extended-XYZ (.extxyz)   : ASE standard; written by VASP, GPAW, CP2K, ORCA
  NumPy archive  (.npz)    : array-based format used by NequIP / MACE datasets

All readers return List[Dict[str, torch.Tensor]] compatible with AtomicDataset.

Required labels in the source files
------------------------------------
  energy  : total DFT energy  [eV]
  forces  : atomic forces     [eV/Å]   (key 'forces' or 'force')
  stress  : virial stress     [eV/Å³]  (optional; 3×3 or Voigt 6-vector)

REQUIRES: ase
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .validation import validate_structure_item


# ── extended-XYZ ─────────────────────────────────────────────────────────────

def read_extxyz(
    path: str,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: Optional[str] = "stress",
    index: str = ":",
) -> List[Dict[str, torch.Tensor]]:
    """
    Read an extended-XYZ file and return a list of structure dicts.

    Parameters
    ----------
    path        : path to .extxyz file
    energy_key  : key for energy in atoms.info  (default 'energy')
    forces_key  : key for forces in atoms.arrays  (default 'forces')
    stress_key  : key for stress in atoms.info  (None = skip stress)
    index       : ASE index selector  (':' = all frames)

    Returns
    -------
    List of dicts, each containing:
      species, positions, energy, forces, [stress], [cell], [pbc], n_atoms
    """
    try:
        from ase.io import read as ase_read
    except ImportError as e:
        raise ImportError("ase is required: pip install ase") from e

    frames = ase_read(path, index=index)
    if not isinstance(frames, list):
        frames = [frames]

    data_list = []
    for atoms in frames:
        item = _atoms_to_dict(atoms, energy_key, forces_key, stress_key)
        if item is not None:
            data_list.append(item)

    print(f"[reader] Loaded {len(data_list)} structures from {path}")
    return data_list


def _atoms_to_dict(
    atoms,
    energy_key: str,
    forces_key: str,
    stress_key: Optional[str],
) -> Optional[Dict[str, torch.Tensor]]:
    """Convert a single ASE Atoms object to a tensor dict. Returns None on error."""
    # Energy
    if energy_key in atoms.info:
        energy = float(atoms.info[energy_key])
    elif hasattr(atoms, "get_potential_energy"):
        try:
            energy = atoms.get_potential_energy()
        except Exception:
            print(f"[reader] Warning: no energy found, skipping structure")
            return None
    else:
        print(f"[reader] Warning: no energy key '{energy_key}', skipping")
        return None

    # Forces
    if forces_key in atoms.arrays:
        forces = np.array(atoms.arrays[forces_key], dtype=np.float32)
    elif "force" in atoms.arrays:
        forces = np.array(atoms.arrays["force"], dtype=np.float32)
    else:
        try:
            forces = atoms.get_forces().astype(np.float32)
        except Exception:
            print(f"[reader] Warning: no forces found, skipping structure")
            return None

    positions = np.array(atoms.get_positions(), dtype=np.float32)   # [N, 3]
    species = np.array(atoms.get_atomic_numbers(), dtype=np.int64)  # [N]
    n_atoms = len(species)

    item: Dict[str, torch.Tensor] = {
        "species":   torch.from_numpy(species),
        "positions": torch.from_numpy(positions),
        "energy":    torch.tensor([energy], dtype=torch.float32),
        "forces":    torch.from_numpy(forces),
        "n_atoms":   n_atoms,
    }

    # Cell + PBC
    cell = np.array(atoms.get_cell(), dtype=np.float32)             # [3, 3]
    pbc = atoms.get_pbc()
    if pbc.any():
        item["cell"] = torch.from_numpy(cell)
        item["pbc"] = torch.tensor(True)

    # Stress (optional)
    if stress_key is not None:
        if stress_key in atoms.info:
            s = np.array(atoms.info[stress_key], dtype=np.float32)
            # Accept Voigt (6,) or full (3,3) or flattened (9,)
            if s.shape == (6,):
                s = _voigt_to_matrix(s)
            elif s.shape == (9,):
                s = s.reshape(3, 3)
            item["stress"] = torch.from_numpy(s.reshape(3, 3))
        else:
            try:
                # ASE convention: stress in eV/Å³, Voigt order
                s = atoms.get_stress(voigt=False).astype(np.float32)   # [3,3]
                item["stress"] = torch.from_numpy(s)
            except Exception:
                pass  # stress is optional

    return validate_structure_item(item)


def _voigt_to_matrix(v: np.ndarray) -> np.ndarray:
    """Convert Voigt 6-vector [xx,yy,zz,yz,xz,xy] to 3×3 symmetric matrix."""
    xx, yy, zz, yz, xz, xy = v
    return np.array([
        [xx, xy, xz],
        [xy, yy, yz],
        [xz, yz, zz],
    ], dtype=np.float32)


# ── NumPy .npz ────────────────────────────────────────────────────────────────

def read_npz(path: str) -> List[Dict[str, torch.Tensor]]:
    """
    Read a .npz dataset file (NequIP / MACE / SchNetPack convention).

    Expected arrays in the .npz:
      R          : [n_frames, n_atoms, 3]  positions, Å
      Z          : [n_frames, n_atoms]     atomic numbers
      E          : [n_frames]              energies, eV
      F          : [n_frames, n_atoms, 3]  forces, eV/Å
      stress     : [n_frames, 3, 3]        optional
      cell       : [n_frames, 3, 3]        optional

    Also accepts the single-molecule convention where n_atoms varies per frame
    (stored as a ragged list of arrays).
    """
    data = np.load(path, allow_pickle=True)

    # Detect key names (handle common variants)
    pos_key    = _find_key(data, ["R", "pos", "positions"])
    z_key      = _find_key(data, ["Z", "z", "atomic_numbers", "species"])
    e_key      = _find_key(data, ["E", "energy", "energies"])
    f_key      = _find_key(data, ["F", "forces", "force"])

    if any(k is None for k in [pos_key, z_key, e_key, f_key]):
        missing = [n for n, k in zip(
            ["positions", "species", "energy", "forces"],
            [pos_key, z_key, e_key, f_key]
        ) if k is None]
        raise KeyError(f"NPZ missing required keys: {missing}. Found: {list(data.keys())}")

    positions_arr = data[pos_key]   # [N_frames, n_atoms, 3] or list
    species_arr   = data[z_key]
    energy_arr    = data[e_key]
    forces_arr    = data[f_key]

    stress_arr = data.get("stress", data.get("virial", None))
    cell_arr   = data.get("cell", None)

    n_frames = len(energy_arr)
    data_list = []

    for i in range(n_frames):
        pos = np.array(positions_arr[i], dtype=np.float32)
        Z   = np.array(species_arr[i],   dtype=np.int64)
        E   = float(energy_arr[i])
        F   = np.array(forces_arr[i],    dtype=np.float32)

        item: Dict[str, torch.Tensor] = {
            "species":   torch.from_numpy(Z),
            "positions": torch.from_numpy(pos),
            "energy":    torch.tensor([E], dtype=torch.float32),
            "forces":    torch.from_numpy(F),
            "n_atoms":   len(Z),
        }
        if cell_arr is not None:
            item["cell"] = torch.from_numpy(
                np.array(cell_arr[i], dtype=np.float32).reshape(3, 3)
            )
            item["pbc"] = torch.tensor(True)
        if stress_arr is not None:
            item["stress"] = torch.from_numpy(
                np.array(stress_arr[i], dtype=np.float32).reshape(3, 3)
            )
        data_list.append(validate_structure_item(item))

    print(f"[reader] Loaded {n_frames} structures from {path}")
    return data_list


def _find_key(data, candidates: list) -> Optional[str]:
    for k in candidates:
        if k in data:
            return k
    return None
