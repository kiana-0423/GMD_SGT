"""AtomicDataset and collate_fn."""

from __future__ import annotations

from typing import Dict, List

import torch


class AtomicDataset(torch.utils.data.Dataset):
    """
    Dataset of atomic structures for MLIP training.

    Each item is a dict with keys:
      species   : [n_atoms]     int64  atomic numbers Z
      positions : [n_atoms, 3]  float32  Cartesian coordinates, Å
      energy    : [1]           float32  total energy, eV
      forces    : [n_atoms, 3]  float32  atomic forces, eV/Å
      stress    : [3, 3]        float32  virial stress, eV/Å³  (optional)
      cell      : [3, 3]        float32  unit cell, Å  (optional, for PBC)
      pbc       : bool          periodic boundary conditions flag  (optional)
      n_atoms   : int           number of atoms (for convenience)

    Build via:
      AtomicDataset(data_list)
      AtomicDataset.from_extxyz(path)   -- reads extended-XYZ via ASE
      AtomicDataset.from_npz(path)      -- reads .npz array format
    """

    def __init__(self, data_list: List[Dict[str, torch.Tensor]]):
        self.data = data_list

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]

    @classmethod
    def from_extxyz(cls, path: str) -> "AtomicDataset":
        from .reader import read_extxyz
        return cls(read_extxyz(path))

    @classmethod
    def from_npz(cls, path: str) -> "AtomicDataset":
        from .reader import read_npz
        return cls(read_npz(path))


def collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Collate a list of single-structure dicts into a batched graph dict.
    Concatenates atom arrays and assigns graph-id (batch index) per atom.
    """
    all_species, all_positions, all_forces = [], [], []
    all_energy, all_batch, all_n_atoms = [], [], []
    has_stress = "stress" in batch[0]
    has_cell = "cell" in batch[0]
    all_stress, all_cell = [], []

    for graph_id, item in enumerate(batch):
        n = item["species"].shape[0]
        all_species.append(item["species"])
        all_positions.append(item["positions"])
        all_forces.append(item["forces"])
        all_energy.append(item["energy"].reshape(1))
        all_batch.append(torch.full((n,), graph_id, dtype=torch.long))
        all_n_atoms.append(n)
        if has_stress:
            all_stress.append(item["stress"].reshape(1, 3, 3))
        if has_cell:
            all_cell.append(item["cell"].reshape(1, 3, 3))

    out = {
        "species":   torch.cat(all_species,   dim=0),
        "positions": torch.cat(all_positions, dim=0),
        "forces":    torch.cat(all_forces,    dim=0),
        "energy":    torch.cat(all_energy,    dim=0),
        "batch":     torch.cat(all_batch,     dim=0),
        "n_atoms":   torch.tensor(all_n_atoms, dtype=torch.long),
    }
    if has_stress:
        out["stress"] = torch.cat(all_stress, dim=0)   # [n_graphs, 3, 3]
    if has_cell:
        out["cell"] = torch.cat(all_cell, dim=0)        # [n_graphs, 3, 3]
    return out
