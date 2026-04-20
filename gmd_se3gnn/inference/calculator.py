"""MLIPCalculator: numpy-in / numpy-out inference interface.

This is the Python-side calculator used for:
  - Standalone validation / benchmarking against DFT
  - ASE calculator integration (optional)
  - Debugging before GMD C++ integration

For production MD, GMD calls the TorchScript model (model.pt) directly via
libtorch; this file is NOT in the hot path during a real MD run.

Usage
-----
  from gmd_se3gnn.inference import MLIPCalculator

  calc = MLIPCalculator.from_checkpoint("outputs/run/ckpt_best.pt", device="cpu")
  print("cutoff:", calc.cutoff)

  result = calc.compute(
      positions=np.array([[0,0,0],[1.5,0,0]], dtype=np.float32),
      species=np.array([6, 1]),      # C, H
  )
  # result["energy"]  → float (eV)
  # result["forces"]  → np.ndarray [N,3] (eV/Å)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from gmd_se3gnn.model import UnifiedEquivariantMLIP


class MLIPCalculator:
    """Numpy-interface calculator wrapping UnifiedEquivariantMLIP.

    Parameters
    ----------
    model:
        Loaded, eval-mode model instance.
    device:
        Torch device string.
    """

    def __init__(self, model: UnifiedEquivariantMLIP, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "MLIPCalculator":
        """Load a checkpoint saved by Trainer.save_checkpoint().

        The checkpoint must contain a 'model_config' key (guaranteed by
        Trainer in gmd_se3gnn.training.trainer).
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model_config" not in ckpt:
            raise KeyError(
                f"Checkpoint {path!r} is missing 'model_config'. "
                "Re-train with the current Trainer to produce a valid checkpoint."
            )
        model = UnifiedEquivariantMLIP(**ckpt["model_config"])
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, device=device)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def cutoff(self) -> float:
        """Short-range cutoff radius (Å). Queried by GMD to build neighbor list."""
        return float(self.model.local_cutoff)

    @property
    def lr_cutoff(self) -> float:
        """Long-range cutoff radius (Å)."""
        return float(self.model.lr_cutoff)

    # ── Main interface ───────────────────────────────────────────────────────

    def compute(
        self,
        positions: np.ndarray,
        species: np.ndarray,
        cell: Optional[np.ndarray] = None,
        pbc: bool = False,
        edge_index: Optional[np.ndarray] = None,
        edge_shift: Optional[np.ndarray] = None,
        compute_stress: bool = False,
    ) -> dict:
        """Compute energy and forces (optionally stress).

        Parameters
        ----------
        positions : np.ndarray [N, 3]
            Atomic positions in Å (Cartesian).
        species : np.ndarray [N]
            Atomic numbers (int).
        cell : np.ndarray [3, 3] or None
            Unit cell in Å. None = non-periodic system.
        pbc : bool
            Whether to apply periodic boundary conditions.
            Ignored when edge_index is provided explicitly.
        edge_index : np.ndarray [2, E] or None
            Pre-built neighbor list from external program (e.g. GMD).
            If None and cell is not None, built internally via ASE.
        edge_shift : np.ndarray [E, 3] or None
            Cartesian PBC shift vectors (Å) corresponding to edge_index.
            Required when edge_index is provided.
        compute_stress : bool
            Whether to compute the virial stress tensor [3, 3] eV/Å³.

        Returns
        -------
        dict with keys:
            "energy"  : float          (eV)
            "forces"  : np.ndarray [N,3] (eV/Å)
            "stress"  : np.ndarray [3,3] (eV/Å³) — only if compute_stress=True
        """
        dev = self.device
        dtype = torch.float32

        pos_t = torch.tensor(positions, dtype=dtype, device=dev)
        spc_t = torch.tensor(species, dtype=torch.long, device=dev)
        batch_t = torch.zeros(len(species), dtype=torch.long, device=dev)

        ei_t: Optional[torch.Tensor] = None
        es_t: Optional[torch.Tensor] = None
        cell_t: Optional[torch.Tensor] = None

        if edge_index is not None:
            if edge_shift is None:
                raise ValueError(
                    "edge_shift must be provided when edge_index is given. "
                    "For non-PBC neighbor lists pass "
                    "edge_shift=np.zeros((E, 3), dtype=np.float32)."
                )
            ei_t = torch.tensor(edge_index, dtype=torch.long, device=dev)
            es_t = torch.tensor(edge_shift, dtype=dtype, device=dev)
        elif cell is not None and pbc:
            cell_t = torch.tensor(cell, dtype=dtype, device=dev)

        with torch.set_grad_enabled(True):
            out = self.model(
                species=spc_t,
                positions=pos_t,
                batch=batch_t,
                edge_index=ei_t,
                edge_shift=es_t,
                cell=cell_t,
                compute_forces=True,
                compute_stress=compute_stress,
            )

        result = {
            "energy": float(out["energy"].sum().item()),
            "forces": out["forces"].detach().cpu().numpy().astype(np.float64),
        }
        if compute_stress and out.get("stress") is not None:
            result["stress"] = out["stress"].detach().cpu().numpy().astype(np.float64)

        return result

    # ── Optional ASE calculator mixin ────────────────────────────────────────

    def get_ase_calculator(self):
        """Return an ASE-compatible Calculator wrapping this MLIPCalculator.

        Requires ASE to be installed. Useful for structure optimisation,
        phonon calculations, etc. without involving GMD.

        Example
        -------
          from ase.io import read
          atoms = read("structure.extxyz")
          atoms.calc = calc.get_ase_calculator()
          print(atoms.get_potential_energy())
        """
        try:
            from ase.calculators.calculator import Calculator, all_changes
        except ImportError as exc:
            raise ImportError("ASE is required for get_ase_calculator()") from exc

        mlip_calc = self  # closure

        class _ASECalc(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=("energy", "forces"),
                          system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                pos = atoms.get_positions().astype(np.float32)
                spc = atoms.get_atomic_numbers()
                cell = atoms.get_cell().array.astype(np.float32) \
                    if any(atoms.get_pbc()) else None
                pbc = bool(any(atoms.get_pbc()))
                res = mlip_calc.compute(pos, spc, cell=cell, pbc=pbc)
                self.results["energy"] = res["energy"]
                self.results["forces"] = res["forces"]

        return _ASECalc()
