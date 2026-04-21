"""
Backward-compatible re-exports for gmd_sgt.train.

All logic has been split into dedicated modules:
  - Loss function  →  gmd_sgt.training.loss.EnergyForceLoss
  - Trainer        →  gmd_sgt.training.trainer.Trainer
  - Dataset        →  gmd_sgt.data.dataset.AtomicDataset
  - collate_fn     →  gmd_sgt.data.dataset.collate_fn
  - Smoke test     →  scripts/smoke_test.py
  - CLI entry      →  scripts/train_cli.py

Importing from this file still works for backward compatibility.
New code should import directly from the submodules above.
"""

from gmd_sgt.training.loss import EnergyForceLoss
from gmd_sgt.training.trainer import Trainer, EarlyStopping
from gmd_sgt.data.dataset import AtomicDataset, collate_fn
from gmd_sgt.model import UnifiedEquivariantMLIP

__all__ = [
    "EnergyForceLoss",
    "Trainer",
    "EarlyStopping",
    "AtomicDataset",
    "collate_fn",
    "UnifiedEquivariantMLIP",
]
