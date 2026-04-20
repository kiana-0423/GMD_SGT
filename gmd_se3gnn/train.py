"""
Backward-compatible re-exports for gmd_se3gnn.train.

All logic has been split into dedicated modules:
  - Loss function  →  gmd_se3gnn.training.loss.EnergyForceLoss
  - Trainer        →  gmd_se3gnn.training.trainer.Trainer
  - Dataset        →  gmd_se3gnn.data.dataset.AtomicDataset
  - collate_fn     →  gmd_se3gnn.data.dataset.collate_fn
  - Smoke test     →  scripts/smoke_test.py
  - CLI entry      →  scripts/train_cli.py

Importing from this file still works for backward compatibility.
New code should import directly from the submodules above.
"""

from gmd_se3gnn.training.loss import EnergyForceLoss
from gmd_se3gnn.training.trainer import Trainer, EarlyStopping
from gmd_se3gnn.data.dataset import AtomicDataset, collate_fn
from gmd_se3gnn.model import UnifiedEquivariantMLIP

__all__ = [
    "EnergyForceLoss",
    "Trainer",
    "EarlyStopping",
    "AtomicDataset",
    "collate_fn",
    "UnifiedEquivariantMLIP",
]
