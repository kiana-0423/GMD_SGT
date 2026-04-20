"""
Command-line training entry point.

Usage
-----
  python scripts/train_cli.py --config configs/default.yaml
  python scripts/train_cli.py --config configs/water.yaml --device cuda
  python scripts/train_cli.py --resume outputs/run/ckpt_best.pt --config configs/water.yaml

Config file format: YAML with sections model / train / data
See configs/default.yaml for all options.

REQUIRES: pyyaml  (pip install pyyaml)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required. Run: pip install pyyaml")
    sys.exit(1)

from gmd_se3gnn.model import UnifiedEquivariantMLIP
from gmd_se3gnn.data import (
    AtomicDataset,
    collate_fn,
    compute_per_species_energy_shift,
    split_dataset,
)
from gmd_se3gnn.training import EnergyForceLoss, Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train GMD-SE3GNN MLIP")
    parser.add_argument("--config",  required=True, help="Path to YAML config file")
    parser.add_argument("--device",  default=None,  help="Override device (cpu/cuda/mps)")
    parser.add_argument("--resume",  default=None,  help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    m_cfg   = cfg.get("model", {})
    t_cfg   = cfg.get("train", {})
    d_cfg   = cfg.get("data",  {})

    device = args.device or t_cfg.get("device", "cpu")
    output_dir = d_cfg.get("output_dir", "outputs/run")

    # ── Load dataset ──────────────────────────────────────────────────────────
    train_file = d_cfg.get("train_file")
    if train_file is None:
        print("ERROR: data.train_file must be set in the config.")
        sys.exit(1)

    print(f"Loading data from: {train_file}")
    if train_file.endswith(".npz"):
        dataset = AtomicDataset.from_npz(train_file)
    else:
        dataset = AtomicDataset.from_extxyz(train_file)

    val_fraction  = float(t_cfg.get("val_fraction",  0.10))
    test_fraction = float(t_cfg.get("test_fraction", 0.05))
    train_set, val_set, test_set = split_dataset(
        dataset, val_fraction=val_fraction, test_fraction=test_fraction
    )

    # ── Per-species energy shift ──────────────────────────────────────────────
    all_species = sorted(set(
        int(Z) for item in train_set for Z in item["species"].tolist()
    ))
    atomic_energies = compute_per_species_energy_shift(train_set, all_species)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    batch_size = int(t_cfg.get("batch_size", 4))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_config = {
        "n_species":        int(m_cfg.get("n_species",        100)),
        "n_blocks":         int(m_cfg.get("n_blocks",         4)),
        "scalar_dim":       int(m_cfg.get("scalar_dim",       128)),
        "irreps":           str(m_cfg.get("irreps",           "128x0e + 64x1o + 32x2e")),
        "n_basis":          int(m_cfg.get("n_basis",          8)),
        "local_cutoff":   float(m_cfg.get("local_cutoff",     5.0)),
        "lr_cutoff":      float(m_cfg.get("lr_cutoff",        12.0)),
        "l_max":            int(m_cfg.get("l_max",            2)),
        "long_range_type":  str(m_cfg.get("long_range_type",  "invariant_attention")),
        "n_heads":          int(m_cfg.get("n_heads",          4)),
        "avg_neighbors":  float(m_cfg.get("avg_neighbors",    20.0)),
        "dropout":        float(m_cfg.get("dropout",          0.0)),
        "atomic_energies":  atomic_energies,
    }
    model = UnifiedEquivariantMLIP(**model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = EnergyForceLoss(
        w_energy    = float(t_cfg.get("w_energy",  1.0)),
        w_force     = float(t_cfg.get("w_force",   1.0)),
        w_stress    = float(t_cfg.get("w_stress",  0.0)),
        energy_loss =   str(t_cfg.get("energy_loss", "mae")),
        force_loss  =   str(t_cfg.get("force_loss",  "rmse")),
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer_kwargs = dict(
        model_config   = model_config,
        loss_fn        = loss_fn,
        train_loader   = train_loader,
        val_loader     = val_loader,
        lr             = float(t_cfg.get("lr",           3e-4)),
        lr_min         = float(t_cfg.get("lr_min",       3e-6)),
        weight_decay   = float(t_cfg.get("weight_decay", 1e-5)),
        max_grad_norm  = float(t_cfg.get("max_grad_norm", 1.0)),
        n_epochs       =   int(t_cfg.get("n_epochs",     500)),
        warmup_steps   =   int(t_cfg.get("warmup_steps", 1000)),
        device         = device,
        output_dir     = output_dir,
        patience       =   int(t_cfg.get("patience",     50)),
    )

    if args.resume:
        trainer = Trainer.from_checkpoint(
            checkpoint_path=args.resume,
            model_cls=UnifiedEquivariantMLIP,
            **trainer_kwargs,
        )
    else:
        trainer = Trainer(model=model, **trainer_kwargs)

    trainer.run()


if __name__ == "__main__":
    main()
