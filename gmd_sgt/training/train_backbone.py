"""Stage 1 training entry point for the Allegro-style local backbone."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional

import torch
from torch.utils.data import DataLoader

from gmd_sgt.api import load_config
from gmd_sgt.data import (
    AtomicDataset,
    collate_fn,
    compute_per_species_energy_shift,
    split_dataset,
)
from gmd_sgt.model import AllegroStyleBackbone
from gmd_sgt.training.loss import EnergyForceLoss
from gmd_sgt.training.trainer import Trainer


def _load_dataset(dataset_path: str | Path) -> AtomicDataset:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if path.suffix == ".npz":
        return AtomicDataset.from_npz(str(path))
    return AtomicDataset.from_extxyz(str(path))


def _make_dry_run_dataset(n_frames: int = 12) -> AtomicDataset:
    """Create a tiny harmonic-dimer dataset for smoke testing the Stage 1 path."""
    data = []
    k = 8.0
    r0 = 1.0
    species_patterns = [
        torch.tensor([1, 1], dtype=torch.long),
        torch.tensor([1, 8], dtype=torch.long),
        torch.tensor([8, 8], dtype=torch.long),
    ]
    for idx in range(n_frames):
        distance = 0.8 + 0.04 * idx
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
            dtype=torch.float32,
        )
        species = species_patterns[idx % len(species_patterns)]
        displacement = distance - r0
        energy = 0.5 * k * displacement * displacement
        force_mag = k * displacement
        forces = torch.tensor(
            [
                [force_mag, 0.0, 0.0],
                [-force_mag, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        data.append(
            {
                "species": species,
                "positions": positions,
                "energy": torch.tensor([energy], dtype=torch.float32),
                "forces": forces,
                "n_atoms": 2,
            }
        )
    return AtomicDataset(data)


def train_backbone(
    train_config: str | Path | Mapping[str, object],
    output_dir: str | Path | None = None,
    resume_checkpoint: Optional[str | Path] = None,
) -> str:
    """Train the Stage 1 Allegro-style backbone and return the best checkpoint."""
    cfg = load_config(train_config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})

    dry_run = bool(train_cfg.get("dry_run", False))
    if dry_run:
        dataset = _make_dry_run_dataset(int(train_cfg.get("dry_run_frames", 12)))
    else:
        dataset_path = data_cfg.get("train_file")
        if not dataset_path:
            raise ValueError("data.train_file must be set unless train.dry_run=true")
        dataset = _load_dataset(str(dataset_path))

    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")

    train_set, val_set, _ = split_dataset(
        dataset,
        val_fraction=float(train_cfg.get("val_fraction", 0.1)),
        test_fraction=float(train_cfg.get("test_fraction", 0.0)),
        seed=int(train_cfg.get("seed", 42)),
    )
    if len(train_set) == 0:
        raise ValueError("Training split is empty; adjust split fractions")

    all_species = sorted(
        {
            int(z)
            for item in train_set
            for z in item["species"].tolist()
        }
    )
    atomic_energies = compute_per_species_energy_shift(train_set, all_species)

    resolved_model_config = {
        "n_species": int(model_cfg.get("n_species", 100)),
        "hidden_channels": int(model_cfg.get("hidden_channels", 64)),
        "num_layers": int(model_cfg.get("num_layers", 2)),
        "n_basis": int(model_cfg.get("n_basis", 8)),
        "cutoff": float(model_cfg.get("cutoff", 5.0)),
        "l_max": int(model_cfg.get("l_max", 2)),
        "avg_neighbors": float(model_cfg.get("avg_neighbors", 12.0)),
        "atomic_energies": atomic_energies,
    }

    batch_size = int(train_cfg.get("batch_size", 4))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    loss_fn = EnergyForceLoss(
        w_energy=float(train_cfg.get("w_energy", 1.0)),
        w_force=float(train_cfg.get("w_force", 1.0)),
        w_stress=float(train_cfg.get("w_stress", 0.0)),
        energy_loss=str(train_cfg.get("energy_loss", "mae")),
        force_loss=str(train_cfg.get("force_loss", "rmse")),
    )

    output_path = Path(output_dir or data_cfg.get("output_dir", "outputs/stage1_backbone"))
    trainer_kwargs = dict(
        model_config=resolved_model_config,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=float(train_cfg.get("lr", 3e-4)),
        lr_min=float(train_cfg.get("lr_min", 3e-6)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        n_epochs=int(train_cfg.get("n_epochs", 100)),
        warmup_steps=int(train_cfg.get("warmup_steps", 100)),
        device=str(train_cfg.get("device", "cpu")),
        output_dir=str(output_path),
        patience=int(train_cfg.get("patience", 20)),
    )

    if resume_checkpoint is not None:
        trainer = Trainer.from_checkpoint(
            checkpoint_path=str(resume_checkpoint),
            model_cls=AllegroStyleBackbone,
            **trainer_kwargs,
        )
    else:
        trainer = Trainer(
            model=AllegroStyleBackbone(**resolved_model_config),
            **trainer_kwargs,
        )

    trainer.run()
    best_path = output_path / "ckpt_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training completed without producing {best_path}")
    return str(best_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 1 Allegro-style backbone")
    parser.add_argument("--config", required=True, help="Path to stage-1 YAML config")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda/mps)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a tiny synthetic harmonic-dimer dataset instead of loading files",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.setdefault("train", {})
    data_cfg = cfg.setdefault("data", {})
    if args.device:
        train_cfg["device"] = args.device
    if args.dry_run:
        train_cfg["dry_run"] = True
        data_cfg.setdefault("output_dir", "outputs/stage1_backbone_dry_run")

    best_path = train_backbone(
        train_config=cfg,
        output_dir=data_cfg.get("output_dir"),
        resume_checkpoint=args.resume,
    )
    print(f"Best checkpoint: {Path(best_path)}")


if __name__ == "__main__":
    main()
