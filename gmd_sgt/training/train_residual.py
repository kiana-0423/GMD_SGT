"""Stage 2 training entry point for residual correction branches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional

import torch
from torch.utils.data import DataLoader

from gmd_sgt.api import load_config
from gmd_sgt.data import collate_fn, split_dataset
from gmd_sgt.model import GMDSGTModel
from gmd_sgt.training.loss import EnergyForceLoss
from gmd_sgt.training.train_backbone import _load_dataset, _make_dry_run_dataset
from gmd_sgt.training.trainer import Trainer


def _build_residual_model_config(
    cfg: dict,
    backbone_config: dict,
) -> dict:
    model_cfg = cfg.get("model", {})
    return {
        "backbone_config": dict(backbone_config),
        "use_gnn": bool(model_cfg.get("use_gnn", True)),
        "gnn_hidden_channels": model_cfg.get("gnn_hidden_channels"),
        "gnn_layers": int(model_cfg.get("gnn_layers", 2)),
        "use_transformer": bool(model_cfg.get("use_transformer", False)),
        "transformer_hidden_channels": model_cfg.get("transformer_hidden_channels"),
        "transformer_layers": int(model_cfg.get("transformer_layers", 1)),
        "transformer_heads": int(model_cfg.get("transformer_heads", 4)),
        "transformer_dropout": float(model_cfg.get("transformer_dropout", 0.0)),
        "lambda_gnn": float(model_cfg.get("lambda_gnn", 1.0)),
        "lambda_attn": float(model_cfg.get("lambda_attn", 1.0)),
    }


def _apply_backbone_freeze(model: GMDSGTModel, train_cfg: dict) -> None:
    freeze_backbone = bool(train_cfg.get("freeze_backbone", False))
    semi_freeze_backbone = bool(train_cfg.get("semi_freeze_backbone", False))
    if freeze_backbone and semi_freeze_backbone:
        raise ValueError("freeze_backbone and semi_freeze_backbone cannot both be true")
    if freeze_backbone:
        model.freeze_backbone()
    elif semi_freeze_backbone:
        model.semi_freeze_backbone()


def train_residual(
    train_config: str | Path | Mapping[str, object],
    output_dir: str | Path | None = None,
    resume_checkpoint: Optional[str | Path] = None,
) -> str:
    """Train the Stage 2 residual model and return the best checkpoint path."""
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

    backbone_checkpoint = model_cfg.get("backbone_checkpoint")
    if not backbone_checkpoint:
        raise ValueError("model.backbone_checkpoint must be set for residual training")

    backbone_meta = torch.load(
        str(backbone_checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    backbone_config = dict(backbone_meta["model_config"])
    resolved_model_config = _build_residual_model_config(cfg, backbone_config)

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

    output_path = Path(output_dir or data_cfg.get("output_dir", "outputs/stage2_residual"))
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
            model_cls=GMDSGTModel,
            **trainer_kwargs,
        )
        _apply_backbone_freeze(trainer.model, train_cfg)
    else:
        model = GMDSGTModel(**resolved_model_config)
        model.load_backbone_checkpoint(str(backbone_checkpoint))
        _apply_backbone_freeze(model, train_cfg)
        trainer = Trainer(model=model, **trainer_kwargs)

    if not any(param.requires_grad for param in trainer.model.parameters()):
        raise ValueError("No trainable parameters remain after applying freeze policy")

    trainer.run()
    best_path = output_path / "ckpt_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training completed without producing {best_path}")
    return str(best_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 residual corrections")
    parser.add_argument("--config", required=True, help="Path to stage-2 YAML config")
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
        data_cfg.setdefault("output_dir", "outputs/stage2_residual_dry_run")

    best_path = train_residual(
        train_config=cfg,
        output_dir=data_cfg.get("output_dir"),
        resume_checkpoint=args.resume,
    )
    print(f"Best checkpoint: {Path(best_path)}")


if __name__ == "__main__":
    main()
