#!/usr/bin/env python
"""Quick training for force prediction (fewer epochs, faster iteration).

Usage:
  python scripts/train_scalar_quick.py \
    --data trail/1.extxyz \
    --config configs/default.yaml \
    --output outputs/force_prediction_quick \
    --device cpu \
    --epochs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gmd_sgt.api import load_config, train


def build_training_config(
    base_config_path: str,
    dataset_path: str,
    output_dir: str,
    device: str,
    n_epochs: int = 8,
) -> dict:
    """Load base config and override for quick scalar-model training."""
    cfg = load_config(base_config_path)
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("data", {})

    # Use full dataset
    cfg["data"]["train_file"] = dataset_path
    cfg["data"]["output_dir"] = output_dir
    cfg["train"]["device"] = device
    cfg["train"]["n_epochs"] = n_epochs
    cfg["train"]["batch_size"] = 8  # Larger batch for speed
    cfg["train"]["val_fraction"] = 0.1
    cfg["train"]["test_fraction"] = 0.1
    cfg["train"]["patience"] = 3
    cfg["train"]["warmup_steps"] = 50
    cfg["train"]["lr"] = 1.0e-3
    cfg["train"]["lr_min"] = 1.0e-5

    # Scalar-only model for stable training
    cfg["model"]["scalar_dim"] = 64
    cfg["model"]["irreps"] = "64x0e"  # Scalar only
    cfg["model"]["n_blocks"] = 2
    cfg["model"]["n_basis"] = 8
    cfg["model"]["local_cutoff"] = 5.0
    cfg["model"]["long_range_type"] = "none"

    print(f"[Config] Quick training: {n_epochs} epochs, batch_size=8")
    print(f"[Config] Model: irreps={cfg['model']['irreps']}, n_blocks={cfg['model']['n_blocks']}")

    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick train scalar model for force prediction")
    parser.add_argument("--data", required=True, help="Path to extxyz dataset")
    parser.add_argument("--config", default="configs/default.yaml", help="Base YAML config")
    parser.add_argument("--output", default="outputs/force_prediction_quick", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Training device: cpu|cuda|mps")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    args = parser.parse_args()

    dataset_path = Path(args.data)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}")
        return 1

    output_dir = Path(args.output)
    cfg = build_training_config(args.config, str(dataset_path), str(output_dir), args.device, args.epochs)

    try:
        ckpt_path = train(str(dataset_path), cfg, str(output_dir))
        print(f"\n[SUCCESS] Quick training completed. Checkpoint: {ckpt_path}")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
