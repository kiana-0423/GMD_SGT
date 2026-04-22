"""Minimal training test on an extxyz dataset.

This script runs a short training job and verifies that ckpt_best.pt is produced.

Usage:
  /home/guozy/workspace/GMD_se3gnn/.venv/bin/python scripts/test_train_extxyz.py \
    --data trail/1.extxyz \
    --config configs/default.yaml \
    --output outputs/smoke_train_extxyz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gmd_sgt.api import load_config, train


def make_subset_extxyz(src_path: Path, dst_path: Path, max_frames: int) -> Path:
    """Write the first max_frames structures into a smaller extxyz file."""
    if max_frames <= 0:
        return src_path

    from ase.io import read as ase_read, write as ase_write

    frames = ase_read(str(src_path), index=f":{max_frames}")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) == 0:
        raise ValueError(f"No frames found in dataset: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(dst_path), frames, format="extxyz")
    return dst_path


def build_test_config(base_config_path: str, dataset_path: str, output_dir: str, device: str) -> dict:
    """Load base config and override it for fast, deterministic smoke training."""
    cfg = load_config(base_config_path)
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("data", {})

    # Keep a tiny but valid train/val split for very small files like 2-frame extxyz.
    cfg["data"]["train_file"] = dataset_path
    cfg["data"]["output_dir"] = output_dir
    cfg["train"]["device"] = device
    cfg["train"]["n_epochs"] = 1
    cfg["train"]["batch_size"] = 1
    cfg["train"]["val_fraction"] = 0.5
    cfg["train"]["test_fraction"] = 0.0
    cfg["train"]["patience"] = 1
    cfg["train"]["warmup_steps"] = 1
    cfg["train"]["lr"] = 3.0e-4
    cfg["train"]["lr_min"] = 3.0e-6

    # Smaller model for a fast smoke run.
    cfg["model"]["n_blocks"] = 1
    cfg["model"]["scalar_dim"] = 32
    cfg["model"]["irreps"] = "32x0e + 16x1o"
    cfg["model"]["n_basis"] = 6
    cfg["model"]["long_range_type"] = "none"

    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal extxyz training smoke test")
    parser.add_argument("--data", required=True, help="Path to extxyz dataset")
    parser.add_argument("--config", default="configs/default.yaml", help="Base YAML config")
    parser.add_argument("--output", default="outputs/smoke_train_extxyz", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Training device: cpu|cuda|mps")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="Use only first N frames for quick smoke training (<=0 means all)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.data)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_for_run = dataset_path
    if args.max_frames > 0:
        subset_path = output_dir / "subset.extxyz"
        dataset_for_run = make_subset_extxyz(dataset_path, subset_path, args.max_frames)

    cfg = build_test_config(
        base_config_path=args.config,
        dataset_path=str(dataset_for_run),
        output_dir=str(output_dir),
        device=args.device,
    )

    print("[test] Starting minimal training run")
    print(f"[test] dataset={dataset_path}")
    print(f"[test] dataset_for_run={dataset_for_run}")
    print(f"[test] output={output_dir}")

    best_ckpt = train(
        dataset_path=str(dataset_for_run),
        train_config=cfg,
        output_dir=str(output_dir),
        resume_checkpoint=None,
    )

    best_path = Path(best_ckpt)
    if not best_path.exists():
        print(f"ERROR: training ended but checkpoint missing: {best_path}")
        return 2

    print(f"[test] SUCCESS: checkpoint generated at {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
