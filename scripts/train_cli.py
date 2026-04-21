"""Command-line training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gmd_sgt.api import load_config, train


def main():
    parser = argparse.ArgumentParser(description="Train GMD-SGT MLIP")
    parser.add_argument("--config",  required=True, help="Path to YAML config file")
    parser.add_argument("--device",  default=None,  help="Override device (cpu/cuda/mps)")
    parser.add_argument("--resume",  default=None,  help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t_cfg = cfg.setdefault("train", {})
    d_cfg = cfg.get("data", {})
    train_file = d_cfg.get("train_file")
    if train_file is None:
        print("ERROR: data.train_file must be set in the config.")
        sys.exit(1)

    if args.device:
        t_cfg["device"] = args.device

    output_dir = d_cfg.get("output_dir", "outputs/run")
    best_path = train(
        dataset_path=train_file,
        train_config=cfg,
        output_dir=output_dir,
        resume_checkpoint=args.resume,
    )
    print(f"Best checkpoint: {Path(best_path)}")


if __name__ == "__main__":
    main()
