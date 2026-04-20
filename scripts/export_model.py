"""Export trained checkpoint to TorchScript for GMD C++ integration.

Usage
-----
  python scripts/export_model.py --checkpoint outputs/run/ckpt_best.pt
  python scripts/export_model.py --checkpoint outputs/run/ckpt_best.pt \\
                                  --output model.pt --device cpu
"""

import argparse
from pathlib import Path

from gmd_se3gnn.api import export_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to TorchScript for GMD")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="model.pt", help="Output .pt path (default: model.pt)")
    parser.add_argument("--device", default="cpu", help="Export device (default: cpu)")
    args = parser.parse_args()

    output_path = Path(args.output)
    exported = export_model(
        model_path=args.checkpoint,
        output_dir=output_path.parent,
        export_config={"device": args.device, "filename": output_path.name},
    )
    print(f"Exported model: {exported}")
