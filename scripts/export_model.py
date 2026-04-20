"""Export trained checkpoint to TorchScript for GMD C++ integration.

Usage
-----
  python scripts/export_model.py --checkpoint outputs/run/ckpt_best.pt
  python scripts/export_model.py --checkpoint outputs/run/ckpt_best.pt \\
                                  --output model.pt --device cpu
"""

import argparse
from gmd_se3gnn.inference.export import export_torchscript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to TorchScript for GMD")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="model.pt", help="Output .pt path (default: model.pt)")
    parser.add_argument("--device", default="cpu", help="Export device (default: cpu)")
    args = parser.parse_args()

    export_torchscript(args.checkpoint, args.output, device=args.device)
