"""Plot one-frame force correlation (DFT vs model prediction).

Usage:
  /home/guozy/workspace/GMD_se3gnn/.venv/bin/python scripts/plot_force_corr_one_frame.py \
    --data trail/1.extxyz \
    --checkpoint outputs/smoke_train_extxyz/ckpt_best.pt \
    --frame 0 \
    --output outputs/force_corr_frame0.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gmd_sgt.inference import MLIPCalculator


COLOR_BY_SYMBOL = {
    "C": "black",
    "O": "red",
    "H": "blue",
    "F": "green",
}


def _parse_frame(path: str, frame_idx: int):
    """Parse one frame from an extxyz-like file.

    Expects per-atom columns: species x y z fx fy fz.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    current = 0
    n_lines = len(lines)
    while i < n_lines:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        n_atoms = int(line)
        if i + 1 + n_atoms >= n_lines:
            raise ValueError("Malformed extxyz: truncated frame")
        comment = lines[i + 1]
        atom_lines = lines[i + 2 : i + 2 + n_atoms]

        if current == frame_idx:
            symbols = []
            positions = []
            forces = []
            for ln in atom_lines:
                parts = ln.split()
                if len(parts) < 7:
                    raise ValueError("Expected at least 7 columns per atom line")
                sym = parts[0]
                x, y, z = map(float, parts[1:4])
                fx, fy, fz = map(float, parts[4:7])
                symbols.append(sym)
                positions.append([x, y, z])
                forces.append([fx, fy, fz])

            lattice_match = re.search(r'Lattice="([^"]+)"', comment)
            cell = None
            pbc = False
            if lattice_match:
                vals = [float(v) for v in lattice_match.group(1).split()]
                if len(vals) == 9:
                    cell = np.array(vals, dtype=np.float32).reshape(3, 3)

            pbc_match = re.search(r'pbc="([^"]+)"', comment)
            if pbc_match:
                flags = pbc_match.group(1).split()
                pbc = any(flag.upper().startswith("T") for flag in flags)

            species = np.array([atomic_numbers[s] for s in symbols], dtype=np.int64)
            return {
                "symbols": symbols,
                "species": species,
                "positions": np.array(positions, dtype=np.float32),
                "forces": np.array(forces, dtype=np.float64),
                "cell": cell,
                "pbc": pbc,
            }

        current += 1
        i += 2 + n_atoms

    raise IndexError(f"Frame index out of range: {frame_idx}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot one-frame force correlation")
    parser.add_argument("--data", required=True, help="Path to extxyz file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--frame", type=int, default=0, help="Frame index in extxyz")
    parser.add_argument(
        "--output",
        default="outputs/force_corr_frame0.png",
        help="Output image path",
    )
    parser.add_argument(
        "--zoom-y",
        type=float,
        default=0.5,
        help="Half-range for zoomed y-axis view (eV/A)",
    )
    parser.add_argument("--device", default="cpu", help="cpu|cuda|mps")
    args = parser.parse_args()

    frame = _parse_frame(args.data, args.frame)
    dft_forces = frame["forces"]
    species = frame["species"]
    symbols = frame["symbols"]
    cell = frame["cell"]
    pbc = bool(frame["pbc"])

    calc = MLIPCalculator.from_checkpoint(args.checkpoint, device=args.device)
    pred = calc.compute(
        positions=frame["positions"],
        species=species,
        cell=cell,
        pbc=pbc,
    )
    model_forces = np.asarray(pred["forces"], dtype=np.float64)

    x = dft_forces.reshape(-1)
    y = model_forces.reshape(-1)
    if x.size < 2:
        print("ERROR: not enough force components to compute correlation")
        return 2
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        r = 0.0
    else:
        r = float(np.corrcoef(x, y)[0, 1])

    lim = float(max(np.max(np.abs(x)), np.max(np.abs(y))))
    lim = max(1.0, np.ceil(lim))

    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(10.2, 5.0), dpi=160)

    unique_symbols = list(dict.fromkeys(symbols))
    for symbol in unique_symbols:
        idx = np.array([i for i, s in enumerate(symbols) if s == symbol], dtype=np.int64)
        xs = dft_forces[idx].reshape(-1)
        ys = model_forces[idx].reshape(-1)
        color = COLOR_BY_SYMBOL.get(symbol, None)
        ax_main.scatter(xs, ys, s=18, alpha=0.75, label=symbol, c=color)
        ax_zoom.scatter(xs, ys, s=18, alpha=0.75, label=symbol, c=color)

    for ax in (ax_main, ax_zoom):
        ax.plot([-lim, lim], [-lim, lim], linewidth=1.8)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("DFT forces (eV/A)")
        ax.grid(True, linestyle="--", alpha=0.35)

    ax_main.set_ylim(-lim, lim)
    ax_main.set_ylabel("Model forces (eV/A)")
    ax_main.set_title(f"Full range (r={r:.3f})")
    ax_main.legend(frameon=True)

    z = max(0.05, float(args.zoom_y))
    ax_zoom.set_ylim(-z, z)
    ax_zoom.set_title(f"Zoom y in [-{z:.2f}, {z:.2f}] eV/A")

    pred_std = float(np.std(y))
    pred_abs_mean = float(np.mean(np.abs(y)))
    fig.suptitle(
        f"Force correlation: r={r:.3f}, pred_std={pred_std:.3f}, pred_abs_mean={pred_abs_mean:.3f}",
        y=1.02,
    )
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
