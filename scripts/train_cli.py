"""Command-line training entry point.

Kept for ``python scripts/train_cli.py --config ...`` compatibility; the
implementation lives in :mod:`gmd_sgt.cli` (installed as ``gmd-train``).
"""

from __future__ import annotations

from gmd_sgt.cli import main

if __name__ == "__main__":
    main()
