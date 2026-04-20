"""
Smoke test: forward pass + backward pass with dummy data.
No real dataset needed.

Usage:
  python scripts/smoke_test.py [--device cpu|cuda] [--n_steps 20]
"""

from __future__ import annotations

import argparse
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW

from gmd_se3gnn.model import UnifiedEquivariantMLIP
from gmd_se3gnn.training import EnergyForceLoss


def make_dummy_batch(
    n_graphs: int = 4,
    n_atoms_per_graph: int = 16,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Random batch with H/C/O atoms for testing."""
    n_total = n_graphs * n_atoms_per_graph
    species_pool = [1, 6, 8]
    species   = torch.tensor([species_pool[i % 3] for i in range(n_total)], dtype=torch.long)
    positions = torch.randn(n_total, 3)
    batch     = torch.arange(n_graphs).repeat_interleave(n_atoms_per_graph)
    energy_gt = torch.randn(n_graphs) * 10.0
    forces_gt = torch.randn(n_total, 3)
    n_atoms   = torch.full((n_graphs,), n_atoms_per_graph, dtype=torch.long)
    return {
        "species":   species.to(device),
        "positions": positions.to(device),
        "batch":     batch.to(device),
        "energy":    energy_gt.to(device),
        "forces":    forces_gt.to(device),
        "n_atoms":   n_atoms.to(device),
    }


def run(n_steps: int = 10, device: str = "cpu"):
    model_config = dict(
        n_species=100,
        n_blocks=2,
        scalar_dim=64,
        irreps="64x0e + 32x1o + 16x2e",
        n_basis=8,
        local_cutoff=5.0,
        lr_cutoff=10.0,
        l_max=2,
        long_range_type="invariant_attention",
        n_heads=4,
        avg_neighbors=8.0,
    )
    model   = UnifiedEquivariantMLIP(**model_config).to(device)
    loss_fn = EnergyForceLoss(w_energy=1.0, w_force=1.0)
    optim   = AdamW(model.parameters(), lr=3e-4)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    for step in range(1, n_steps + 1):
        batch = make_dummy_batch(device=device)
        optim.zero_grad()

        pred = model(
            species=batch["species"],
            positions=batch["positions"],
            batch=batch["batch"],
            compute_forces=True,
        )
        target = {"energy": batch["energy"], "forces": batch["forces"]}
        loss, ld = loss_fn(pred, target, batch["n_atoms"])

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        print(
            f"step {step:3d} | total={loss.item():.4f} | "
            f"E={ld.get('energy', torch.tensor(0.)).item():.4f} | "
            f"F={ld.get('force',  torch.tensor(0.)).item():.4f}"
        )

    print("Smoke test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--n_steps", type=int, default=10)
    args = parser.parse_args()
    run(n_steps=args.n_steps, device=args.device)
