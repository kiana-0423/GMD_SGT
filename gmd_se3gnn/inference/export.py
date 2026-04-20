"""Export trained model to TorchScript (.pt) for GMD C++ integration.

GMD loads the exported model via libtorch:
  torch::jit::load("model.pt")

The exported forward() signature (agreed with GMD):
  forward(species:     Tensor[N]    int64,
          positions:   Tensor[N,3]  float32,
          edge_index:  Tensor[2,E]  int64,
          edge_shift:  Tensor[E,3]  float32) -> Dict[str, Tensor]

Returns:
  {"energy": Tensor[1] float64,   # total energy in eV
   "forces": Tensor[N,3] float32} # forces in eV/Å

Usage
-----
  python scripts/export_model.py --checkpoint outputs/run/ckpt_best.pt \\
                                  --output model.pt [--device cpu]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gmd_se3gnn.model import UnifiedEquivariantMLIP


class _ScriptWrapper(nn.Module):
    """Thin wrapper with a TorchScript-compatible fixed signature.

    Differences from UnifiedEquivariantMLIP.forward():
      - edge_index and edge_shift are required (not Optional) — GMD always passes them
      - batch is synthesised internally (single graph per call)
      - Returns only energy and forces; no stress (stress requires virial from GMD)
    """

    def __init__(self, model: UnifiedEquivariantMLIP):
        super().__init__()
        self.model = model
        self.local_cutoff: float = model.local_cutoff
        self.lr_cutoff: float = model.lr_cutoff

    def forward(
        self,
        species: torch.Tensor,      # [N] int64
        positions: torch.Tensor,    # [N, 3] float32
        edge_index: torch.Tensor,   # [2, E] int64
        edge_shift: torch.Tensor,   # [E, 3] float32
    ) -> dict[str, torch.Tensor]:
        batch = torch.zeros(positions.shape[0], dtype=torch.long,
                            device=positions.device)
        out = self.model(
            species=species,
            positions=positions,
            batch=batch,
            edge_index=edge_index,
            edge_shift=edge_shift,
            compute_forces=True,
            compute_stress=False,
        )
        return {
            "energy": out["energy"].to(torch.float64),
            "forces": out["forces"],
        }


def export_torchscript(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu",
) -> None:
    """Export a trained checkpoint to a TorchScript .pt file for GMD.

    Parameters
    ----------
    checkpoint_path:
        Path to a .pt checkpoint saved by Trainer.save_checkpoint().
    output_path:
        Destination path for the exported TorchScript model (e.g. 'model.pt').
    device:
        Device to run the export trace on ('cpu' recommended for portability).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_config" not in ckpt:
        raise KeyError(
            f"Checkpoint {checkpoint_path!r} is missing 'model_config'. "
            "Re-train with the current Trainer to produce a valid checkpoint."
        )

    model = UnifiedEquivariantMLIP(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    wrapper = _ScriptWrapper(model)
    wrapper.eval()

    scripted = torch.jit.script(wrapper)
    scripted.save(output_path)

    print(f"Exported TorchScript model → {output_path}")
    print(f"  local_cutoff : {model.local_cutoff} Å")
    print(f"  lr_cutoff    : {model.lr_cutoff} Å")
    print(f"  Parameters   : {sum(p.numel() for p in model.parameters()):,}")
    print()
    print("GMD run.in usage:")
    print("  force_field ml")
    print(f"  model_path  {output_path}")
