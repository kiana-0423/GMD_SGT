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

Edges follow the training convention ``r_ij = pos[j] - pos[i] + shift`` with
``i = edge_index[0]`` and ``j = edge_index[1]``; both directions of every pair
within ``local_cutoff`` must be supplied. Stress is not part of the exported
signature. Forces use autograd, so the caller must keep gradient mode enabled
(do not wrap calls in ``torch.no_grad()``, ``NoGradGuard`` or inference mode).

Electrostatic models (``long_range_type="electrostatic"``) are non-periodic
only. The exported model raises when any ``edge_shift`` is non-zero, which is
the only periodicity the four-tensor signature reveals. It cannot detect a
periodic system whose neighbor list happens to contain no image edges (e.g. a
cell larger than twice the cutoff, or unwrapped positions passed with zero
shifts): such inputs are evaluated as an isolated cluster, without images.
The caller is responsible for not passing periodic systems to these models.

Supported model types: ``UnifiedEquivariantMLIP`` (every long-range type),
``AllegroStyleBackbone`` and ``GMDSGTModel``. The model is compiled with
``torch.jit.script`` (no tracing), saved, reloaded from disk and checked
against the eager model on several structures (different atom and edge
counts, an empty edge list and periodic shifts) for both energy and forces.
Any mismatch raises :class:`ExportValidationError`.

Usage
-----
  python scripts/export_model.py --checkpoint outputs/run/ckpt_best.pt \\
                                  --output model.pt [--device cpu]
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from gmd_sgt.models import load_model_from_checkpoint
from gmd_sgt.models.blocks import EquivariantLongRangeBlock
from gmd_sgt.models.dependencies import E3NN_AVAILABLE, IrrepsBatchNorm
from gmd_sgt.models.factory import MODEL_REGISTRY
from gmd_sgt.models.geometry import _dense_radius_graph, build_neighbor_graph_pbc


class ExportValidationError(RuntimeError):
    """Raised when the exported model disagrees with the eager model."""


class _FrozenIrrepsAffine(nn.Module):
    """Eval-mode ``e3nn.nn.BatchNorm`` as a per-component affine map.

    In evaluation mode e3nn's BatchNorm uses running statistics, i.e. it is
    ``x * scale + shift`` with ``shift`` non-zero only on scalar channels. That
    map is equivariant and, unlike the e3nn module, TorchScript-compatible.
    """

    def __init__(self, scale: torch.Tensor, shift: torch.Tensor):
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    @classmethod
    def from_batchnorm(cls, bn: nn.Module) -> "_FrozenIrrepsAffine":
        if getattr(bn, "instance", False):
            raise NotImplementedError("instance-norm e3nn BatchNorm cannot be frozen")
        scales: List[torch.Tensor] = []
        shifts: List[torch.Tensor] = []
        i_mean = i_var = i_weight = i_bias = 0
        for mul, dim, is_scalar in bn.irs:
            var = bn.running_var[i_var : i_var + mul]
            i_var += mul
            factor = (var + bn.eps).pow(-0.5)
            if bn.affine:
                factor = factor * bn.weight[i_weight : i_weight + mul]
                i_weight += mul
            offset = torch.zeros_like(factor)
            if is_scalar:
                mean = bn.running_mean[i_mean : i_mean + mul]
                i_mean += mul
                offset = -mean * factor
                if bn.affine and bn.include_bias:
                    offset = offset + bn.bias[i_bias : i_bias + mul]
                    i_bias += mul
            scales.append(factor.repeat_interleave(dim))
            shifts.append(offset.repeat_interleave(dim))
        return cls(torch.cat(scales).detach().clone(), torch.cat(shifts).detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class _ScriptWrapper(nn.Module):
    """Thin wrapper with a TorchScript-compatible fixed signature.

    Differences from the eager model ``forward()``:
      - edge_index and edge_shift are required (not Optional) — GMD always passes them
      - batch is synthesised internally (single graph per call)
      - Returns only energy and forces; no stress (stress requires virial from GMD)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.local_cutoff: float = float(getattr(model, "local_cutoff"))
        self.lr_cutoff: float = float(getattr(model, "lr_cutoff", self.local_cutoff))
        # Electrostatics has no periodic images; mirror the eager forward()
        # rejection for the periodicity the signature can express (shifts).
        self.reject_periodic_shifts: bool = (
            getattr(model, "long_range_type", "none") == "electrostatic"
        )

    def forward(
        self,
        species: torch.Tensor,      # [N] int64
        positions: torch.Tensor,    # [N, 3] float32
        edge_index: torch.Tensor,   # [2, E] int64
        edge_shift: torch.Tensor,   # [E, 3] float32
    ) -> Dict[str, torch.Tensor]:
        if not torch.is_grad_enabled():
            raise RuntimeError(
                "Forces are computed with autograd; call the model with gradient "
                "mode enabled (not under torch.no_grad / NoGradGuard / InferenceMode)"
            )
        if self.reject_periodic_shifts and bool((edge_shift != 0).any()):
            raise RuntimeError(
                "long_range_type='electrostatic' sums bare Coulomb pairs without periodic "
                "images and is only valid for non-periodic structures; received non-zero "
                "edge_shift"
            )
        pos = positions.detach().requires_grad_(True)
        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        out = self.model.compute_energy(species, pos, batch, edge_index, edge_shift, 1, None)
        energy = out["energy"]
        forces = torch.zeros_like(pos)
        if energy.requires_grad:
            grad = torch.autograd.grad([energy.sum()], [pos], allow_unused=True)[0]
            if grad is not None:
                forces = -grad
        return {
            "energy": energy.detach().to(torch.float64),
            "forces": forces.detach(),
        }


def prepare_for_export(model: nn.Module) -> nn.Module:
    """Eval-mode copy with frozen parameters and scriptable normalisation."""
    if type(model).__name__ not in MODEL_REGISTRY:
        raise TypeError(
            f"Unsupported model type {type(model).__name__!r} for TorchScript export; "
            f"supported: {sorted(MODEL_REGISTRY)}"
        )
    prepared = copy.deepcopy(model).eval()
    for module in prepared.modules():
        if (
            isinstance(module, EquivariantLongRangeBlock)
            and IrrepsBatchNorm is not None
            and isinstance(module.norm_equivariant, IrrepsBatchNorm)
        ):
            module.norm_equivariant = _FrozenIrrepsAffine.from_batchnorm(module.norm_equivariant)
    for param in prepared.parameters():
        param.requires_grad_(False)
    return prepared


def script_model(model: nn.Module) -> torch.jit.ScriptModule:
    """Compile ``model`` behind the fixed GMD signature with ``torch.jit.script``."""
    wrapper = _ScriptWrapper(prepare_for_export(model)).eval()
    if E3NN_AVAILABLE:
        from e3nn.util.jit import script as e3nn_script

        return e3nn_script(wrapper)
    return torch.jit.script(wrapper)


def _validation_cases(
    model: nn.Module,
    device: torch.device,
    seed: int = 0,
) -> List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Synthetic structures spanning atom counts, edge counts and periodic shifts."""
    generator = torch.Generator().manual_seed(seed)
    cutoff = float(model.local_cutoff)
    n_species = int(model.atomic_energies_ref.shape[0]) if hasattr(
        model, "atomic_energies_ref"
    ) else int(model.backbone.atomic_energies_ref.shape[0])
    max_z = max(n_species - 1, 1)

    def species_for(n: int) -> torch.Tensor:
        return torch.randint(1, max_z + 1, (n,), generator=generator)

    cases = []
    for n_atoms, box in ((1, 1.0), (2, 0.6), (5, 0.9), (9, 1.3)):
        positions = torch.rand((n_atoms, 3), generator=generator) * cutoff * box
        edge_index = _dense_radius_graph(positions, cutoff)
        edge_shift = torch.zeros((edge_index.shape[1], 3))
        cases.append((f"N={n_atoms}", species_for(n_atoms), positions, edge_index, edge_shift))

    far = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]) * cutoff
    cases.append(
        ("empty edges", species_for(3), far, torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3)))
    )

    if getattr(model, "long_range_type", "none") != "electrostatic":
        cell = torch.eye(3) * (0.8 * cutoff)
        positions = torch.rand((4, 3), generator=generator) @ cell
        edge_index, edge_shift = build_neighbor_graph_pbc(positions, cell, cutoff)
        cases.append(("periodic shifts", species_for(4), positions, edge_index, edge_shift))

    return [
        (name, s.to(device), p.to(device), ei.to(device), es.to(device))
        for name, s, p, ei, es in cases
    ]


def validate_exported_model(
    exported: torch.jit.ScriptModule,
    model: nn.Module,
    device: str | torch.device = "cpu",
    atol: float = 1e-4,
    rtol: float = 1e-4,
    cases: Optional[list] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare exported and eager energy *and* forces; raise on mismatch.

    The cases are evaluated twice so that graphs re-optimised by the
    TorchScript profiling executor after the first calls are also checked.
    Returns a per-case report of edge counts and maximum absolute errors.
    """
    device = torch.device(device)
    model = model.eval()
    cases = cases if cases is not None else _validation_cases(model, device)
    report: Dict[str, Dict[str, float]] = {}
    for name, species, positions, edge_index, edge_shift in list(cases) * 2:
        batch = torch.zeros(positions.shape[0], dtype=torch.long, device=device)
        with torch.enable_grad():
            eager = model(
                species=species,
                positions=positions.clone(),
                batch=batch,
                edge_index=edge_index,
                edge_shift=edge_shift,
                compute_forces=True,
            )
        scripted = exported(species, positions.clone(), edge_index, edge_shift)

        e_ref = eager["energy"].detach().to(torch.float64)
        f_ref = eager["forces"].detach()
        e_out = scripted["energy"]
        f_out = scripted["forces"]
        if e_out.shape != e_ref.shape or f_out.shape != f_ref.shape:
            raise ExportValidationError(
                f"[{name}] shape mismatch: energy {tuple(e_out.shape)} vs {tuple(e_ref.shape)}, "
                f"forces {tuple(f_out.shape)} vs {tuple(f_ref.shape)}"
            )
        if e_out.dtype != torch.float64 or f_out.dtype != positions.dtype:
            raise ExportValidationError(f"[{name}] unexpected output dtypes")
        e_err = float((e_out - e_ref).abs().max())
        f_err = float((f_out - f_ref).abs().max()) if f_ref.numel() else 0.0
        e_tol = atol + rtol * float(e_ref.abs().max())
        f_tol = atol + rtol * (float(f_ref.abs().max()) if f_ref.numel() else 0.0)
        if not (e_err <= e_tol and f_err <= f_tol):
            raise ExportValidationError(
                f"[{name}] exported model disagrees with eager model: "
                f"|dE|={e_err:.3e} (tol {e_tol:.1e}), max|dF|={f_err:.3e} (tol {f_tol:.1e})"
            )
        previous = report.get(name, {"energy_abs_err": 0.0, "forces_max_abs_err": 0.0})
        report[name] = {
            "n_atoms": float(positions.shape[0]),
            "n_edges": float(edge_index.shape[1]),
            "energy_abs_err": max(e_err, previous["energy_abs_err"]),
            "forces_max_abs_err": max(f_err, previous["forces_max_abs_err"]),
        }
    return report


def export_torchscript(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu",
    validate: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Export a trained checkpoint to a TorchScript .pt file for GMD.

    Parameters
    ----------
    checkpoint_path:
        Path to a .pt checkpoint saved by Trainer.save_checkpoint().
    output_path:
        Destination path for the exported TorchScript model (e.g. 'model.pt').
    device:
        Device to compile and validate on ('cpu' recommended for portability).
    validate:
        Reload the saved file and compare energies and forces with the eager
        model (default). Disabling this is not recommended.

    Returns
    -------
    Validation report (empty when ``validate=False``).
    """
    _, model = load_model_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    model.to(device)

    scripted = script_model(model)
    scripted.save(output_path)

    report: Dict[str, Dict[str, float]] = {}
    if validate:
        loaded = torch.jit.load(output_path, map_location=device)
        report = validate_exported_model(loaded, model, device=device)

    print(f"Exported TorchScript model → {output_path}")
    print(f"  model_type   : {type(model).__name__}")
    print(f"  local_cutoff : {model.local_cutoff} Å")
    print(f"  lr_cutoff    : {model.lr_cutoff} Å")
    print(f"  Parameters   : {sum(p.numel() for p in model.parameters()):,}")
    if report:
        worst_e = max(r["energy_abs_err"] for r in report.values())
        worst_f = max(r["forces_max_abs_err"] for r in report.values())
        print(
            f"  Validated    : {len(report)} structures vs eager "
            f"(max |dE|={worst_e:.2e} eV, max |dF|={worst_f:.2e} eV/Å)"
        )
    print()
    print("GMD run.in usage:")
    print("  force_field ml")
    print(f"  model_path  {output_path}")
    return report


__all__ = [
    "ExportValidationError",
    "export_torchscript",
    "prepare_for_export",
    "script_model",
    "validate_exported_model",
]
