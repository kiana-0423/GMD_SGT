"""Stable training, export, and online inference APIs for external adapters."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError as exc:  # pragma: no cover - declared dependency
    raise ImportError("pyyaml is required for configuration support") from exc

from gmd_se3gnn.data import (
    AtomicDataset,
    collate_fn,
    compute_per_species_energy_shift,
    split_dataset,
)
from gmd_se3gnn.inference.calculator import MLIPCalculator
from gmd_se3gnn.inference.export import export_torchscript
from gmd_se3gnn.model import UnifiedEquivariantMLIP
from gmd_se3gnn.training import EnergyForceLoss, Trainer

LOGGER = logging.getLogger(__name__)

_ONLINE_MONITORING_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "return_energy": True,
    "return_ensemble_forces": False,
    "return_latent_descriptor": False,
    "return_unsafe_probability": False,
    "batch_size": 1,
    "device": "cpu",
    "ensemble": {
        "enabled": False,
        "members": None,
        "checkpoint_paths": [],
    },
}


@dataclass(frozen=True)
class StructureInput:
    """Single structure input for online inference."""

    positions: np.ndarray
    species: np.ndarray
    cell: Optional[np.ndarray] = None
    pbc: bool = False
    edge_index: Optional[np.ndarray] = None
    edge_shift: Optional[np.ndarray] = None


@dataclass(frozen=True)
class OnlineMonitoringEnsembleConfig:
    """Ensemble inference settings."""

    enabled: bool = False
    members: Optional[int] = None
    checkpoint_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class OnlineMonitoringConfig:
    """Configuration for online-monitoring-friendly inference outputs."""

    enabled: bool = False
    return_energy: bool = True
    return_ensemble_forces: bool = False
    return_latent_descriptor: bool = False
    return_unsafe_probability: bool = False
    batch_size: int = 1
    device: str = "cpu"
    ensemble: OnlineMonitoringEnsembleConfig = field(
        default_factory=OnlineMonitoringEnsembleConfig
    )


@dataclass
class PredictionResult:
    """Stable, adapter-friendly inference result."""

    energy: Optional[float]
    forces: np.ndarray
    ensemble_forces: Optional[np.ndarray]
    latent_descriptor: Optional[np.ndarray]
    unsafe_probability: Optional[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for external consumers."""
        return {
            "energy": self.energy,
            "forces": self.forces,
            "ensemble_forces": self.ensemble_forces,
            "latent_descriptor": self.latent_descriptor,
            "unsafe_probability": self.unsafe_probability,
            "metadata": self.metadata,
        }


def load_config(config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    """Load configuration from YAML path or mapping."""
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))

    path = Path(config)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration at {path} must deserialize to a mapping")
    return loaded


def train(
    dataset_path: str | Path,
    train_config: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    resume_checkpoint: Optional[str | Path] = None,
) -> str:
    """Train a model and return the best checkpoint path."""
    dataset_path = str(Path(dataset_path))
    output_dir_path = Path(output_dir)
    cfg = load_config(train_config)
    cfg.setdefault("data", {})
    cfg["data"]["train_file"] = dataset_path
    cfg["data"]["output_dir"] = str(output_dir_path)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    LOGGER.info("Starting training with dataset=%s output_dir=%s", dataset_path, output_dir)

    dataset = _load_dataset(dataset_path)
    if len(dataset) == 0:
        raise ValueError(f"Dataset {dataset_path!r} is empty")

    train_set, val_set, _ = split_dataset(
        dataset,
        val_fraction=float(train_cfg.get("val_fraction", 0.10)),
        test_fraction=float(train_cfg.get("test_fraction", 0.05)),
    )
    if len(train_set) == 0:
        raise ValueError("Training split is empty; adjust val/test fractions or dataset size")
    if len(val_set) == 0:
        LOGGER.warning("Validation split is empty; validation metrics will be degenerate")

    all_species = sorted(
        {int(z) for item in train_set for z in item["species"].tolist()}
    )
    atomic_energies = compute_per_species_energy_shift(train_set, all_species)

    batch_size = int(train_cfg.get("batch_size", 4))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    resolved_model_config = {
        "n_species": int(model_cfg.get("n_species", 100)),
        "n_blocks": int(model_cfg.get("n_blocks", 4)),
        "scalar_dim": int(model_cfg.get("scalar_dim", 128)),
        "irreps": str(model_cfg.get("irreps", "128x0e + 64x1o + 32x2e")),
        "n_basis": int(model_cfg.get("n_basis", 8)),
        "local_cutoff": float(model_cfg.get("local_cutoff", 5.0)),
        "lr_cutoff": float(model_cfg.get("lr_cutoff", 12.0)),
        "l_max": int(model_cfg.get("l_max", 2)),
        "long_range_type": str(
            model_cfg.get("long_range_type", "invariant_attention")
        ),
        "n_heads": int(model_cfg.get("n_heads", 4)),
        "avg_neighbors": float(model_cfg.get("avg_neighbors", 20.0)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "atomic_energies": atomic_energies,
    }
    model = UnifiedEquivariantMLIP(**resolved_model_config)

    loss_fn = EnergyForceLoss(
        w_energy=float(train_cfg.get("w_energy", 1.0)),
        w_force=float(train_cfg.get("w_force", 1.0)),
        w_stress=float(train_cfg.get("w_stress", 0.0)),
        energy_loss=str(train_cfg.get("energy_loss", "mae")),
        force_loss=str(train_cfg.get("force_loss", "rmse")),
    )

    trainer_kwargs = dict(
        model_config=resolved_model_config,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=float(train_cfg.get("lr", 3e-4)),
        lr_min=float(train_cfg.get("lr_min", 3e-6)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        n_epochs=int(train_cfg.get("n_epochs", 500)),
        warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        device=str(train_cfg.get("device", "cpu")),
        output_dir=str(output_dir_path),
        patience=int(train_cfg.get("patience", 50)),
    )

    try:
        if resume_checkpoint is not None:
            trainer = Trainer.from_checkpoint(
                checkpoint_path=str(resume_checkpoint),
                model_cls=UnifiedEquivariantMLIP,
                **trainer_kwargs,
            )
        else:
            trainer = Trainer(model=model, **trainer_kwargs)
        trainer.run()
    except Exception as exc:
        LOGGER.exception("Training failed")
        raise RuntimeError("Training failed") from exc

    best_path = output_dir_path / "ckpt_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Training completed without producing {best_path}"
        )
    LOGGER.info("Training finished; best checkpoint at %s", best_path)
    return str(best_path)


def export_model(
    model_path: str | Path,
    output_dir: str | Path,
    export_config: str | Path | Mapping[str, Any] | None = None,
) -> str:
    """Export a checkpoint to a deployable TorchScript artifact."""
    cfg = load_config(export_config)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    device = str(cfg.get("device", "cpu"))
    filename = str(cfg.get("filename", "model.pt"))
    output_path = output_dir_path / filename

    LOGGER.info("Exporting checkpoint=%s to %s", model_path, output_path)
    try:
        export_torchscript(str(model_path), str(output_path), device=device)
    except Exception as exc:
        LOGGER.exception("Export failed")
        raise RuntimeError("Model export failed") from exc

    if not output_path.exists():
        raise FileNotFoundError(f"Export did not produce artifact at {output_path}")
    return str(output_path)


def predict(
    checkpoint_path: str | Path,
    structure: StructureInput | Mapping[str, Any],
    predict_config: str | Path | Mapping[str, Any] | None = None,
) -> PredictionResult:
    """Convenience API for one-off online inference."""
    predictor = OnlinePredictor.from_checkpoint(checkpoint_path, predict_config)
    return predictor.predict(structure)


class OnlinePredictor:
    """Config-driven inference wrapper exposing stable prediction outputs."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        predict_config: str | Path | Mapping[str, Any] | None = None,
    ):
        self.checkpoint_path = str(Path(checkpoint_path))
        self.config = _resolve_online_monitoring_config(predict_config)
        self.calculator = MLIPCalculator.from_checkpoint(
            self.checkpoint_path,
            device=self.config.device,
        )
        self._ensemble_calculators = self._load_ensemble_calculators()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        predict_config: str | Path | Mapping[str, Any] | None = None,
    ) -> "OnlinePredictor":
        """Create a predictor from a checkpoint."""
        return cls(checkpoint_path=checkpoint_path, predict_config=predict_config)

    def predict(
        self,
        structure: StructureInput | Mapping[str, Any],
    ) -> PredictionResult:
        """Run single-structure online inference."""
        prepared = _normalize_structure_input(structure)

        result = self.calculator.compute(
            positions=prepared.positions,
            species=prepared.species,
            cell=prepared.cell,
            pbc=prepared.pbc,
            edge_index=prepared.edge_index,
            edge_shift=prepared.edge_shift,
        )

        ensemble_forces: Optional[np.ndarray] = None
        ensemble_energies: Optional[np.ndarray] = None
        if self.config.ensemble.enabled:
            force_members: list[np.ndarray] = []
            energy_members: list[float] = []
            for calc in self._ensemble_calculators:
                member_result = calc.compute(
                    positions=prepared.positions,
                    species=prepared.species,
                    cell=prepared.cell,
                    pbc=prepared.pbc,
                    edge_index=prepared.edge_index,
                    edge_shift=prepared.edge_shift,
                )
                force_members.append(member_result["forces"])
                energy_members.append(float(member_result["energy"]))

            if self.config.return_ensemble_forces:
                ensemble_forces = np.stack(force_members, axis=0)
            ensemble_energies = np.asarray(energy_members, dtype=np.float64)

        metadata = {
            "checkpoint_path": self.checkpoint_path,
            "device": self.config.device,
            "online_monitoring_enabled": self.config.enabled,
            "ensemble_enabled": self.config.ensemble.enabled,
            "ensemble_member_count": len(self._ensemble_calculators),
            "requested_outputs": {
                "energy": self.config.return_energy,
                "ensemble_forces": self.config.return_ensemble_forces,
                "latent_descriptor": self.config.return_latent_descriptor,
                "unsafe_probability": self.config.return_unsafe_probability,
            },
            "unsupported_outputs": _collect_unsupported_outputs(self.config),
        }
        if ensemble_energies is not None:
            metadata["ensemble_energy"] = ensemble_energies

        return PredictionResult(
            energy=float(result["energy"]) if self.config.return_energy else None,
            forces=np.asarray(result["forces"], dtype=np.float64),
            ensemble_forces=ensemble_forces,
            latent_descriptor=None,
            unsafe_probability=None,
            metadata=metadata,
        )

    def predict_batch(
        self,
        structures: Sequence[StructureInput | Mapping[str, Any]],
    ) -> list[PredictionResult]:
        """Run small-batch inference by evaluating structures sequentially."""
        if len(structures) > self.config.batch_size:
            raise ValueError(
                f"Received batch of size {len(structures)} but "
                f"online_monitoring.batch_size={self.config.batch_size}"
            )
        return [self.predict(structure) for structure in structures]

    def _load_ensemble_calculators(self) -> list[MLIPCalculator]:
        if not self.config.ensemble.enabled:
            return []

        calculators: list[MLIPCalculator] = []
        for checkpoint_path in self.config.ensemble.checkpoint_paths:
            calculators.append(
                MLIPCalculator.from_checkpoint(checkpoint_path, device=self.config.device)
            )
        LOGGER.info("Loaded %d ensemble member(s)", len(calculators))
        return calculators


def _load_dataset(dataset_path: str) -> AtomicDataset:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if path.suffix == ".npz":
        return AtomicDataset.from_npz(str(path))
    return AtomicDataset.from_extxyz(str(path))


def _resolve_online_monitoring_config(
    config: str | Path | Mapping[str, Any] | None,
) -> OnlineMonitoringConfig:
    loaded = load_config(config)
    raw = loaded.get("online_monitoring", loaded)
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("online_monitoring configuration must be a mapping")

    merged = _deep_merge_dict(copy.deepcopy(_ONLINE_MONITORING_DEFAULTS), dict(raw))
    ensemble_cfg = merged["ensemble"]
    if int(merged["batch_size"]) < 1:
        raise ValueError("online_monitoring.batch_size must be >= 1")

    checkpoint_paths = tuple(str(Path(p)) for p in ensemble_cfg.get("checkpoint_paths", []))
    members = ensemble_cfg.get("members")
    if members is not None:
        members = int(members)
        if members < 1:
            raise ValueError("online_monitoring.ensemble.members must be >= 1")
    if ensemble_cfg.get("enabled", False):
        if not checkpoint_paths:
            raise ValueError(
                "online_monitoring.ensemble.checkpoint_paths must be provided when "
                "ensemble is enabled"
            )
        if members is None:
            members = len(checkpoint_paths)
        if members < 2:
            raise ValueError(
                "online_monitoring ensemble requires at least 2 member checkpoints"
            )
        if len(checkpoint_paths) < members:
            raise ValueError(
                "online_monitoring.ensemble.members exceeds available checkpoint_paths"
            )
        checkpoint_paths = checkpoint_paths[:members]
    else:
        checkpoint_paths = ()

    return OnlineMonitoringConfig(
        enabled=bool(merged["enabled"]),
        return_energy=bool(merged["return_energy"]),
        return_ensemble_forces=bool(merged["return_ensemble_forces"]),
        return_latent_descriptor=bool(merged["return_latent_descriptor"]),
        return_unsafe_probability=bool(merged["return_unsafe_probability"]),
        batch_size=int(merged["batch_size"]),
        device=str(merged["device"]),
        ensemble=OnlineMonitoringEnsembleConfig(
            enabled=bool(ensemble_cfg.get("enabled", False)),
            members=members,
            checkpoint_paths=checkpoint_paths,
        ),
    )


def _deep_merge_dict(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_merge_dict(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _normalize_structure_input(
    structure: StructureInput | Mapping[str, Any],
) -> StructureInput:
    if isinstance(structure, StructureInput):
        positions = np.asarray(structure.positions, dtype=np.float32)
        species = np.asarray(structure.species, dtype=np.int64)
        cell = None if structure.cell is None else np.asarray(structure.cell, dtype=np.float32)
        edge_index = (
            None if structure.edge_index is None
            else np.asarray(structure.edge_index, dtype=np.int64)
        )
        edge_shift = (
            None if structure.edge_shift is None
            else np.asarray(structure.edge_shift, dtype=np.float32)
        )
        pbc = bool(structure.pbc)
    else:
        if not isinstance(structure, Mapping):
            raise TypeError("structure must be a StructureInput or mapping")
        positions = np.asarray(structure.get("positions"), dtype=np.float32)
        species = _coerce_species(
            species=structure.get("species"),
            symbols=structure.get("symbols"),
        )
        cell = structure.get("cell")
        edge_index = structure.get("edge_index")
        edge_shift = structure.get("edge_shift")
        pbc = _coerce_pbc(structure.get("pbc", False))
        if cell is not None:
            cell = np.asarray(cell, dtype=np.float32)
        if edge_index is not None:
            edge_index = np.asarray(edge_index, dtype=np.int64)
        if edge_shift is not None:
            edge_shift = np.asarray(edge_shift, dtype=np.float32)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if species.ndim != 1:
        raise ValueError("species must have shape (N,)")
    if len(species) != len(positions):
        raise ValueError("positions and species must describe the same atom count")
    if cell is not None and cell.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    if edge_index is not None:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        if edge_shift is None:
            raise ValueError("edge_shift must be provided when edge_index is given")
        if edge_shift.ndim != 2 or edge_shift.shape[1] != 3:
            raise ValueError("edge_shift must have shape (E, 3)")
        if edge_shift.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_shift and edge_index must have the same edge count")

    return StructureInput(
        positions=positions,
        species=species,
        cell=cell,
        pbc=pbc,
        edge_index=edge_index,
        edge_shift=edge_shift,
    )


def _coerce_species(species: Any, symbols: Any) -> np.ndarray:
    if species is not None:
        species_array = np.asarray(species, dtype=np.int64)
        return species_array
    if symbols is None:
        raise ValueError("structure must contain either 'species' or 'symbols'")

    try:
        from ase.data import atomic_numbers
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError("ase is required for symbol-to-species conversion") from exc

    resolved = [atomic_numbers[str(symbol)] for symbol in symbols]
    return np.asarray(resolved, dtype=np.int64)


def _coerce_pbc(pbc: Any) -> bool:
    if isinstance(pbc, (list, tuple, np.ndarray)):
        return bool(np.any(np.asarray(pbc, dtype=bool)))
    return bool(pbc)


def _collect_unsupported_outputs(config: OnlineMonitoringConfig) -> list[str]:
    unsupported: list[str] = []
    if config.return_latent_descriptor:
        unsupported.append("latent_descriptor")
    if config.return_unsafe_probability:
        unsupported.append("unsafe_probability")
    return unsupported


__all__ = [
    "OnlineMonitoringConfig",
    "OnlineMonitoringEnsembleConfig",
    "OnlinePredictor",
    "PredictionResult",
    "StructureInput",
    "export_model",
    "load_config",
    "predict",
    "train",
]
