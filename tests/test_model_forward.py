from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gmd_sgt.inference import MLIPCalculator
from gmd_sgt.model import AllegroStyleBackbone, GMDSGTModel
from gmd_sgt.training.train_backbone import train_backbone
from gmd_sgt.training.train_residual import train_residual
from gmd_sgt.training.trainer import Trainer


def _model_config() -> dict:
    return {
        "n_species": 10,
        "hidden_channels": 32,
        "num_layers": 2,
        "n_basis": 4,
        "cutoff": 4.0,
        "l_max": 1,
        "avg_neighbors": 4.0,
    }


def _structure():
    species = torch.tensor([8, 1, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.95, 0.0, 0.0],
            [-0.3, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )
    batch = torch.zeros(3, dtype=torch.long)
    return species, positions, batch


def _write_backbone_checkpoint(tmp_path, model_config: dict | None = None) -> Path:
    model_config = model_config or _model_config()
    model = AllegroStyleBackbone(**model_config)
    checkpoint_path = tmp_path / "backbone_ckpt.pt"
    torch.save(
        {
            "epoch": 1,
            "val_loss": 0.0,
            "best_val": 0.0,
            "early_stopping_counter": 0,
            "model_type": "AllegroStyleBackbone",
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_backbone_forward_outputs_energy_and_forces():
    model = AllegroStyleBackbone(**_model_config())
    model.eval()
    species, positions, batch = _structure()

    out = model(
        species=species,
        positions=positions,
        batch=batch,
        compute_forces=True,
    )

    assert out["energy"].shape == (1,)
    assert out["forces"].shape == (3, 3)
    assert out["atomic_energies"].shape == (3,)


def test_backbone_accepts_explicit_neighbor_list():
    model = AllegroStyleBackbone(**_model_config())
    model.eval()
    species, positions, batch = _structure()
    edge_index = torch.tensor(
        [
            [0, 1, 0, 2, 1, 2],
            [1, 0, 2, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    edge_shift = torch.zeros((edge_index.shape[1], 3), dtype=torch.float32)

    out = model(
        species=species,
        positions=positions,
        batch=batch,
        neighbor_list={"edge_index": edge_index, "edge_shift": edge_shift},
        compute_forces=True,
    )

    assert out["energy"].shape == (1,)
    assert out["forces"].shape == (3, 3)


def test_backbone_checkpoint_roundtrip_predictions_match(tmp_path):
    model_config = _model_config()
    model = AllegroStyleBackbone(**model_config)
    model.eval()
    species, positions, batch = _structure()

    with torch.enable_grad():
        out = model(
            species=species,
            positions=positions.clone(),
            batch=batch,
            compute_forces=True,
        )

    checkpoint_path = _write_backbone_checkpoint(tmp_path, model_config)

    calculator = MLIPCalculator.from_checkpoint(str(checkpoint_path), device="cpu")
    result = calculator.compute(
        positions=positions.numpy().astype(np.float32),
        species=species.numpy().astype(np.int64),
    )

    assert result["energy"] == pytest.approx(out["energy"].sum().item())
    np.testing.assert_allclose(
        result["forces"],
        out["forces"].detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_train_backbone_dry_run_returns_checkpoint(tmp_path, monkeypatch):
    def fake_run(self):
        self.best_val = 0.0
        self.save_checkpoint(epoch=1, val_loss=0.0, tag="best")

    monkeypatch.setattr(Trainer, "run", fake_run)

    best_path = train_backbone(
        {
            "model": _model_config(),
            "train": {
                "device": "cpu",
                "dry_run": True,
                "batch_size": 2,
                "n_epochs": 1,
                "warmup_steps": 1,
            },
            "data": {
                "output_dir": str(tmp_path / "stage1_run"),
            },
        }
    )

    assert Path(best_path).exists()
    assert Path(best_path).name == "ckpt_best.pt"


def test_residual_model_forward_outputs_total_and_component_energies(tmp_path):
    backbone_checkpoint = _write_backbone_checkpoint(tmp_path)
    model = GMDSGTModel.from_backbone_checkpoint(
        backbone_checkpoint,
        use_gnn=True,
        gnn_hidden_channels=32,
        gnn_layers=2,
        use_transformer=False,
        lambda_gnn=1.0,
        lambda_attn=0.0,
    )
    model.eval()
    species, positions, batch = _structure()

    out = model(
        species=species,
        positions=positions,
        batch=batch,
        compute_forces=True,
    )

    assert out["energy"].shape == (1,)
    assert out["energy_backbone"].shape == (1,)
    assert out["delta_energy_gnn"].shape == (1,)
    assert out["delta_energy_attn"].shape == (1,)
    assert out["forces"].shape == (3, 3)


def test_residual_model_accepts_optional_transformer_branch(tmp_path):
    backbone_checkpoint = _write_backbone_checkpoint(tmp_path)
    model = GMDSGTModel.from_backbone_checkpoint(
        backbone_checkpoint,
        use_gnn=True,
        gnn_hidden_channels=32,
        gnn_layers=1,
        use_transformer=True,
        transformer_hidden_channels=32,
        transformer_layers=1,
        transformer_heads=4,
        lambda_gnn=1.0,
        lambda_attn=1.0,
    )
    model.eval()
    species, positions, batch = _structure()

    out = model(
        species=species,
        positions=positions,
        batch=batch,
        compute_forces=True,
    )

    assert out["energy"].shape == (1,)
    assert out["forces"].shape == (3, 3)


def test_residual_model_checkpoint_roundtrip_predictions_match(tmp_path):
    backbone_checkpoint = _write_backbone_checkpoint(tmp_path)
    model = GMDSGTModel.from_backbone_checkpoint(
        backbone_checkpoint,
        use_gnn=True,
        gnn_hidden_channels=32,
        gnn_layers=2,
        use_transformer=False,
        lambda_gnn=1.0,
        lambda_attn=0.0,
    )
    model.eval()
    species, positions, batch = _structure()

    with torch.enable_grad():
        out = model(
            species=species,
            positions=positions.clone(),
            batch=batch,
            compute_forces=True,
        )

    checkpoint_path = tmp_path / "residual_ckpt.pt"
    torch.save(
        {
            "epoch": 1,
            "val_loss": 0.0,
            "best_val": 0.0,
            "early_stopping_counter": 0,
            "model_type": "GMDSGTModel",
            "model_config": {
                "backbone_config": _model_config(),
                "use_gnn": True,
                "gnn_hidden_channels": 32,
                "gnn_layers": 2,
                "use_transformer": False,
                "lambda_gnn": 1.0,
                "lambda_attn": 0.0,
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
        },
        checkpoint_path,
    )

    calculator = MLIPCalculator.from_checkpoint(str(checkpoint_path), device="cpu")
    result = calculator.compute(
        positions=positions.numpy().astype(np.float32),
        species=species.numpy().astype(np.int64),
    )

    assert result["energy"] == pytest.approx(out["energy"].sum().item())
    np.testing.assert_allclose(
        result["forces"],
        out["forces"].detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_train_residual_dry_run_returns_checkpoint(tmp_path, monkeypatch):
    def fake_run(self):
        self.best_val = 0.0
        self.save_checkpoint(epoch=1, val_loss=0.0, tag="best")

    monkeypatch.setattr(Trainer, "run", fake_run)
    backbone_checkpoint = _write_backbone_checkpoint(tmp_path)

    best_path = train_residual(
        {
            "model": {
                "type": "GMDSGTModel",
                "backbone_checkpoint": str(backbone_checkpoint),
                "use_gnn": True,
                "gnn_hidden_channels": 32,
                "gnn_layers": 2,
                "use_transformer": False,
            },
            "train": {
                "device": "cpu",
                "dry_run": True,
                "freeze_backbone": True,
                "batch_size": 2,
                "n_epochs": 1,
                "warmup_steps": 1,
            },
            "data": {
                "output_dir": str(tmp_path / "stage2_run"),
            },
        }
    )

    assert Path(best_path).exists()
    assert Path(best_path).name == "ckpt_best.pt"
