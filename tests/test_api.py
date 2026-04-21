from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gmd_sgt.api import OnlinePredictor, export_model, predict, train
from gmd_sgt.model import UnifiedEquivariantMLIP
from gmd_sgt.training.trainer import Trainer


@pytest.fixture(scope="module")
def model_config():
    return dict(
        n_species=10,
        n_blocks=1,
        scalar_dim=16,
        irreps="16x0e",
        n_basis=4,
        local_cutoff=4.0,
        lr_cutoff=8.0,
        l_max=0,
        long_range_type="none",
        n_heads=1,
        avg_neighbors=4.0,
    )


@pytest.fixture(scope="module")
def checkpoint_path(model_config, tmp_path_factory):
    model = UnifiedEquivariantMLIP(**model_config)
    path = tmp_path_factory.mktemp("api_ckpt") / "ckpt_a.pt"
    torch.save(
        {
            "epoch": 1,
            "val_loss": 0.5,
            "best_val": 0.5,
            "early_stopping_counter": 0,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
        },
        path,
    )
    return str(path)


@pytest.fixture(scope="module")
def ensemble_checkpoint_paths(model_config, tmp_path_factory):
    root = tmp_path_factory.mktemp("ensemble_ckpt")
    paths: list[str] = []
    for seed, name in [(0, "member_a.pt"), (1, "member_b.pt")]:
        torch.manual_seed(seed)
        model = UnifiedEquivariantMLIP(**model_config)
        path = root / name
        torch.save(
            {
                "epoch": 1,
                "val_loss": 0.5,
                "best_val": 0.5,
                "early_stopping_counter": 0,
                "model_config": model_config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "scheduler_state_dict": {},
            },
            path,
        )
        paths.append(str(path))
    return paths


@pytest.fixture()
def structure():
    return {
        "positions": np.array(
            [
                [0.000, 0.000, 0.000],
                [0.757, 0.586, 0.000],
                [-0.757, 0.586, 0.000],
            ],
            dtype=np.float32,
        ),
        "symbols": ["O", "H", "H"],
    }


@pytest.fixture()
def dataset_path(tmp_path):
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, -0.1, 0.0]],
        ],
        dtype=np.float32,
    )
    species = np.array([[1, 1]] * 4, dtype=np.int64)
    energy = np.array([0.1, 0.2, 0.15, 0.12], dtype=np.float32)
    forces = np.zeros((4, 2, 3), dtype=np.float32)
    path = tmp_path / "train_data.npz"
    np.savez(path, R=positions, Z=species, E=energy, F=forces)
    return path


def test_single_model_predict_returns_forces_and_energy(checkpoint_path, structure):
    result = predict(
        checkpoint_path,
        structure,
        {"online_monitoring": {"enabled": True, "return_energy": True}},
    )

    assert isinstance(result.energy, float)
    assert result.forces.shape == (3, 3)
    assert result.ensemble_forces is None


def test_predict_without_ensemble_returns_none_for_ensemble_forces(
    checkpoint_path, structure
):
    predictor = OnlinePredictor.from_checkpoint(
        checkpoint_path,
        {"online_monitoring": {"enabled": True, "return_ensemble_forces": False}},
    )

    result = predictor.predict(structure)

    assert result.ensemble_forces is None
    assert result.metadata["ensemble_enabled"] is False


def test_predict_with_ensemble_returns_expected_shape(
    checkpoint_path, ensemble_checkpoint_paths, structure
):
    predictor = OnlinePredictor.from_checkpoint(
        checkpoint_path,
        {
            "online_monitoring": {
                "enabled": True,
                "return_ensemble_forces": True,
                "ensemble": {
                    "enabled": True,
                    "members": 2,
                    "checkpoint_paths": ensemble_checkpoint_paths,
                },
            }
        },
    )

    result = predictor.predict(structure)

    assert result.ensemble_forces is not None
    assert result.ensemble_forces.shape == (2, 3, 3)


def test_predict_invalid_structure_raises_clear_error(checkpoint_path):
    predictor = OnlinePredictor.from_checkpoint(checkpoint_path)

    with pytest.raises(ValueError, match="positions must have shape"):
        predictor.predict({"positions": np.array([0.0, 0.0, 0.0]), "species": [1]})


def test_latent_and_unsafe_disabled_return_none(checkpoint_path, structure):
    predictor = OnlinePredictor.from_checkpoint(
        checkpoint_path,
        {
            "online_monitoring": {
                "enabled": True,
                "return_latent_descriptor": False,
                "return_unsafe_probability": False,
            }
        },
    )

    result = predictor.predict(structure)

    assert result.latent_descriptor is None
    assert result.unsafe_probability is None


def test_train_returns_model_path(dataset_path, tmp_path, monkeypatch):
    def fake_run(self):
        self.best_val = 0.0
        self.save_checkpoint(epoch=1, val_loss=0.0, tag="best")

    monkeypatch.setattr(Trainer, "run", fake_run)

    config = {
        "model": {
            "n_species": 10,
            "n_blocks": 1,
            "scalar_dim": 8,
            "irreps": "8x0e",
            "n_basis": 2,
            "l_max": 0,
            "long_range_type": "none",
            "n_heads": 1,
            "avg_neighbors": 2.0,
        },
        "train": {
            "device": "cpu",
            "n_epochs": 1,
            "batch_size": 2,
            "val_fraction": 0.25,
            "test_fraction": 0.25,
            "warmup_steps": 1,
        },
    }

    best_path = train(dataset_path, config, tmp_path / "train_output")

    assert Path(best_path).exists()
    assert Path(best_path).name == "ckpt_best.pt"


def test_export_model_returns_artifact_path(checkpoint_path, tmp_path):
    exported = export_model(
        checkpoint_path,
        tmp_path,
        {"device": "cpu", "filename": "online_model.pt"},
    )

    assert Path(exported).exists()
    assert Path(exported).name == "online_model.pt"
