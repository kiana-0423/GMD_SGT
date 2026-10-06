"""Real (unmocked) short training runs.

These run the actual Trainer loop with energy *and* force supervision, check
that optimiser steps change the intended parameters, that losses decrease,
and that the saved best checkpoint reproduces its recorded validation loss
when reloaded.
"""

from __future__ import annotations

import csv
import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from gmd_sgt.api import train
from gmd_sgt.data import AtomicDataset, collate_fn, split_dataset
from gmd_sgt.inference import MLIPCalculator
from gmd_sgt.models import load_model_from_checkpoint
from gmd_sgt.training import EnergyForceLoss, Trainer
from gmd_sgt.training.train_backbone import _make_dry_run_dataset, train_backbone
from gmd_sgt.training.train_residual import train_residual
from tests._helpers import UNIFIED_IRREPS, backbone_config

STAGE1_TRAIN = {
    "device": "cpu",
    "dry_run": True,
    "dry_run_frames": 12,
    "batch_size": 3,
    "n_epochs": 20,
    "lr": 1e-2,
    "warmup_steps": 1,
    "val_fraction": 0.25,
    "test_fraction": 0.0,
    "patience": 0,
    "w_energy": 1.0,
    "w_force": 1.0,
}


def _csv_rows(output_dir):
    with open(output_dir / "training_log.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def _state(path):
    return torch.load(str(path), map_location="cpu", weights_only=False)["model_state_dict"]


def _recompute_val_loss(checkpoint_path, train_cfg):
    """Validation loss of the reloaded checkpoint on the same deterministic split."""
    checkpoint, model = load_model_from_checkpoint(checkpoint_path)
    _, val_set, _ = split_dataset(
        _make_dry_run_dataset(train_cfg["dry_run_frames"]),
        val_fraction=train_cfg["val_fraction"],
        test_fraction=train_cfg["test_fraction"],
        seed=42,
    )
    loader = DataLoader(val_set, batch_size=train_cfg["batch_size"], collate_fn=collate_fn)
    loss_fn = EnergyForceLoss(w_energy=1.0, w_force=1.0, w_stress=0.0)
    model.eval()
    total, n_batches = 0.0, 0
    for batch in loader:
        pred = model(species=batch["species"], positions=batch["positions"], batch=batch["batch"])
        loss, _ = loss_fn(pred, batch, batch["n_atoms"])
        total += float(loss)
        n_batches += 1
    return checkpoint, total / n_batches


@pytest.fixture(scope="module")
def stage1_checkpoint(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("stage1")
    torch.manual_seed(0)
    best = train_backbone(
        {"model": backbone_config(l_max=2), "train": STAGE1_TRAIN, "data": {"output_dir": str(output_dir)}}
    )
    return output_dir, best


def test_stage1_training_reduces_loss_and_checkpoint_reloads(stage1_checkpoint):
    output_dir, best = stage1_checkpoint
    rows = _csv_rows(output_dir)
    assert len(rows) == STAGE1_TRAIN["n_epochs"]
    train_losses = [float(r["train_total"]) for r in rows]
    assert all(np.isfinite(train_losses))
    assert all(r["train_force"] != "" for r in rows), "force term missing from the loss"
    assert min(train_losses[-3:]) < 0.7 * train_losses[0], train_losses

    checkpoint, val_loss = _recompute_val_loss(best, STAGE1_TRAIN)
    assert checkpoint["model_type"] == "AllegroStyleBackbone"
    assert checkpoint["selection_metric"] == "val_total"
    assert val_loss == pytest.approx(checkpoint["val_loss"], rel=1e-5, abs=1e-7)
    assert val_loss == pytest.approx(min(float(r["val_total"]) for r in rows), rel=1e-5)

    # Calculator predictions from the checkpoint equal the reloaded model's.
    _, model = load_model_from_checkpoint(best)
    model.eval()
    positions = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]], dtype=np.float32)
    calc = MLIPCalculator.from_checkpoint(best)
    result = calc.compute(positions, np.array([1, 8]))
    out = model(
        species=torch.tensor([1, 8]),
        positions=torch.tensor(positions),
        batch=torch.zeros(2, dtype=torch.long),
    )
    assert result["energy"] == pytest.approx(float(out["energy"]))
    np.testing.assert_allclose(result["forces"], out["forces"].detach().numpy(), atol=1e-6)
    assert np.abs(result["forces"]).max() > 1e-3


@pytest.mark.parametrize(
    "policy",
    [
        {"freeze_backbone": True},
        {"semi_freeze_backbone": True},
        {"freeze_backbone": False, "semi_freeze_backbone": False},
    ],
    ids=["frozen", "semi_frozen", "unfrozen"],
)
def test_stage2_training_respects_freeze_policy(stage1_checkpoint, tmp_path, policy):
    _, stage1_best = stage1_checkpoint
    torch.manual_seed(1)
    best = train_residual(
        {
            "model": {
                "type": "GMDSGTModel",
                "backbone_checkpoint": stage1_best,
                "use_gnn": True,
                "gnn_hidden_channels": 16,
                "gnn_layers": 1,
                "use_transformer": True,
                "transformer_hidden_channels": 16,
                "transformer_heads": 4,
            },
            "train": {**STAGE1_TRAIN, "n_epochs": 6, **policy},
            "data": {"output_dir": str(tmp_path / "stage2")},
        }
    )
    rows = _csv_rows(tmp_path / "stage2")
    train_losses = [float(r["train_total"]) for r in rows]
    assert all(np.isfinite(train_losses)) and all(r["train_force"] != "" for r in rows)
    assert min(train_losses[1:]) < train_losses[0], train_losses

    stage1 = _state(stage1_best)
    stage2 = _state(best)
    backbone = {k[len("backbone."):]: v for k, v in stage2.items() if k.startswith("backbone.")}
    trainable_changed = [
        k for k, v in backbone.items() if v.is_floating_point() and not torch.equal(v, stage1[k])
    ]
    if policy.get("freeze_backbone"):
        assert trainable_changed == []
    elif policy.get("semi_freeze_backbone"):
        assert trainable_changed and all(k.startswith("readout.") for k in trainable_changed)
    else:
        assert any(not k.startswith("readout.") for k in trainable_changed)

    checkpoint, model = load_model_from_checkpoint(best)
    assert checkpoint["model_type"] == "GMDSGTModel"
    out = model.eval()(
        species=torch.tensor([1, 8]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        batch=torch.zeros(2, dtype=torch.long),
    )
    assert torch.isfinite(out["energy"]).all() and out["delta_energy_gnn"].abs().sum() > 0


def test_unified_training_via_api(tmp_path):
    dataset = tmp_path / "dimers.npz"
    frames = _make_dry_run_dataset(10)
    np.savez(
        dataset,
        R=np.stack([f["positions"].numpy() for f in frames]),
        Z=np.stack([f["species"].numpy() for f in frames]),
        E=np.array([float(f["energy"]) for f in frames]),
        F=np.stack([f["forces"].numpy() for f in frames]),
    )
    torch.manual_seed(0)
    best = train(
        dataset,
        {
            "model": {
                "n_species": 10, "n_blocks": 2, "scalar_dim": 8, "irreps": UNIFIED_IRREPS,
                "n_basis": 4, "local_cutoff": 3.0, "l_max": 2,
                "long_range_type": "invariant_attention", "n_heads": 2, "avg_neighbors": 2.0,
            },
            "train": {
                "device": "cpu", "n_epochs": 8, "batch_size": 4, "lr": 5e-3, "warmup_steps": 1,
                "val_fraction": 0.2, "test_fraction": 0.0, "patience": 0,
            },
        },
        tmp_path / "unified",
    )
    rows = _csv_rows(tmp_path / "unified")
    losses = [float(r["train_total"]) for r in rows]
    assert all(r["train_force"] != "" for r in rows)
    assert min(losses[1:]) < losses[0], losses

    _, model = load_model_from_checkpoint(best)
    calc = MLIPCalculator.from_checkpoint(best)
    pos = np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32)
    res = calc.compute(pos, np.array([1, 8]))
    ref = model.eval()(torch.tensor([1, 8]), torch.tensor(pos), torch.zeros(2, dtype=torch.long))
    assert res["energy"] == pytest.approx(float(ref["energy"]), rel=1e-6, abs=1e-6)
    assert np.abs(res["forces"]).max() > 1e-4


# ── Empty validation sets and split validation ──────────────────────────────


def test_empty_validation_selects_on_training_loss(tmp_path):
    cfg = {
        "model": backbone_config(l_max=1),
        "train": {**STAGE1_TRAIN, "n_epochs": 3, "val_fraction": 0.0},
        "data": {"output_dir": str(tmp_path / "noval")},
    }
    with pytest.warns(UserWarning, match="Validation set is empty"):
        best = train_backbone(cfg)
    checkpoint = torch.load(best, weights_only=False)
    rows = _csv_rows(tmp_path / "noval")
    assert checkpoint["selection_metric"] == "train_total"
    assert all(r["val_total"] == "" for r in rows)
    assert checkpoint["val_loss"] == pytest.approx(min(float(r["train_total"]) for r in rows))


def test_tiny_dataset_keeps_a_validation_structure_when_requested(tmp_path):
    train_set, val_set, _ = split_dataset(_make_dry_run_dataset(3), val_fraction=0.1, test_fraction=0.0)
    assert len(train_set) == 2 and len(val_set) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # must not fall back to train loss
        best = train_backbone(
            {
                "model": backbone_config(l_max=1),
                "train": {**STAGE1_TRAIN, "dry_run_frames": 3, "n_epochs": 2, "val_fraction": 0.1},
                "data": {"output_dir": str(tmp_path / "tiny")},
            }
        )
    assert torch.load(best, weights_only=False)["selection_metric"] == "val_total"


@pytest.mark.parametrize(
    "val,test,match",
    [(1.2, 0.0, "val_fraction"), (-0.1, 0.0, "val_fraction"), (0.6, 0.5, "< 1"), (0.0, 1.0, "test_fraction")],
)
def test_split_fractions_are_validated(val, test, match):
    with pytest.raises(ValueError, match=match):
        split_dataset(_make_dry_run_dataset(10), val_fraction=val, test_fraction=test)


def test_single_structure_dataset_trains_without_validation():
    train_set, val_set, _ = split_dataset(
        AtomicDataset(_make_dry_run_dataset(1).data), val_fraction=0.5, test_fraction=0.0
    )
    assert len(train_set) == 1 and len(val_set) == 0


# ── Stress supervision ──────────────────────────────────────────────────────


def _lj_dataset(n_frames=6):
    from ase import Atoms
    from ase.calculators.lj import LennardJones

    rng = np.random.default_rng(0)
    items = []
    for _ in range(n_frames):
        cell = np.eye(3) * 3.4 + rng.normal(scale=0.05, size=(3, 3))
        base = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
        atoms = Atoms("Ar4", scaled_positions=base + rng.normal(scale=0.02, size=(4, 3)), cell=cell, pbc=True)
        atoms.calc = LennardJones(sigma=2.2, epsilon=0.1, rc=2.9, smooth=True)
        items.append(
            {
                "species": torch.tensor(atoms.numbers, dtype=torch.long),
                "positions": torch.tensor(atoms.positions, dtype=torch.float32),
                "energy": torch.tensor([atoms.get_potential_energy()], dtype=torch.float32),
                "forces": torch.tensor(atoms.get_forces(), dtype=torch.float32),
                "stress": torch.tensor(atoms.get_stress(voigt=False), dtype=torch.float32),
                "cell": torch.tensor(atoms.cell.array, dtype=torch.float32),
                "pbc": torch.tensor([True, True, True]),
                "n_atoms": 4,
            }
        )
    return AtomicDataset(items)


def test_stress_supervision_trains(tmp_path):
    from gmd_sgt.models import AllegroStyleBackbone

    dataset = _lj_dataset()
    loader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)
    torch.manual_seed(0)
    model = AllegroStyleBackbone(**backbone_config(n_species=20, cutoff=2.9, l_max=1))
    trainer = Trainer(
        model=model, model_config=backbone_config(n_species=20, cutoff=2.9, l_max=1),
        loss_fn=EnergyForceLoss(w_energy=1.0, w_force=1.0, w_stress=10.0),
        train_loader=loader, val_loader=loader, n_epochs=6, lr=5e-3, warmup_steps=1,
        output_dir=str(tmp_path / "stress"),
    )
    batch = next(iter(loader))
    model.train()
    pred = trainer._predict(batch)
    assert pred["stress"].shape == (3, 3, 3)
    _, parts = trainer.loss_fn(pred, batch, batch["n_atoms"])
    stress_before = float(parts["stress"])
    trainer.run()

    model.train()
    _, parts = trainer.loss_fn(trainer._predict(batch), batch, batch["n_atoms"])
    assert float(parts["stress"]) < stress_before


def test_stress_weight_without_labels_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="stress"):
        train_backbone(
            {
                "model": backbone_config(l_max=1),
                "train": {**STAGE1_TRAIN, "w_stress": 1.0},
                "data": {"output_dir": str(tmp_path / "bad")},
            }
        )
    loss_fn = EnergyForceLoss(w_stress=1.0)
    pred = {"energy": torch.zeros(1), "forces": torch.zeros(2, 3)}
    target = {"energy": torch.zeros(1), "forces": torch.zeros(2, 3), "stress": torch.zeros(1, 3, 3)}
    with pytest.raises(ValueError, match="compute_stress"):
        loss_fn(pred, target, torch.tensor([2]))


def test_rmse_force_loss_has_finite_gradient_at_zero_error():
    forces = torch.zeros(3, 3, requires_grad=True)
    loss_fn = EnergyForceLoss(w_energy=0.0, w_force=1.0, w_stress=0.0, force_loss="rmse")
    pred = {"energy": torch.zeros(1), "forces": forces}
    target = {"energy": torch.zeros(1), "forces": torch.zeros(3, 3)}
    loss, _ = loss_fn(pred, target, torch.tensor([3]))
    loss.backward()
    assert torch.isfinite(forces.grad).all()
