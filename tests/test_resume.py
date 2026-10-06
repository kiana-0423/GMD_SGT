"""Resuming training: model-configuration rules and early-stopping state."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from gmd_sgt.data import collate_fn, split_dataset
from gmd_sgt.models import load_model_from_checkpoint
from gmd_sgt.training import EnergyForceLoss, Trainer
from gmd_sgt.training.train_backbone import _make_dry_run_dataset, train_backbone
from gmd_sgt.training.train_residual import train_residual
from tests._helpers import backbone_config, make_backbone

TRAIN = {
    "device": "cpu",
    "dry_run": True,
    "dry_run_frames": 12,
    "batch_size": 3,
    "lr": 1e-2,
    "warmup_steps": 1,
    "val_fraction": 0.25,
    "test_fraction": 0.0,
    "patience": 0,
    "seed": 42,
}
PROBE = (
    torch.tensor([1, 8, 1]),
    torch.tensor([[0.0, 0.0, 0.0], [0.97, 0.1, 0.0], [-0.2, 1.1, 0.3]]),
    torch.zeros(3, dtype=torch.long),
)


def _predict(model):
    out = model.eval()(PROBE[0], PROBE[1].clone(), PROBE[2])
    return out["energy"].detach(), out["forces"].detach()


def _val_loss(model, train_cfg):
    _, val_set, _ = split_dataset(
        _make_dry_run_dataset(train_cfg["dry_run_frames"]),
        train_cfg["val_fraction"], train_cfg["test_fraction"], train_cfg["seed"],
    )
    loss_fn = EnergyForceLoss(w_energy=1.0, w_force=1.0)
    total = n = 0
    for batch in DataLoader(val_set, batch_size=train_cfg["batch_size"], collate_fn=collate_fn):
        loss, _ = loss_fn(model.eval()(batch["species"], batch["positions"], batch["batch"]), batch, batch["n_atoms"])
        total, n = total + float(loss), n + 1
    return total / n


@pytest.fixture()
def stage1_run(tmp_path):
    torch.manual_seed(0)
    config = {"model": backbone_config(cutoff=3.0), "train": {**TRAIN, "n_epochs": 3},
              "data": {"output_dir": str(tmp_path / "first")}}
    return config, train_backbone(config)


# ── Model configuration on resume ───────────────────────────────────────────


@pytest.mark.parametrize("override", [{"cutoff": 2.0}, {"hidden_channels": 32}, {"l_max": 1}])
def test_resume_rejects_incompatible_model_config_before_training(stage1_run, tmp_path, override):
    config, checkpoint = stage1_run
    resumed = {**config, "model": {**config["model"], **override},
               "train": {**config["train"], "n_epochs": 5},
               "data": {"output_dir": str(tmp_path / "resumed")}}
    key = next(iter(override))
    with pytest.raises(ValueError, match=rf"{key}: checkpoint="):
        train_backbone(resumed, resume_checkpoint=checkpoint)
    assert not (tmp_path / "resumed" / "ckpt_best.pt").exists()


def test_resumed_backbone_checkpoint_reload_reproduces_the_trained_model(stage1_run, tmp_path):
    config, checkpoint = stage1_run
    # Ordinary training settings may change on resume.
    resumed_dir = tmp_path / "resumed"
    resumed_cfg = {**config, "train": {**config["train"], "n_epochs": 15},
                   "data": {"output_dir": str(resumed_dir)}}
    best = train_backbone(resumed_cfg, resume_checkpoint=checkpoint)

    saved, reloaded = load_model_from_checkpoint(best)
    original_meta, _ = load_model_from_checkpoint(checkpoint)
    rows = (resumed_dir / "training_log.csv").read_text().splitlines()[1:]
    # Training continues right after the epoch stored in the resumed checkpoint.
    assert [int(r.split(",")[0]) for r in rows] == list(range(original_meta["epoch"] + 1, 16))
    assert saved["epoch"] > original_meta["epoch"], "resumed run should improve"
    assert saved["model_config"]["cutoff"] == reloaded.local_cutoff == 3.0
    assert saved["model_config"] == original_meta["model_config"]
    # The saved config + weights reproduce exactly the loss measured in training.
    assert _val_loss(reloaded, TRAIN) == pytest.approx(saved["val_loss"], rel=1e-5, abs=1e-7)

    # Continue from the resumed best checkpoint once more via the public entry
    # point; reloading its output still reproduces energy and forces.
    best2 = train_backbone(
        {**resumed_cfg, "train": {**resumed_cfg["train"], "n_epochs": saved["epoch"] + 1},
         "data": {"output_dir": str(tmp_path / "again")}},
        resume_checkpoint=best,
    )
    meta2, reloaded2 = load_model_from_checkpoint(best2)
    assert meta2["model_config"] == original_meta["model_config"]
    assert _val_loss(reloaded2, TRAIN) == pytest.approx(meta2["val_loss"], rel=1e-5, abs=1e-7)
    _, forces = _predict(reloaded2)
    assert forces.abs().max() > 0


def test_trainer_resume_saves_config_of_the_actual_model(tmp_path):
    torch.manual_seed(0)
    loader = DataLoader(_make_dry_run_dataset(4), batch_size=2, collate_fn=collate_fn)
    config = backbone_config(cutoff=3.0)
    first = Trainer(model=make_backbone(cutoff=3.0), model_config=config, loss_fn=EnergyForceLoss(),
                    train_loader=loader, val_loader=loader, n_epochs=1, warmup_steps=1,
                    output_dir=str(tmp_path / "first"))
    first.run()

    # atomic_energies is recomputed from data by the entry points; it is taken
    # from the checkpoint (it lives in a buffer), so a different value is fine.
    requested = {**config, "atomic_energies": {1: -13.6}}
    resumed = Trainer.from_checkpoint(
        str(tmp_path / "first" / "ckpt_best.pt"), model_config=requested,
        loss_fn=EnergyForceLoss(), train_loader=loader, val_loader=loader,
        n_epochs=3, warmup_steps=1, output_dir=str(tmp_path / "resumed"),
    )
    assert resumed.model_config == config
    resumed.run()
    resumed.save_checkpoint(3, 0.0, tag="final")
    meta, reloaded = load_model_from_checkpoint(tmp_path / "resumed" / "ckpt_final.pt")
    assert meta["model_config"] == config
    e_ref, f_ref = _predict(resumed.model)
    e_new, f_new = _predict(reloaded)
    torch.testing.assert_close(e_new, e_ref, atol=0, rtol=0)
    torch.testing.assert_close(f_new, f_ref, atol=0, rtol=0)

    with pytest.raises(ValueError, match="cutoff: checkpoint=3.0, requested=2.0"):
        Trainer.from_checkpoint(
            str(tmp_path / "first" / "ckpt_best.pt"), model_config={**config, "cutoff": 2.0},
            loss_fn=EnergyForceLoss(), train_loader=loader, val_loader=loader,
            n_epochs=3, warmup_steps=1, output_dir=str(tmp_path / "bad"),
        )
    from gmd_sgt.models import GMDSGTModel

    with pytest.raises(ValueError, match="cannot resume it as GMDSGTModel"):
        Trainer.from_checkpoint(
            str(tmp_path / "first" / "ckpt_best.pt"), model_cls=GMDSGTModel,
            loss_fn=EnergyForceLoss(), train_loader=loader, val_loader=loader,
            n_epochs=3, warmup_steps=1, output_dir=str(tmp_path / "bad2"),
        )


def test_resumed_residual_training_keeps_nested_config(stage1_run, tmp_path):
    _, backbone_checkpoint = stage1_run
    config = {
        "model": {"type": "GMDSGTModel", "backbone_checkpoint": backbone_checkpoint,
                  "use_gnn": True, "gnn_hidden_channels": 16, "gnn_layers": 1},
        "train": {**TRAIN, "n_epochs": 2, "freeze_backbone": True},
        "data": {"output_dir": str(tmp_path / "s2")},
    }
    first = train_residual(config)
    best = train_residual(
        {**config, "train": {**config["train"], "n_epochs": 4}, "data": {"output_dir": str(tmp_path / "s2b")}},
        resume_checkpoint=first,
    )
    meta, reloaded = load_model_from_checkpoint(best)
    assert meta["model_config"] == load_model_from_checkpoint(first)[0]["model_config"]
    assert _val_loss(reloaded, TRAIN) == pytest.approx(meta["val_loss"], rel=1e-5, abs=1e-7)

    with pytest.raises(ValueError, match="gnn_hidden_channels"):
        train_residual(
            {**config, "model": {**config["model"], "gnn_hidden_channels": 8},
             "data": {"output_dir": str(tmp_path / "bad")}},
            resume_checkpoint=first,
        )


# ── Early stopping state in checkpoints ─────────────────────────────────────


def _scripted_trainer(tmp_path, name, losses, n_epochs, patience, checkpoint=None):
    """Trainer whose epochs return a fixed validation-loss sequence."""
    loader = DataLoader(_make_dry_run_dataset(4), batch_size=2, collate_fn=collate_fn)
    kwargs = dict(loss_fn=EnergyForceLoss(), train_loader=loader, val_loader=loader,
                  n_epochs=n_epochs, warmup_steps=1, patience=patience,
                  output_dir=str(tmp_path / name))
    if checkpoint is None:
        torch.manual_seed(0)
        trainer = Trainer(model=make_backbone(), model_config=backbone_config(), **kwargs)
    else:
        trainer = Trainer.from_checkpoint(str(checkpoint), **kwargs)
    epochs_run = []
    trainer._train_epoch = lambda: {"total": 1.0}

    def val_epoch():
        epoch = trainer.start_epoch + len(epochs_run) + 1
        epochs_run.append(epoch)
        return {"total": losses[epoch - 1]}

    trainer._val_epoch = val_epoch
    trainer.run()
    return trainer, epochs_run


def test_best_checkpoint_stores_early_stopping_state_of_its_epoch(tmp_path):
    losses = [1.0, 2.0, 2.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    _, run = _scripted_trainer(tmp_path, "full", losses, n_epochs=9, patience=3)
    # 0.5 improves at epoch 4; epochs 5-7 do not -> stop at 7.
    assert run == [1, 2, 3, 4, 5, 6, 7]

    best = torch.load(tmp_path / "full" / "ckpt_best.pt", weights_only=False)
    assert best["epoch"] == 4 and best["early_stopping_counter"] == 0
    assert best["best_val"] == 0.5

    # Resuming from the best checkpoint stops at the same epoch as the
    # uninterrupted run (with the stale counter=2 it stopped at epoch 5).
    _, resumed_run = _scripted_trainer(
        tmp_path, "resumed", losses, n_epochs=9, patience=3,
        checkpoint=tmp_path / "full" / "ckpt_best.pt",
    )
    assert resumed_run == [5, 6, 7]


def test_periodic_checkpoint_stores_early_stopping_state_of_its_epoch(tmp_path):
    # Improve until epoch 48, then three non-improving epochs (49, 50, 51).
    losses = [1.0 / k for k in range(1, 49)] + [1.0, 1.0, 1.0, 0.001] + [1.0] * 10
    _, run = _scripted_trainer(tmp_path, "full", losses, n_epochs=60, patience=4)
    assert run[-1] == 56  # 52 improves again, then 53-56 do not
    periodic = torch.load(tmp_path / "full" / "ckpt_epoch0050.pt", weights_only=False)
    assert periodic["epoch"] == 50 and periodic["early_stopping_counter"] == 2

    _, uninterrupted = _scripted_trainer(tmp_path, "full2", losses, n_epochs=60, patience=4)
    _, resumed = _scripted_trainer(
        tmp_path, "resumed", losses, n_epochs=60, patience=4,
        checkpoint=tmp_path / "full" / "ckpt_epoch0050.pt",
    )
    assert resumed == [e for e in uninterrupted if e > 50]


def test_resume_into_new_directory_without_improvement_keeps_best_checkpoint(tmp_path):
    losses = [1.0, 0.5, 0.9, 0.9, 0.9]
    _scripted_trainer(tmp_path, "first", losses, n_epochs=3, patience=0)
    original = torch.load(tmp_path / "first" / "ckpt_best.pt", weights_only=False)
    assert original["epoch"] == 2

    _, run = _scripted_trainer(
        tmp_path, "resumed", losses, n_epochs=5, patience=0,
        checkpoint=tmp_path / "first" / "ckpt_best.pt",
    )
    assert run == [3, 4, 5]
    carried = torch.load(tmp_path / "resumed" / "ckpt_best.pt", weights_only=False)
    assert carried["epoch"] == 2 and carried["val_loss"] == 0.5
    assert carried["model_config"] == original["model_config"]
    for key, value in original["model_state_dict"].items():
        assert torch.equal(carried["model_state_dict"][key], value)
