from __future__ import annotations

import pytest
import torch

from gmd_sgt.model import GMDSGTModel


def _backbone_config() -> dict:
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


def _hybrid_model() -> GMDSGTModel:
    return GMDSGTModel(
        backbone_config=_backbone_config(),
        use_gnn=True,
        gnn_hidden_channels=32,
        gnn_layers=2,
        use_transformer=False,
        lambda_gnn=1.0,
        lambda_attn=0.0,
    )


def test_freeze_backbone_marks_all_backbone_params_frozen():
    model = _hybrid_model()
    model.freeze_backbone()

    assert all(not param.requires_grad for param in model.backbone.parameters())
    assert any(param.requires_grad for param in model.gnn_correction.parameters())


def test_semi_freeze_backbone_keeps_readout_trainable():
    model = _hybrid_model()
    model.semi_freeze_backbone()

    readout_names = []
    frozen_names = []
    for name, param in model.backbone.named_parameters():
        if name.startswith("readout"):
            readout_names.append(name)
            assert param.requires_grad
        else:
            frozen_names.append(name)
            assert not param.requires_grad

    assert readout_names
    assert frozen_names


def _float_state(module):
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
        if torch.is_tensor(tensor) and tensor.is_floating_point()
    }


@pytest.mark.parametrize("policy", ["frozen", "semi_frozen", "unfrozen"])
def test_energy_force_optimizer_step_respects_freeze_policy(policy):
    """A real training step with energy *and* force supervision.

    Needs the force graph to survive until ``loss.backward()`` (previously it
    was freed by ``retain_graph=False``) and must only move trainable params.
    """
    torch.manual_seed(0)
    model = _hybrid_model()
    if policy == "frozen":
        model.freeze_backbone()
    elif policy == "semi_frozen":
        model.semi_freeze_backbone()
    model.train()
    species, positions, batch = _structure()

    backbone_before = _float_state(model.backbone)
    branch_before = _float_state(model.gnn_correction)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    optimizer.zero_grad()
    out = model(species=species, positions=positions, batch=batch, compute_forces=True)
    assert out["forces"].requires_grad, "force graph must be kept for the force loss"
    target_energy = torch.tensor([1.5], dtype=out["energy"].dtype)
    target_forces = torch.full_like(out["forces"], 0.3)
    loss = ((out["energy"] - target_energy) ** 2).mean() + (
        (out["forces"] - target_forces) ** 2
    ).mean()
    loss.backward()

    # The force term alone must reach the trainable parameters.
    grads = [p.grad for p in model.gnn_correction.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    optimizer.step()

    backbone_after = _float_state(model.backbone)
    changed = {k for k in backbone_before if not torch.equal(backbone_before[k], backbone_after[k])}
    if policy == "frozen":
        assert changed == set()
    elif policy == "semi_frozen":
        assert changed and all(k.startswith("readout.") for k in changed)
    else:
        assert any(not k.startswith("readout.") for k in changed)

    branch_after = _float_state(model.gnn_correction)
    assert any(not torch.equal(branch_before[k], branch_after[k]) for k in branch_before)


@pytest.mark.parametrize("model_name", ["backbone", "unified"])
def test_single_model_energy_force_step(model_name):
    from tests._helpers import make_backbone, make_unified

    model = make_backbone() if model_name == "backbone" else make_unified()
    model.train()
    species, positions, batch = _structure()
    before = _float_state(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    out = model(species=species, positions=positions, batch=batch, compute_forces=True)
    loss = (out["energy"] - 1.0).pow(2).mean() + (out["forces"] - 0.2).pow(2).mean()
    loss.backward()
    optimizer.step()
    after = _float_state(model)
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_inference_frees_the_force_graph():
    model = _hybrid_model().eval()
    species, positions, batch = _structure()
    out = model(species=species, positions=positions, batch=batch, compute_forces=True)
    assert not out["forces"].requires_grad
