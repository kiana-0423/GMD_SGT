from __future__ import annotations

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


def test_frozen_backbone_parameters_do_not_update_after_step():
    model = _hybrid_model()
    model.freeze_backbone()
    species, positions, batch = _structure()

    backbone_before = {
        name: tensor.detach().clone()
        for name, tensor in model.backbone.state_dict().items()
        if torch.is_tensor(tensor)
        and (tensor.is_floating_point() or tensor.is_complex())
    }
    branch_before = {
        name: tensor.detach().clone()
        for name, tensor in model.gnn_correction.state_dict().items()
        if torch.is_tensor(tensor)
        and (tensor.is_floating_point() or tensor.is_complex())
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    optimizer.zero_grad()
    out = model(
        species=species,
        positions=positions,
        batch=batch,
        compute_forces=True,
    )
    target_energy = torch.tensor([1.5], dtype=out["energy"].dtype)
    loss = ((out["energy"] - target_energy) ** 2).mean() + 0.1 * out["forces"].pow(2).mean()
    loss.backward()
    optimizer.step()

    for name, tensor in model.backbone.state_dict().items():
        if name in backbone_before:
            assert torch.allclose(tensor, backbone_before[name])

    assert any(
        not torch.allclose(tensor, branch_before[name])
        for name, tensor in model.gnn_correction.state_dict().items()
        if name in branch_before
    )
