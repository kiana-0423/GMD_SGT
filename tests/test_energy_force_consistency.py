from __future__ import annotations

import torch

from gmd_sgt.model import AllegroStyleBackbone, GMDSGTModel


def test_backbone_forces_are_negative_energy_gradient():
    model = AllegroStyleBackbone(
        n_species=10,
        hidden_channels=24,
        num_layers=2,
        n_basis=4,
        cutoff=4.0,
        l_max=1,
        avg_neighbors=4.0,
    )
    model.eval()

    species = torch.tensor([8, 1, 1, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    positions_force = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [-0.2, 0.85, 0.0],
            [0.0, -0.9, 0.1],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    positions_grad = positions_force.detach().clone().requires_grad_(True)

    out_force = model(
        species=species,
        positions=positions_force,
        batch=batch,
        compute_forces=True,
    )
    out_energy = model(
        species=species,
        positions=positions_grad,
        batch=batch,
        compute_forces=False,
    )
    manual_forces = -torch.autograd.grad(
        out_energy["energy"].sum(),
        positions_grad,
        create_graph=False,
        retain_graph=False,
    )[0]

    assert torch.allclose(out_force["forces"], manual_forces, atol=1e-6, rtol=1e-5)


def test_residual_model_forces_are_negative_total_energy_gradient():
    model = GMDSGTModel(
        backbone_config={
            "n_species": 10,
            "hidden_channels": 24,
            "num_layers": 2,
            "n_basis": 4,
            "cutoff": 4.0,
            "l_max": 1,
            "avg_neighbors": 4.0,
        },
        use_gnn=True,
        gnn_hidden_channels=24,
        gnn_layers=2,
        use_transformer=True,
        transformer_hidden_channels=24,
        transformer_layers=1,
        transformer_heads=4,
        lambda_gnn=1.0,
        lambda_attn=1.0,
    )
    model.eval()

    species = torch.tensor([8, 1, 1, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    positions_force = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [-0.2, 0.85, 0.0],
            [0.0, -0.9, 0.1],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    positions_grad = positions_force.detach().clone().requires_grad_(True)

    out_force = model(
        species=species,
        positions=positions_force,
        batch=batch,
        compute_forces=True,
    )
    out_energy = model(
        species=species,
        positions=positions_grad,
        batch=batch,
        compute_forces=False,
    )
    manual_forces = -torch.autograd.grad(
        out_energy["energy"].sum(),
        positions_grad,
        create_graph=False,
        retain_graph=False,
    )[0]

    assert torch.allclose(out_force["forces"], manual_forces, atol=1e-6, rtol=1e-5)
