"""Shared helpers for the test-suite (models, rotations, structures)."""

from __future__ import annotations

import torch

from gmd_sgt.models import AllegroStyleBackbone, GMDSGTModel, UnifiedEquivariantMLIP
from gmd_sgt.models.dependencies import E3NN_AVAILABLE

# Non-scalar irreps need e3nn; without it the unified model only supports scalars.
UNIFIED_IRREPS = "8x0e + 4x1o + 2x2e" if E3NN_AVAILABLE else "8x0e"


def backbone_config(**overrides) -> dict:
    cfg = dict(
        n_species=10,
        hidden_channels=16,
        num_layers=2,
        n_basis=4,
        cutoff=3.0,
        l_max=2,
        avg_neighbors=4.0,
    )
    cfg.update(overrides)
    return cfg


def unified_config(**overrides) -> dict:
    cfg = dict(
        n_species=10,
        n_blocks=2,
        scalar_dim=8,
        irreps=UNIFIED_IRREPS,
        n_basis=4,
        local_cutoff=3.0,
        lr_cutoff=6.0,
        l_max=2,
        long_range_type="none",
        n_heads=2,
        avg_neighbors=4.0,
    )
    cfg.update(overrides)
    return cfg


def make_backbone(seed: int = 0, **overrides) -> AllegroStyleBackbone:
    torch.manual_seed(seed)
    model = AllegroStyleBackbone(**backbone_config(**overrides))
    randomize_last_linear(model.readout.net, seed)
    return model


def make_unified(seed: int = 0, **overrides) -> UnifiedEquivariantMLIP:
    """Unified model with a *non-zero* energy head (the default init is zero)."""
    torch.manual_seed(seed)
    model = UnifiedEquivariantMLIP(**unified_config(**overrides))
    randomize_energy_head(model, seed)
    return model


def make_residual(seed: int = 0, use_transformer: bool = True, **backbone_overrides) -> GMDSGTModel:
    torch.manual_seed(seed)
    model = GMDSGTModel(
        backbone_config=backbone_config(**backbone_overrides),
        use_gnn=True,
        gnn_hidden_channels=16,
        gnn_layers=1,
        use_transformer=use_transformer,
        transformer_hidden_channels=16,
        transformer_layers=1,
        transformer_heads=4,
        lambda_gnn=1.0,
        lambda_attn=1.0,
    )
    randomize_last_linear(model.backbone.readout.net, seed)
    randomize_last_linear(model.gnn_correction.readout.net, seed + 2)
    if model.transformer_correction is not None:
        randomize_last_linear(model.transformer_correction.readout.net, seed + 3)
    return model


def randomize_last_linear(sequential: torch.nn.Sequential, seed: int = 0) -> None:
    """Give a readout MLP a clearly non-zero final layer."""
    generator = torch.Generator().manual_seed(seed + 7)
    layer = sequential[-1]
    with torch.no_grad():
        layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator))
        layer.bias.zero_()


def randomize_energy_head(model: UnifiedEquivariantMLIP, seed: int = 0) -> None:
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for layer in model.energy_head:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * 0.5)
                layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator) * 0.1)


def random_orthogonal(seed: int = 0, proper: bool = True, dtype=torch.float64) -> torch.Tensor:
    """Random rotation (det=+1) or improper rotation/reflection (det=-1)."""
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if (torch.det(q) > 0) != proper:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)


def random_cluster(n_atoms: int = 6, seed: int = 0, scale: float = 1.2, dtype=torch.float64):
    """Compact random cluster (all atoms within a few Å) and species."""
    generator = torch.Generator().manual_seed(seed)
    positions = torch.randn(n_atoms, 3, generator=generator, dtype=dtype) * scale
    species = torch.randint(1, 9, (n_atoms,), generator=generator)
    batch = torch.zeros(n_atoms, dtype=torch.long)
    return species, positions, batch


def evaluate(model, species, positions, batch, **kwargs):
    """Energy [G] and forces [N, 3] from the eager model."""
    out = model(
        species=species,
        positions=positions.clone().requires_grad_(True),
        batch=batch,
        compute_forces=True,
        **kwargs,
    )
    return out["energy"].detach(), out["forces"].detach()


def all_model_factories():
    """(name, factory) pairs covering every model family."""
    return [
        ("backbone_l0", lambda: make_backbone(l_max=0)),
        ("backbone_l1", lambda: make_backbone(l_max=1)),
        ("backbone_l2", lambda: make_backbone(l_max=2)),
        ("residual_gnn_attn", lambda: make_residual()),
        ("unified_none", lambda: make_unified(long_range_type="none")),
        ("unified_inv_attn", lambda: make_unified(long_range_type="invariant_attention")),
        ("unified_eq_attn", lambda: make_unified(long_range_type="equivariant_attention")),
        ("unified_electrostatic", lambda: make_unified(long_range_type="electrostatic")),
    ]
