"""TorchScript export: compiled (not traced) models must reproduce eager energy
and forces for every supported model type, across atom and edge counts,
empty edge lists and periodic shifts."""

from __future__ import annotations

import pytest
import torch

from gmd_sgt.inference.export import (
    ExportValidationError,
    export_torchscript,
    script_model,
    validate_exported_model,
)
from gmd_sgt.models import UnifiedEquivariantMLIP
from gmd_sgt.models.geometry import _dense_radius_graph, build_neighbor_graph_pbc
from tests._helpers import (
    all_model_factories,
    backbone_config,
    make_backbone,
    unified_config,
)

MODELS = all_model_factories()


def _config_for(name: str, model) -> tuple[str, dict]:
    if name.startswith("backbone"):
        return "AllegroStyleBackbone", backbone_config(l_max=model.l_max)
    if name.startswith("residual"):
        return "GMDSGTModel", {
            "backbone_config": backbone_config(),
            "use_gnn": True,
            "gnn_hidden_channels": 16,
            "gnn_layers": 1,
            "use_transformer": True,
            "transformer_hidden_channels": 16,
            "transformer_layers": 1,
            "transformer_heads": 4,
        }
    return "UnifiedEquivariantMLIP", unified_config(long_range_type=model.long_range_type)


def _warm_batchnorm(model) -> None:
    """Give e3nn BatchNorm non-trivial running statistics."""
    if isinstance(model, UnifiedEquivariantMLIP):
        model.train()
        generator = torch.Generator().manual_seed(5)
        model(
            torch.randint(1, 9, (8,), generator=generator),
            torch.rand(8, 3, generator=generator) * 3.0,
            torch.zeros(8, dtype=torch.long),
        )
    model.eval()


def _fresh_cases(model):
    """Structures not used by the built-in export validation."""
    generator = torch.Generator().manual_seed(1234)
    cutoff = model.local_cutoff
    cases = []
    for n_atoms in (3, 6, 10):
        pos = torch.rand(n_atoms, 3, generator=generator) * cutoff * 1.1
        edge_index = _dense_radius_graph(pos, cutoff)
        cases.append((n_atoms, pos, edge_index, torch.zeros(edge_index.shape[1], 3)))
    isolated = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    cases.append((2, isolated, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 3)))
    if getattr(model, "long_range_type", "none") != "electrostatic":
        cell = torch.eye(3) * cutoff * 0.9
        pos = torch.rand(5, 3, generator=generator) @ cell
        edge_index, edge_shift = build_neighbor_graph_pbc(pos, cell, cutoff)
        assert edge_shift.abs().sum() > 0
        cases.append((5, pos, edge_index, edge_shift))
    return [
        (n, torch.randint(1, 9, (n,), generator=generator), pos, ei, es)
        for n, pos, ei, es in cases
    ]


@pytest.mark.parametrize("name,factory", MODELS, ids=[n for n, _ in MODELS])
def test_export_matches_eager_energy_and_forces(name, factory, tmp_path):
    model = factory()
    _warm_batchnorm(model)
    model_type, model_config = _config_for(name, model)
    checkpoint = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_type": model_type,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "format_version": 2,
        },
        checkpoint,
    )
    output = tmp_path / "model.pt"

    report = export_torchscript(str(checkpoint), str(output))
    assert report and any(r["n_edges"] == 0 for r in report.values())
    assert len({r["n_atoms"] for r in report.values()}) >= 3

    loaded = torch.jit.load(str(output))
    args = [a.name for a in loaded.forward.schema.arguments][1:]
    assert args == ["species", "positions", "edge_index", "edge_shift"]
    assert loaded.local_cutoff == pytest.approx(model.local_cutoff)

    nontrivial = False
    for n_atoms, species, pos, edge_index, edge_shift in _fresh_cases(model):
        batch = torch.zeros(n_atoms, dtype=torch.long)
        eager = model(
            species=species, positions=pos.clone(), batch=batch,
            edge_index=edge_index, edge_shift=edge_shift,
        )
        out = loaded(species, pos.clone(), edge_index, edge_shift)
        assert out["energy"].dtype == torch.float64 and out["energy"].shape == (1,)
        assert out["forces"].shape == (n_atoms, 3)
        torch.testing.assert_close(
            out["energy"], eager["energy"].detach().double(), atol=1e-4, rtol=1e-5
        )
        torch.testing.assert_close(out["forces"], eager["forces"].detach(), atol=1e-4, rtol=1e-4)
        if edge_index.shape[1] > 0 and eager["forces"].abs().max() > 1e-3:
            nontrivial = True
        if edge_index.shape[1] == 0 and not isinstance(model, UnifiedEquivariantMLIP):
            assert torch.count_nonzero(out["forces"]) == 0
    assert nontrivial, "forces vanish for every case; the comparison would be vacuous"


def test_validation_rejects_wrong_forces():
    model = make_backbone().eval()
    scripted = script_model(model)

    class ZeroForces(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, species, positions, edge_index, edge_shift):
            out = self.inner(species, positions, edge_index, edge_shift)
            return {"energy": out["energy"], "forces": torch.zeros_like(out["forces"])}

    with pytest.raises(ExportValidationError, match="forces|dF"):
        validate_exported_model(ZeroForces(scripted), model)


def test_exported_model_requires_grad_mode():
    scripted = script_model(make_backbone().eval())
    species = torch.tensor([1, 6])
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    with torch.no_grad(), pytest.raises(Exception, match="gradient mode"):
        scripted(species, pos, edge_index, torch.zeros(2, 3))


# ── Electrostatic periodic inputs ───────────────────────────────────────────


def test_exported_electrostatic_model_rejects_periodic_shifts(tmp_path):
    from tests._helpers import make_unified

    model = make_unified(long_range_type="electrostatic").eval()
    checkpoint = tmp_path / "elec.pt"
    torch.save(
        {
            "model_type": "UnifiedEquivariantMLIP",
            "model_config": unified_config(long_range_type="electrostatic"),
            "model_state_dict": model.state_dict(),
            "format_version": 2,
        },
        checkpoint,
    )
    output = tmp_path / "elec_model.pt"
    export_torchscript(str(checkpoint), str(output))  # validated non-periodic export
    loaded = torch.jit.load(str(output))

    species = torch.tensor([1, 8])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    batch = torch.zeros(2, dtype=torch.long)
    periodic_shift = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="non-periodic"):
        model(species, positions.clone(), batch, edge_index=edge_index, edge_shift=periodic_shift)
    with pytest.raises(Exception, match="non-periodic"):
        loaded(species, positions.clone(), edge_index, periodic_shift)

    # Zero shifts (non-periodic): eager and exported agree.
    zero_shift = torch.zeros(2, 3)
    eager = model(species, positions.clone(), batch, edge_index=edge_index, edge_shift=zero_shift)
    out = loaded(species, positions.clone(), edge_index, zero_shift)
    torch.testing.assert_close(out["energy"], eager["energy"].detach().double())
    torch.testing.assert_close(out["forces"], eager["forces"].detach())
    assert eager["forces"].abs().max() > 1e-4
    # Still usable after a rejected call (no stale state in the scripted module).
    out_again = loaded(species, positions.clone(), edge_index, zero_shift)
    torch.testing.assert_close(out_again["energy"], out["energy"])


def test_non_electrostatic_exports_still_accept_periodic_shifts():
    from tests._helpers import make_unified

    scripted = script_model(make_unified(long_range_type="invariant_attention").eval())
    out = scripted(
        torch.tensor([1, 8]),
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    )
    assert torch.isfinite(out["energy"]).all()


# ── Envelope-weighted attention in exported residual models ─────────────────


@pytest.mark.parametrize("logit", [0.0, 25.0, 60.0])
def test_exported_attention_ignores_edges_beyond_the_cutoff(tmp_path, logit):
    """Edges with zero envelope (beyond the cutoff) are equivalent to absent
    edges, in eager and in the saved/reloaded TorchScript model."""
    from tests._helpers import make_residual

    model = make_residual()
    with torch.no_grad():
        for layer in model.transformer_correction.layers:
            layer.bias_mlp[-1].bias.fill_(logit)
    model.eval()
    path = tmp_path / "residual.pt"
    script_model(model).save(str(path))
    loaded = torch.jit.load(str(path))

    species = torch.tensor([6, 1, 8, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 3.4, 0.0], [-0.3, 1.0, 0.2]]
    )
    cutoff = model.local_cutoff
    inside = _dense_radius_graph(positions, cutoff)
    # Add every pair beyond the cutoff (atom 2 is > cutoff from 0 and 1).
    far = _dense_radius_graph(positions, 10.0)
    far = far[:, (positions[far[1]] - positions[far[0]]).norm(dim=-1) >= cutoff]
    assert far.shape[1] > 0
    padded = torch.cat([inside, far], dim=1)
    batch = torch.zeros(4, dtype=torch.long)

    results = []
    for edges in (inside, padded):
        shifts = torch.zeros(edges.shape[1], 3)
        eager = model(species, positions.clone(), batch, edge_index=edges, edge_shift=shifts)
        exported = loaded(species, positions.clone(), edges, shifts)
        torch.testing.assert_close(exported["energy"], eager["energy"].detach().double())
        torch.testing.assert_close(exported["forces"], eager["forces"].detach(), atol=1e-5, rtol=1e-5)
        assert torch.isfinite(exported["forces"]).all()
        results.append(exported)
    torch.testing.assert_close(results[1]["energy"], results[0]["energy"])
    torch.testing.assert_close(results[1]["forces"], results[0]["forces"], atol=1e-6, rtol=1e-6)
