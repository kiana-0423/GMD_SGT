"""l_max / irreps consistency and spherical-harmonic bases."""

from __future__ import annotations

import pytest
import torch

from gmd_sgt.models import AllegroStyleBackbone, UnifiedEquivariantMLIP
from gmd_sgt.models.dependencies import E3NN_AVAILABLE
from gmd_sgt.models.geometry import cartesian_spherical_harmonics, directional_basis
from tests._helpers import evaluate, random_cluster, random_orthogonal, unified_config

needs_e3nn = pytest.mark.skipif(not E3NN_AVAILABLE, reason="requires e3nn")


def _unit_vectors(n=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(
        torch.randn(n, 3, generator=generator, dtype=torch.float64), dim=-1
    )


@pytest.mark.parametrize("l_max", [0, 1, 2])
def test_cartesian_harmonics_have_rotation_invariant_norms_per_l(l_max):
    """Without e3nn the l=2 basis must be normalised so |sum_e w_e Y_l(e)|^2 is invariant."""
    unit = _unit_vectors()
    weights = torch.randn(unit.shape[0], dtype=torch.float64)
    for proper in (True, False):
        rot = random_orthogonal(seed=3, proper=proper)
        before = directional_basis(unit, l_max)
        after = directional_basis(unit @ rot.T, l_max)
        for ell, (b0, b1) in enumerate(zip(before, after)):
            assert b0.shape[-1] == 2 * ell + 1
            n0 = ((weights[:, None] * b0).sum(0) ** 2).sum()
            n1 = ((weights[:, None] * b1).sum(0) ** 2).sum()
            torch.testing.assert_close(n1, n0)
            # component normalisation: |Y_l(u)|^2 = 2l + 1 for unit u
            torch.testing.assert_close(
                (b0 ** 2).sum(-1), torch.full((unit.shape[0],), 2.0 * ell + 1, dtype=torch.float64)
            )


@needs_e3nn
def test_cartesian_harmonics_equal_e3nn():
    from e3nn import o3

    unit = _unit_vectors()
    expected = o3.spherical_harmonics([0, 1, 2], unit, normalize=True, normalization="component")
    torch.testing.assert_close(cartesian_spherical_harmonics(unit, 2), expected)


UNIFIED_LMAX_IRREPS = {0: "8x0e", 1: "8x0e + 4x1o + 2x1e", 2: "8x0e + 4x1o + 2x2e"}


@needs_e3nn
@pytest.mark.parametrize("l_max", [0, 1, 2])
def test_unified_model_uses_harmonics_consistent_with_l_max(l_max):
    from e3nn import o3

    torch.manual_seed(0)
    model = UnifiedEquivariantMLIP(**unified_config(l_max=l_max, irreps=UNIFIED_LMAX_IRREPS[l_max]))
    for block in model.blocks:
        assert block.local_mp.tp.irreps_in2 == o3.Irreps.spherical_harmonics(l_max)
    with torch.no_grad():
        model.energy_head[-1].weight.normal_()
    model = model.double().eval()

    species, pos, batch = random_cluster(6, seed=1)
    e0, f0 = evaluate(model, species, pos, batch)
    assert f0.abs().max() > 1e-4
    for proper in (True, False):
        rot = random_orthogonal(seed=5, proper=proper)
        e1, f1 = evaluate(model, species, pos @ rot.T, batch)
        torch.testing.assert_close(e1, e0)
        torch.testing.assert_close(f1, f0 @ rot.T)


@needs_e3nn
def test_vector_channels_start_at_zero_and_are_populated_by_message_passing():
    torch.manual_seed(0)
    model = UnifiedEquivariantMLIP(**unified_config(n_blocks=1)).double().eval()
    species = torch.tensor([1, 6, 8])
    h0 = model.input_proj(model.species_embedding(species).double()) * model.input_irreps_mask
    assert torch.count_nonzero(h0[:, 8:]) == 0  # all l > 0 components
    assert torch.count_nonzero(h0[:, :8]) > 0


@needs_e3nn
@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(l_max=0, irreps="8x0e + 4x1o"), "never be populated"),
        (dict(l_max=1, n_blocks=1, irreps="8x0e + 4x1o + 2x2e"), "never be populated"),
        (dict(irreps="4x1o + 8x0e"), "start with at least"),
        (dict(scalar_dim=16, irreps="8x0e + 4x1o"), "start with at least"),
    ],
)
def test_unified_rejects_inconsistent_irreps(overrides, match):
    with pytest.raises(ValueError, match=match):
        UnifiedEquivariantMLIP(**unified_config(**overrides))


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(l_max=-1), "l_max"),
        (dict(l_max=1.5), "l_max"),
        (dict(long_range_type="ewald"), "long_range_type"),
        (dict(local_cutoff=0.0), "local_cutoff"),
    ],
)
def test_unified_rejects_invalid_configuration(overrides, match):
    with pytest.raises(ValueError, match=match):
        UnifiedEquivariantMLIP(**unified_config(**overrides))


@needs_e3nn
def test_backbone_supports_l_max_above_two_with_e3nn():
    torch.manual_seed(0)
    model = AllegroStyleBackbone(n_species=10, hidden_channels=8, num_layers=1, n_basis=4,
                                 cutoff=3.0, l_max=3).double().eval()
    species, pos, batch = random_cluster(5, seed=2)
    e0, f0 = evaluate(model, species, pos, batch)
    rot = random_orthogonal(seed=1, proper=False)
    e1, f1 = evaluate(model, species, pos @ rot.T, batch)
    torch.testing.assert_close(e1, e0)
    torch.testing.assert_close(f1, f0 @ rot.T)


def test_backbone_rejects_invalid_l_max():
    with pytest.raises(ValueError, match="l_max"):
        AllegroStyleBackbone(l_max=-1)
