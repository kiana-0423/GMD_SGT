"""Energy and force continuity when an edge crosses the cutoff radius.

Each configuration is evaluated with one interatomic distance at
``r_c - delta`` and ``r_c + delta``. The smooth polynomial envelope makes every
message (and attention weight) vanish like ``(r_c - r)^3``, so both the energy
jump and the force jump must shrink with ``delta`` and be negligible for tiny
``delta``. Before the fix, biased MLPs left messages non-zero at the cutoff
and the energy jumped by ~1e-4 eV regardless of ``delta``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests._helpers import all_model_factories, evaluate

MODELS = all_model_factories()
CUTOFF = 3.0
DELTAS = (1e-2, 1e-3, 1e-4)


def _dimer(r: float):
    """Two atoms: the only edge (and the last neighbour) crosses the cutoff."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=torch.float64)
    return torch.tensor([6, 8]), positions


def _trimer(r: float):
    """A-B stays bonded while A-C crosses the cutoff (C-B stays outside)."""
    theta = math.radians(100.0)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [r * math.cos(theta), r * math.sin(theta), 0.0],
        ],
        dtype=torch.float64,
    )
    if abs(r - CUTOFF) < 0.1:
        assert (positions[2] - positions[1]).norm() > CUTOFF + 0.1
    return torch.tensor([6, 1, 8]), positions


def _jumps(model, geometry, delta):
    out = []
    for r in (CUTOFF - delta, CUTOFF + delta):
        species, positions = geometry(r)
        batch = torch.zeros(len(species), dtype=torch.long)
        out.append(evaluate(model, species, positions, batch))
    (e_in, f_in), (e_out, f_out) = out
    return float((e_in - e_out).abs().max()), float((f_in - f_out).abs().max())


@pytest.mark.parametrize("geometry", [_dimer, _trimer], ids=["dimer", "trimer"])
@pytest.mark.parametrize("name,factory", MODELS, ids=[n for n, _ in MODELS])
def test_energy_and_forces_continuous_across_cutoff(name, factory, geometry):
    model = factory().double().eval()
    assert getattr(model, "local_cutoff") == CUTOFF

    # The crossing edge must matter inside the cutoff (non-vacuous test).
    species, positions = geometry(CUTOFF - 1.0)
    e_near, _ = evaluate(model, species, positions, torch.zeros(len(species), dtype=torch.long))
    species, positions = geometry(CUTOFF + 0.3)
    e_far, _ = evaluate(model, species, positions, torch.zeros(len(species), dtype=torch.long))
    assert float((e_near - e_far).abs().max()) > 1e-6

    e_jumps, f_jumps = zip(*(_jumps(model, geometry, d) for d in DELTAS))
    # Monotone decrease with the perturbation size ...
    assert e_jumps[0] > e_jumps[1] > e_jumps[2] or e_jumps[2] < 1e-13, e_jumps
    assert f_jumps[0] > f_jumps[1] > f_jumps[2] or f_jumps[2] < 1e-11, f_jumps
    # ... at the rate expected from the C^2 envelope, down to negligible values.
    assert e_jumps[-1] < 1e-10, e_jumps
    assert f_jumps[-1] < 1e-6, f_jumps


# ── float32 ─────────────────────────────────────────────────────────────────


def _residual_with_large_logits(logit: float):
    """Residual model whose sparse attention logits are shifted to ~``logit``."""
    from tests._helpers import make_residual

    model = make_residual()
    with torch.no_grad():
        for layer in model.transformer_correction.layers:
            layer.bias_mlp[-1].bias.fill_(logit)
    return model


FLOAT32_MODELS = [
    ("backbone", lambda: dict(MODELS)["backbone_l2"]()),
    ("unified_inv_attn", lambda: dict(MODELS)["unified_inv_attn"]()),
    ("residual_logit25", lambda: _residual_with_large_logits(25.0)),
    ("residual_logit60", lambda: _residual_with_large_logits(60.0)),
]
# Distances approaching the cutoff from inside (incl. the audited float32
# value 2.99365497) and a few just outside.
_GRID = sorted(
    {CUTOFF - d for d in (3e-2, 1e-2, 6.34503e-3, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6)}
    | {CUTOFF + d for d in (1e-6, 1e-5, 1e-3)}
)


def _evaluate_grid(model, geometry, radii, dtype):
    energies, forces = [], []
    for r in radii:
        species, positions = geometry(r)
        positions = positions.to(torch.float32).to(dtype)  # identical inputs
        e, f = evaluate(model, species, positions, torch.zeros(len(species), dtype=torch.long))
        energies.append(e.double())
        forces.append(f.double())
    return torch.cat(energies), torch.stack(forces)


@pytest.mark.parametrize("geometry", [_dimer, _trimer], ids=["dimer", "trimer"])
@pytest.mark.parametrize("name,factory", FLOAT32_MODELS, ids=[n for n, _ in FLOAT32_MODELS])
def test_float32_energy_and_forces_track_float64_near_cutoff(name, factory, geometry):
    model32 = factory().float().eval()
    model64 = factory().double().eval()  # same seed -> same weights
    e32, f32 = _evaluate_grid(model32, geometry, _GRID, torch.float32)
    e64, f64 = _evaluate_grid(model64, geometry, _GRID, torch.float64)
    assert torch.isfinite(f32).all()

    # Inherent float32 conditioning: interatomic distances computed in float32
    # carry ~1 ulp of rounding, so near a steep (large-logit) switch the exact
    # answer itself moves by this much. Measured in float64 at r +- 4 ulp.
    ulp = torch.finfo(torch.float32).eps * CUTOFF
    e_lo, f_lo = _evaluate_grid(model64, geometry, [r - 4 * ulp for r in _GRID], torch.float64)
    e_hi, f_hi = _evaluate_grid(model64, geometry, [r + 4 * ulp for r in _GRID], torch.float64)
    e_cond = torch.maximum((e_lo - e64).abs(), (e_hi - e64).abs())
    f_cond = torch.maximum((f_lo - f64).abs(), (f_hi - f64).abs()).amax(dim=(1, 2))

    e_scale = 1.0 + float(e64.abs().max())
    f_scale = max(1.0, float(f64.abs().max()))
    # float32 tracks float64 at every distance (no energy jumps, no force
    # spikes). With large logits the exact function is steep near the cutoff;
    # the claim is that float32 reproduces it, not that it is flat.
    e_err = (e32 - e64).abs()
    f_err = (f32 - f64).abs().amax(dim=(1, 2))
    assert (e_err <= 1e-5 * e_scale + e_cond).all(), (e_err, e_cond)
    assert (f_err <= 1e-4 * f_scale + f_cond).all(), (f_err, f_cond)
    assert float(f32.abs().max()) <= float(f64.abs().max()) * 1.001 + 1e-4 + float(f_cond.max())
    # Across the closest pair of points straddling the cutoff, float32 changes
    # as much as the float64 reference.
    i = _GRID.index(CUTOFF - 3e-6)
    jump32_e, jump64_e = e32[i] - e32[i + 1], e64[i] - e64[i + 1]
    jump32_f, jump64_f = f32[i] - f32[i + 1], f64[i] - f64[i + 1]
    assert float((jump32_e - jump64_e).abs()) < 1e-5 * e_scale + float(e_cond[i : i + 2].sum())
    assert float((jump32_f - jump64_f).abs().max()) < 1e-4 * f_scale + float(f_cond[i : i + 2].sum())
