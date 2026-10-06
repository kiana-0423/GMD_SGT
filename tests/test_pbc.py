"""Periodic boundary conditions: real ASE neighbor lists, per-axis PBC through
data loading, collation, training, prediction and ASE, and stress."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests._helpers import (
    evaluate,
    make_backbone,
    make_residual,
    make_unified,
    unified_config,
)

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.neighborlist import neighbor_list as ase_neighbor_list  # noqa: E402

from gmd_sgt.data import AtomicDataset, collate_fn, read_extxyz  # noqa: E402
from gmd_sgt.models.geometry import (  # noqa: E402
    build_neighbor_graph,
    build_neighbor_graph_pbc,
    normalize_pbc,
)

CUTOFF = 3.0
TRICLINIC = np.array([[2.6, 0.0, 0.0], [0.7, 2.4, 0.0], [0.3, -0.4, 2.8]])


def _positions(n: int = 4, seed: int = 0, cell=TRICLINIC) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 3)) @ cell


def _edge_set(edge_index, edge_shift) -> set:
    ei = np.asarray(edge_index)
    es = np.round(np.asarray(edge_shift, dtype=np.float64), 5)
    return {(int(i), int(j), *map(float, s)) for i, j, s in zip(ei[0], ei[1], es)}


def _ase_edge_set(positions, cell, pbc, cutoff=CUTOFF) -> set:
    atoms = Atoms("H" * len(positions), positions=positions, cell=cell, pbc=pbc)
    i, j, shifts = ase_neighbor_list("ijS", atoms, cutoff)
    return _edge_set(np.stack([i, j]), shifts @ np.asarray(cell))


# ── Neighbor construction against ASE ────────────────────────────────────────


@pytest.mark.parametrize(
    "pbc",
    [(True, True, True), (True, True, False), (True, False, False), (False, True, True)],
    ids=["full", "slab_xy", "wire_x", "slab_yz"],
)
def test_periodic_graph_matches_ase(pbc):
    positions = _positions(5, seed=1)
    edge_index, edge_shift = build_neighbor_graph_pbc(
        torch.tensor(positions), torch.tensor(TRICLINIC), CUTOFF, torch.tensor(pbc)
    )
    expected = _ase_edge_set(positions, TRICLINIC, pbc)
    assert len(expected) > 0
    assert _edge_set(edge_index, edge_shift) == expected
    # Small cell: atoms see several periodic images of themselves when periodic.
    if all(pbc):
        assert any(i == j for i, j, *_ in expected)
    # No shift along non-periodic axes (fractional shift component is zero).
    frac = np.asarray(edge_shift) @ np.linalg.inv(TRICLINIC)
    for axis, periodic in enumerate(pbc):
        if not periodic:
            assert np.allclose(frac[:, axis], 0.0)


def test_unwrapped_positions_give_same_graph_geometry():
    positions = _positions(4, seed=2)
    unwrapped = positions.copy()
    unwrapped[1] += TRICLINIC[0] - 2 * TRICLINIC[2]
    e0 = build_neighbor_graph_pbc(torch.tensor(positions), torch.tensor(TRICLINIC), CUTOFF)
    e1 = build_neighbor_graph_pbc(torch.tensor(unwrapped), torch.tensor(TRICLINIC), CUTOFF)

    def distances(pos, graph):
        ei, es = graph
        vec = torch.tensor(pos)[ei[1]] - torch.tensor(pos)[ei[0]] + es
        return sorted(np.round(vec.norm(dim=-1).numpy(), 8).tolist())

    assert distances(positions, e0) == distances(unwrapped, e1)


def test_scalar_and_default_pbc_are_fully_periodic():
    positions = torch.tensor(_positions(4, seed=3))
    cell = torch.tensor(TRICLINIC)
    reference = _edge_set(*build_neighbor_graph_pbc(positions, cell, CUTOFF, [True] * 3))
    for pbc in (None, True, torch.tensor(True), np.bool_(True), [1, 1, 1]):
        assert _edge_set(*build_neighbor_graph_pbc(positions, cell, CUTOFF, pbc)) == reference
    np.testing.assert_array_equal(normalize_pbc(False, 2).numpy(), np.zeros((2, 3), dtype=bool))


def test_cell_with_all_axes_nonperiodic_is_open_boundary():
    positions = torch.tensor(_positions(5, seed=4))
    batch = torch.zeros(5, dtype=torch.long)
    with_cell = build_neighbor_graph(positions, batch, CUTOFF, torch.tensor(TRICLINIC), [False] * 3)
    without_cell = build_neighbor_graph(positions, batch, CUTOFF)
    expected = _ase_edge_set(positions.numpy(), TRICLINIC, (False, False, False))
    assert _edge_set(*with_cell) == expected
    assert _edge_set(without_cell[0], torch.zeros(without_cell[0].shape[1], 3)) == expected


def test_batched_graph_with_mixed_pbc_matches_individual_ase_lists():
    pos0, pos1 = _positions(4, seed=5), _positions(3, seed=6)
    cell1 = TRICLINIC * 1.1
    pbc = torch.tensor([[True, True, True], [True, False, True]])
    edge_index, edge_shift = build_neighbor_graph(
        torch.tensor(np.concatenate([pos0, pos1])),
        torch.tensor([0] * 4 + [1] * 3),
        CUTOFF,
        torch.tensor(np.stack([TRICLINIC, cell1])),
        pbc,
    )
    expected = _ase_edge_set(pos0, TRICLINIC, (True, True, True))
    expected |= {
        (i + 4, j + 4, *s) for i, j, *s in _ase_edge_set(pos1, cell1, (True, False, True))
    }
    assert _edge_set(edge_index, edge_shift) == expected
    # No edge crosses between the two structures.
    assert not ((edge_index[0] < 4) ^ (edge_index[1] < 4)).any()


# ── Model energies with PBC ──────────────────────────────────────────────────

PERIODIC_MODELS = [
    ("backbone", lambda: make_backbone()),
    ("residual", lambda: make_residual()),
    ("unified_none", lambda: make_unified(long_range_type="none")),
    ("unified_inv_attn", lambda: make_unified(long_range_type="invariant_attention")),
]


@pytest.fixture(params=PERIODIC_MODELS, ids=[n for n, _ in PERIODIC_MODELS])
def periodic_model(request):
    return request.param[1]().double().eval()


def _structure(n=4, seed=7, cell=TRICLINIC):
    pos = torch.tensor(_positions(n, seed=seed, cell=cell))
    species = torch.tensor([1, 6, 8, 1, 6, 8, 1, 6][:n])
    return species, pos, torch.zeros(n, dtype=torch.long), torch.tensor(cell)


def test_energy_invariant_to_wrapping_atoms(periodic_model):
    species, pos, batch, cell = _structure()
    e0, f0 = evaluate(periodic_model, species, pos, batch, cell=cell)
    assert f0.abs().max() > 1e-4
    moved = pos.clone()
    moved[2] += cell[0] - cell[1] + 2 * cell[2]
    e1, f1 = evaluate(periodic_model, species, moved, batch, cell=cell)
    torch.testing.assert_close(e1, e0)
    torch.testing.assert_close(f1, f0)


@pytest.mark.parametrize("name", ["backbone", "residual", "unified_none"])
def test_local_models_are_extensive_over_supercells(name):
    model = dict(PERIODIC_MODELS)[name]().double().eval()
    species, pos, batch, cell = _structure()
    e1, f1 = evaluate(model, species, pos, batch, cell=cell)

    super_pos = torch.cat([pos, pos + cell[0]])
    super_species = torch.cat([species, species])
    super_cell = cell.clone()
    super_cell[0] *= 2
    e2, f2 = evaluate(
        model, super_species, super_pos, torch.zeros(8, dtype=torch.long), cell=super_cell
    )
    torch.testing.assert_close(e2, 2 * e1)
    torch.testing.assert_close(f2, torch.cat([f1, f1]))


def test_partial_pbc_is_periodic_only_along_flagged_axes(periodic_model):
    species, pos, batch, cell = _structure()
    slab = torch.tensor([True, True, False])
    e_slab, _ = evaluate(periodic_model, species, pos, batch, cell=cell, pbc=slab)
    e_full, _ = evaluate(periodic_model, species, pos, batch, cell=cell)
    assert not torch.allclose(e_slab, e_full), "c is shorter than the cutoff; z-PBC must matter"

    along_a = pos.clone()
    along_a[0] += cell[0]
    e_a, _ = evaluate(periodic_model, species, along_a, batch, cell=cell, pbc=slab)
    torch.testing.assert_close(e_a, e_slab)

    along_c = pos.clone()
    along_c[0] += cell[2]
    e_c, _ = evaluate(periodic_model, species, along_c, batch, cell=cell, pbc=slab)
    assert not torch.allclose(e_c, e_slab)


def test_all_false_pbc_matches_cell_free_evaluation(periodic_model):
    species, pos, batch, cell = _structure()
    e_open, f_open = evaluate(periodic_model, species, pos, batch)
    e_flag, f_flag = evaluate(periodic_model, species, pos, batch, cell=cell, pbc=[False] * 3)
    torch.testing.assert_close(e_flag, e_open)
    torch.testing.assert_close(f_flag, f_open)


def test_electrostatic_rejects_periodic_input():
    model = make_unified(long_range_type="electrostatic").double().eval()
    species, pos, batch, cell = _structure()
    with pytest.raises(ValueError, match="non-periodic"):
        evaluate(model, species, pos, batch, cell=cell)


# ── Stress ───────────────────────────────────────────────────────────────────

STRESS_MODELS = PERIODIC_MODELS


@pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False)], ids=["full", "slab"])
@pytest.mark.parametrize("name", [n for n, _ in STRESS_MODELS])
def test_stress_matches_strain_finite_difference(name, pbc):
    model = dict(STRESS_MODELS)[name]().double().eval()
    species, pos, batch, cell = _structure()
    pbc_t = torch.tensor(pbc)
    out = model(
        species=species,
        positions=pos.clone(),
        batch=batch,
        cell=cell,
        pbc=pbc_t,
        compute_forces=True,
        compute_stress=True,
    )
    stress = out["stress"].detach()[0]
    assert stress.shape == (3, 3)
    assert stress.abs().max() > 1e-5
    torch.testing.assert_close(stress, stress.T)

    volume = torch.linalg.det(cell).abs()
    eps = 1e-6
    fd = torch.zeros(3, 3, dtype=torch.float64)
    for a in range(3):
        for b in range(3):
            def energy(sign):
                strain = torch.zeros(3, 3, dtype=torch.float64)
                strain[a, b] += sign * eps / 2
                strain[b, a] += sign * eps / 2
                deform = torch.eye(3, dtype=torch.float64) + strain
                e, _ = evaluate(
                    model, species, pos @ deform, batch, cell=cell @ deform, pbc=pbc_t
                )
                return e.sum()

            fd[a, b] = (energy(1) - energy(-1)) / (2 * eps) / volume
    torch.testing.assert_close(stress, fd, atol=1e-7, rtol=1e-5)


def test_stress_requires_periodic_cell():
    model = make_backbone().double().eval()
    species, pos, batch, cell = _structure()
    with pytest.raises(ValueError, match="periodic cell"):
        model(species=species, positions=pos, batch=batch, compute_stress=True)
    with pytest.raises(ValueError, match="periodic cell"):
        model(
            species=species, positions=pos, batch=batch, cell=cell,
            pbc=torch.tensor([False] * 3), compute_stress=True,
        )


# ── Data pipeline, collation, Trainer, calculators ──────────────────────────


def _write_extxyz(path, pbcs):
    from ase.io import write

    frames = []
    for k, pbc in enumerate(pbcs):
        atoms = Atoms("HCO", positions=_positions(3, seed=10 + k), cell=TRICLINIC, pbc=pbc)
        atoms.info["energy"] = -1.0 - k
        atoms.arrays["forces"] = np.zeros((3, 3))
        frames.append(atoms)
    write(str(path), frames, format="extxyz")


def test_reader_and_collate_preserve_per_axis_pbc(tmp_path):
    path = tmp_path / "mixed.extxyz"
    _write_extxyz(path, [(True, True, False), (True, False, True)])
    items = read_extxyz(str(path))
    assert items[0]["pbc"].tolist() == [True, True, False]
    assert items[1]["pbc"].tolist() == [True, False, True]

    batch = collate_fn(items)
    assert batch["pbc"].tolist() == [[True, True, False], [True, False, True]]
    assert batch["cell"].shape == (2, 3, 3)

    legacy = dict(items[0])
    legacy["pbc"] = torch.tensor(True)  # scalar flag from older pipelines
    assert collate_fn([legacy])["pbc"].tolist() == [[True, True, True]]


def test_npz_reader_pbc_and_virial(tmp_path):
    from gmd_sgt.data import read_npz

    cell = np.stack([TRICLINIC, TRICLINIC])
    virial = np.stack([np.eye(3), 2 * np.eye(3)])
    path = tmp_path / "data.npz"
    np.savez(
        path,
        R=np.stack([_positions(3, 1), _positions(3, 2)]),
        Z=np.array([[1, 6, 8]] * 2),
        E=np.array([-1.0, -2.0]),
        F=np.zeros((2, 3, 3)),
        cell=cell,
        pbc=np.array([[True, True, False], [True, True, True]]),
        virial=virial,
    )
    items = read_npz(str(path))
    assert items[0]["pbc"].tolist() == [True, True, False]
    volume = abs(np.linalg.det(TRICLINIC))
    np.testing.assert_allclose(items[1]["stress"].numpy(), -2 * np.eye(3) / volume, rtol=1e-6)


def test_trainer_forwards_per_axis_pbc_to_model(tmp_path):
    from torch.utils.data import DataLoader

    from gmd_sgt.training import EnergyForceLoss, Trainer

    seen = []

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = make_backbone()

        def forward(self, **kwargs):
            seen.append(kwargs.get("pbc"))
            return self.inner(**kwargs)

    path = tmp_path / "slab.extxyz"
    _write_extxyz(path, [(True, True, False)] * 2)
    loader = DataLoader(AtomicDataset(read_extxyz(str(path))), batch_size=2, collate_fn=collate_fn)
    trainer = Trainer(
        model=Recorder(), model_config={}, loss_fn=EnergyForceLoss(w_stress=0.0),
        train_loader=loader, val_loader=loader, n_epochs=1, warmup_steps=1,
        output_dir=str(tmp_path / "run"),
    )
    trainer.run()
    assert seen and all(p.tolist() == [[True, True, False]] * 2 for p in seen)


def _save_checkpoint(model, model_type, model_config, path):
    torch.save(
        {
            "model_type": model_type,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "format_version": 2,
        },
        path,
    )
    return str(path)


@pytest.fixture()
def unified_checkpoint(tmp_path):
    model = make_unified(long_range_type="none")
    return model, _save_checkpoint(
        model, "UnifiedEquivariantMLIP", unified_config(long_range_type="none"), tmp_path / "u.pt"
    )


def test_calculator_and_ase_preserve_partial_pbc(unified_checkpoint):
    from gmd_sgt.inference import MLIPCalculator

    model, path = unified_checkpoint
    calc = MLIPCalculator.from_checkpoint(path)
    positions = _positions(4, seed=7).astype(np.float32)
    species = np.array([1, 6, 8, 1])
    slab = [True, True, False]

    ref_e, ref_f = evaluate(
        model.eval(), torch.tensor(species), torch.tensor(positions), torch.zeros(4, dtype=torch.long),
        cell=torch.tensor(TRICLINIC, dtype=torch.float32), pbc=torch.tensor(slab),
    )
    res = calc.compute(positions, species, cell=TRICLINIC, pbc=slab)
    assert res["energy"] == pytest.approx(float(ref_e), rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(res["forces"], ref_f.numpy(), atol=1e-5)
    full = calc.compute(positions, species, cell=TRICLINIC, pbc=True)
    assert full["energy"] != pytest.approx(res["energy"], rel=1e-6, abs=1e-6)

    atoms = Atoms(numbers=species, positions=positions, cell=TRICLINIC, pbc=slab)
    atoms.calc = calc.get_ase_calculator()
    assert atoms.get_potential_energy() == pytest.approx(res["energy"], rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(atoms.get_forces(), res["forces"], atol=1e-6)


def test_ase_stress_matches_numerical_stress(unified_checkpoint):
    from gmd_sgt.inference import MLIPCalculator

    _, path = unified_checkpoint
    calc = MLIPCalculator.from_checkpoint(path)
    atoms = Atoms(
        numbers=[1, 6, 8, 1], positions=_positions(4, seed=7), cell=TRICLINIC, pbc=True
    )
    atoms.calc = calc.get_ase_calculator()
    stress = atoms.get_stress()
    numerical = atoms.calc.calculate_numerical_stress(atoms, d=1e-3)
    assert np.abs(stress).max() > 1e-4
    np.testing.assert_allclose(stress, numerical, atol=2e-4, rtol=2e-2)


def test_online_predictor_accepts_scalar_and_per_axis_pbc(unified_checkpoint):
    from gmd_sgt.api import OnlinePredictor, StructureInput

    _, path = unified_checkpoint
    predictor = OnlinePredictor.from_checkpoint(path)
    positions = _positions(4, seed=7).astype(np.float32)
    base = {"positions": positions, "species": [1, 6, 8, 1], "cell": TRICLINIC}
    calc = predictor.calculator

    slab = predictor.predict({**base, "pbc": [True, True, False]})
    expected = calc.compute(positions, np.array([1, 6, 8, 1]), cell=TRICLINIC, pbc=[True, True, False])
    assert slab.energy == pytest.approx(expected["energy"])

    scalar = predictor.predict({**base, "pbc": True})
    dataclass_input = predictor.predict(
        StructureInput(positions=positions, species=np.array([1, 6, 8, 1]), cell=TRICLINIC, pbc=True)
    )
    full = calc.compute(positions, np.array([1, 6, 8, 1]), cell=TRICLINIC, pbc=True)
    assert scalar.energy == pytest.approx(full["energy"])
    assert dataclass_input.energy == pytest.approx(full["energy"])

    with pytest.raises(ValueError, match="pbc"):
        predictor.predict({"positions": positions, "species": [1, 6, 8, 1], "pbc": True})
