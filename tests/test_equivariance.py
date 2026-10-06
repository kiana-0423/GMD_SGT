"""tests/test_equivariance.py

E(3) symmetry tests for every model family.

All models use non-zero readouts and every test first checks that the forces
are clearly non-zero, so the symmetry checks cannot pass merely because the
energy is constant or the forces vanish.

Tests
-----
- Translation invariance of energy and forces
- Rotation: energy invariant, forces co-rotate (R·F(x) == F(R·x))
- Reflection / improper rotation: same, with det(R) = -1
- Permutation equivariance of energy and forces
- Batch consistency  (batched result == per-graph result)
- Forces match finite differences of the energy
"""

from __future__ import annotations

import pytest
import torch

from tests._helpers import (
    all_model_factories,
    evaluate,
    make_unified,
    random_cluster,
    random_orthogonal,
)

MODELS = all_model_factories()
MODEL_IDS = [name for name, _ in MODELS]


@pytest.fixture(params=MODELS, ids=MODEL_IDS)
def model(request):
    _, factory = request.param
    return factory().double().eval()


def _assert_nontrivial(forces: torch.Tensor) -> None:
    assert forces.abs().max() > 1e-4, "forces vanish; the symmetry test would be vacuous"


def _frame(n_atoms: int = 7, seed: int = 3):
    return random_cluster(n_atoms=n_atoms, seed=seed)


class TestTranslationInvariance:
    def test_energy_and_forces_translation_invariant(self, model):
        species, pos, batch = _frame()
        e0, f0 = evaluate(model, species, pos, batch)
        _assert_nontrivial(f0)

        shift = torch.tensor([3.0, -1.5, 2.7], dtype=pos.dtype)
        e1, f1 = evaluate(model, species, pos + shift, batch)

        torch.testing.assert_close(e1, e0, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(f1, f0, atol=1e-9, rtol=1e-7)


class TestRotationEquivariance:
    @pytest.mark.parametrize("proper", [True, False], ids=["rotation", "reflection"])
    def test_energy_invariant_forces_equivariant(self, model, proper):
        species, pos, batch = _frame()
        rot = random_orthogonal(seed=11, proper=proper)
        assert torch.det(rot).item() == pytest.approx(1.0 if proper else -1.0)

        e0, f0 = evaluate(model, species, pos, batch)
        _assert_nontrivial(f0)
        e1, f1 = evaluate(model, species, pos @ rot.T, batch)

        torch.testing.assert_close(e1, e0, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(f1, f0 @ rot.T, atol=1e-9, rtol=1e-7)


class TestPermutationEquivariance:
    def test_energy_invariant_forces_permute(self, model):
        species, pos, batch = _frame(n_atoms=6)
        perm = torch.tensor([3, 0, 5, 1, 4, 2])

        e0, f0 = evaluate(model, species, pos, batch)
        _assert_nontrivial(f0)
        e1, f1 = evaluate(model, species[perm], pos[perm], batch)

        torch.testing.assert_close(e1, e0, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(f1, f0[perm], atol=1e-9, rtol=1e-7)


class TestBatchConsistency:
    """Batching multiple graphs must give same result as individual passes."""

    def test_batch_matches_individual(self, model):
        s0, p0, b0 = _frame(n_atoms=5, seed=1)
        s1, p1, b1 = _frame(n_atoms=7, seed=2)
        e0, f0 = evaluate(model, s0, p0, b0)
        e1, f1 = evaluate(model, s1, p1, b1)
        _assert_nontrivial(f0)

        e_batch, f_batch = evaluate(
            model,
            torch.cat([s0, s1]),
            torch.cat([p0, p1]),
            torch.cat([b0, b1 + 1]),
        )
        torch.testing.assert_close(e_batch, torch.cat([e0, e1]), atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(f_batch, torch.cat([f0, f1]), atol=1e-9, rtol=1e-7)


class TestForceConsistency:
    """Forces must equal -dE/dr (central finite differences in float64)."""

    def test_forces_match_finite_difference(self, model):
        species, pos, batch = _frame(n_atoms=4, seed=99)
        _, f_auto = evaluate(model, species, pos, batch)
        _assert_nontrivial(f_auto)

        eps = 1e-5
        f_fd = torch.zeros_like(pos)
        for i in range(pos.shape[0]):
            for d in range(3):
                pos_p = pos.clone()
                pos_p[i, d] += eps
                pos_m = pos.clone()
                pos_m[i, d] -= eps
                e_p, _ = evaluate(model, species, pos_p, batch)
                e_m, _ = evaluate(model, species, pos_m, batch)
                f_fd[i, d] = -(e_p.sum() - e_m.sum()) / (2 * eps)

        torch.testing.assert_close(f_auto, f_fd, atol=1e-6, rtol=1e-5)


def _mock_pbc_builder(positions, cell, cutoff, pbc=None):
    """Deterministic per-graph neighbor builder used to test batch stitching."""
    del cell, cutoff, pbc
    n_atoms = positions.shape[0]
    if n_atoms <= 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=positions.device)
        edge_shift = torch.zeros((0, 3), dtype=positions.dtype, device=positions.device)
        return edge_index, edge_shift

    src = torch.arange(n_atoms - 1, device=positions.device, dtype=torch.long)
    dst = src + 1
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    edge_shift = torch.zeros(edge_index.shape[1], 3, dtype=positions.dtype, device=positions.device)
    return edge_index, edge_shift


class TestBatchedPBCNeighborGraph:
    """Batched PBC graphs must be built per structure and stitched with offsets.

    These tests isolate the stitching logic with a mock builder; real ASE
    neighbor lists are tested in tests/test_pbc.py.
    """

    def test_batched_pbc_edge_indices_are_offset_per_graph(self, monkeypatch):
        model = make_unified()
        monkeypatch.setattr(model, "_build_neighbor_graph_pbc", _mock_pbc_builder)

        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        cell = torch.stack([torch.eye(3), torch.eye(3) * 2.0], dim=0)

        edge_index, edge_shift = model.build_neighbor_graph(
            positions=positions,
            batch=batch,
            cutoff=model.local_cutoff,
            cell=cell,
        )

        expected_edge_index = torch.tensor(
            [[0, 1, 1, 2, 3, 4], [1, 2, 0, 1, 4, 3]],
            dtype=torch.long,
        )
        assert torch.equal(edge_index.cpu(), expected_edge_index)
        assert edge_shift.shape == (6, 3)
        assert torch.allclose(edge_shift, torch.zeros_like(edge_shift))

    def test_batched_pbc_forward_matches_individual_forward(self, monkeypatch):
        model = make_unified().double().eval()
        monkeypatch.setattr(model, "_build_neighbor_graph_pbc", _mock_pbc_builder)

        s0, p0, b0 = random_cluster(n_atoms=4, seed=11)
        s1, p1, b1 = random_cluster(n_atoms=5, seed=12)
        cell0 = torch.eye(3, dtype=torch.float64) * 4.0
        cell1 = torch.eye(3, dtype=torch.float64) * 4.5

        e0, f0 = evaluate(model, s0, p0, b0, cell=cell0)
        e1, f1 = evaluate(model, s1, p1, b1, cell=cell1)
        _assert_nontrivial(f0)
        e_batch, f_batch = evaluate(
            model,
            torch.cat([s0, s1]),
            torch.cat([p0, p1]),
            torch.cat([b0, b1 + 1]),
            cell=torch.stack([cell0, cell1]),
        )

        torch.testing.assert_close(e_batch, torch.cat([e0, e1]))
        torch.testing.assert_close(f_batch, torch.cat([f0, f1]))
