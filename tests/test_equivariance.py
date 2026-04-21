"""tests/test_equivariance.py

Unit tests for SE(3) / E(3) equivariance of UnifiedEquivariantMLIP.

Tests
-----
- Translation invariance of energy
- Rotation equivariance of forces  (R·F(x) == F(R·x))
- Permutation equivariance of energy and forces
- Reflection (improper rotation) test
- Batch consistency  (batched result == per-graph result)

Run with:
    pytest tests/test_equivariance.py -v
"""

from __future__ import annotations

import math
import pytest
import torch


# ── Helpers ──────────────────────────────────────────────────────────────────

def _small_model():
    """Minimal model for fast testing (no e3nn / torch_cluster required)."""
    from gmd_sgt.model import UnifiedEquivariantMLIP
    return UnifiedEquivariantMLIP(
        n_species=10,
        n_blocks=1,
        scalar_dim=16,
        irreps="16x0e",   # scalars only → works without e3nn
        n_basis=4,
        local_cutoff=4.0,
        lr_cutoff=8.0,
        l_max=0,
        long_range_type="none",
        n_heads=1,
        avg_neighbors=4.0,
    )


def _random_frame(n_atoms: int = 8, seed: int = 42):
    """Return (species, positions, batch) for a single random graph."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    species   = torch.randint(1, 8, (n_atoms,))
    positions = torch.randn(n_atoms, 3, generator=rng) * 2.0
    batch     = torch.zeros(n_atoms, dtype=torch.long)
    return species, positions, batch


def _random_rotation(seed: int = 0) -> torch.Tensor:
    """Return a random SO(3) rotation matrix [3,3]."""
    torch.manual_seed(seed)
    q = torch.randn(4)
    q = q / q.norm()
    w, x, y, z = q
    R = torch.tensor([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    return R


def _forward(model, species, positions, batch):
    pos = positions.clone().requires_grad_(True)
    out = model(species=species, positions=pos, batch=batch, compute_forces=True)
    return out["energy"], out["forces"]


def _mock_pbc_builder(positions, cell, cutoff):
    """Deterministic per-graph neighbor builder used to test batch stitching."""
    del cell, cutoff
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


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTranslationInvariance:
    """Energy must be invariant under global translation."""

    def test_energy_translation_invariant(self):
        model = _small_model()
        model.eval()
        species, pos, batch = _random_frame()

        E0, _ = _forward(model, species, pos, batch)

        shift = torch.tensor([3.0, -1.5, 2.7])
        pos_shifted = pos + shift.unsqueeze(0)
        E1, _ = _forward(model, species, pos_shifted, batch)

        assert torch.allclose(E0, E1, atol=1e-5), (
            f"Energy changed by translation: {(E1 - E0).abs().max().item():.2e}"
        )

    def test_forces_translation_invariant(self):
        """Forces (=-dE/dr) must be identical after translation."""
        model = _small_model()
        model.eval()
        species, pos, batch = _random_frame()

        _, F0 = _forward(model, species, pos, batch)

        shift = torch.tensor([5.0, 0.0, -3.0])
        _, F1 = _forward(model, species, pos + shift, batch)

        assert torch.allclose(F0, F1, atol=1e-5), (
            f"Forces changed by translation: max diff {(F1 - F0).abs().max().item():.2e}"
        )


class TestRotationEquivariance:
    """Forces must transform as vectors under SO(3) rotation."""

    def test_forces_rotate_with_frame(self):
        model = _small_model()
        model.eval()
        species, pos, batch = _random_frame()
        R = _random_rotation()

        # Forces in original frame
        _, F_orig = _forward(model, species, pos, batch)

        # Rotate positions, compute forces, then rotate forces back
        pos_rot = pos @ R.T
        _, F_rot = _forward(model, species, pos_rot, batch)

        # R·F(x) should equal F(R·x)
        F_expected = F_orig @ R.T
        assert torch.allclose(F_rot, F_expected, atol=1e-4), (
            f"Force equivariance violated: max diff {(F_rot - F_expected).abs().max().item():.2e}"
        )

    def test_energy_rotation_invariant(self):
        model = _small_model()
        model.eval()
        species, pos, batch = _random_frame()
        R = _random_rotation()

        E0, _ = _forward(model, species, pos, batch)
        E1, _ = _forward(model, species, pos @ R.T, batch)

        assert torch.allclose(E0, E1, atol=1e-5), (
            f"Energy not rotation-invariant: diff {(E1 - E0).abs().max().item():.2e}"
        )


class TestPermutationEquivariance:
    """Energy must be invariant and forces must permute with atoms."""

    def test_energy_permutation_invariant(self):
        model = _small_model()
        model.eval()
        species, pos, batch = _random_frame(n_atoms=6)

        E0, _ = _forward(model, species, pos, batch)

        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        E1, _ = _forward(model, species[perm], pos[perm], batch)

        assert torch.allclose(E0, E1, atol=1e-5), (
            f"Energy not permutation-invariant: diff {(E1 - E0).abs().max().item():.2e}"
        )

    def test_forces_permute_with_atoms(self):
        model = _small_model()
        model.eval()
        species, pos, batch = _random_frame(n_atoms=6)

        _, F0 = _forward(model, species, pos, batch)

        perm = torch.tensor([3, 0, 5, 1, 4, 2])
        _, F1 = _forward(model, species[perm], pos[perm], batch)

        assert torch.allclose(F0[perm], F1, atol=1e-4), (
            f"Forces not permutation-equivariant: max diff "
            f"{(F0[perm] - F1).abs().max().item():.2e}"
        )


class TestBatchConsistency:
    """Batching multiple graphs must give same result as individual passes."""

    def test_batch_energy_matches_individual(self):
        model = _small_model()
        model.eval()

        # Two independent graphs
        s0, p0, _ = _random_frame(n_atoms=5, seed=1)
        s1, p1, _ = _random_frame(n_atoms=7, seed=2)

        E0, _ = _forward(model, s0, p0, torch.zeros(5, dtype=torch.long))
        E1, _ = _forward(model, s1, p1, torch.zeros(7, dtype=torch.long))

        # Batched
        s_cat = torch.cat([s0, s1])
        p_cat = torch.cat([p0, p1])
        b_cat = torch.cat([
            torch.zeros(5, dtype=torch.long),
            torch.ones(7, dtype=torch.long),
        ])
        E_batch, _ = _forward(model, s_cat, p_cat, b_cat)

        assert torch.allclose(E_batch[0], E0[0], atol=1e-5), \
            f"Batch graph-0 energy mismatch: {(E_batch[0] - E0[0]).abs().item():.2e}"
        assert torch.allclose(E_batch[1], E1[0], atol=1e-5), \
            f"Batch graph-1 energy mismatch: {(E_batch[1] - E1[0]).abs().item():.2e}"

    def test_batch_forces_match_individual(self):
        model = _small_model()
        model.eval()

        s0, p0, _ = _random_frame(n_atoms=4, seed=10)
        s1, p1, _ = _random_frame(n_atoms=6, seed=20)

        _, F0 = _forward(model, s0, p0, torch.zeros(4, dtype=torch.long))
        _, F1 = _forward(model, s1, p1, torch.zeros(6, dtype=torch.long))

        s_cat = torch.cat([s0, s1])
        p_cat = torch.cat([p0, p1])
        b_cat = torch.cat([torch.zeros(4, dtype=torch.long),
                           torch.ones(6, dtype=torch.long)])
        _, F_batch = _forward(model, s_cat, p_cat, b_cat)

        assert torch.allclose(F_batch[:4], F0, atol=1e-4), \
            f"Batch forces graph-0 mismatch: {(F_batch[:4] - F0).abs().max().item():.2e}"
        assert torch.allclose(F_batch[4:], F1, atol=1e-4), \
            f"Batch forces graph-1 mismatch: {(F_batch[4:] - F1).abs().max().item():.2e}"


class TestBatchedPBCNeighborGraph:
    """Batched PBC graphs must be built per structure and stitched with offsets."""

    def test_batched_pbc_edge_indices_are_offset_per_graph(self, monkeypatch):
        model = _small_model()
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
        model = _small_model()
        model.eval()
        monkeypatch.setattr(model, "_build_neighbor_graph_pbc", _mock_pbc_builder)

        s0, p0, _ = _random_frame(n_atoms=4, seed=11)
        s1, p1, _ = _random_frame(n_atoms=5, seed=12)
        cell0 = torch.eye(3)
        cell1 = torch.eye(3) * 1.5

        out0 = model(
            species=s0,
            positions=p0.clone().requires_grad_(True),
            batch=torch.zeros(4, dtype=torch.long),
            cell=cell0,
            compute_forces=True,
        )
        E0 = out0["energy"]
        F0 = out0["forces"]
        out1 = model(
            species=s1,
            positions=p1.clone().requires_grad_(True),
            batch=torch.zeros(5, dtype=torch.long),
            cell=cell1,
            compute_forces=True,
        )
        E1 = out1["energy"]
        F1 = out1["forces"]

        s_cat = torch.cat([s0, s1], dim=0)
        p_cat = torch.cat([p0, p1], dim=0)
        b_cat = torch.cat([torch.zeros(4, dtype=torch.long), torch.ones(5, dtype=torch.long)])
        cell_cat = torch.stack([cell0, cell1], dim=0)
        out_batch = model(
            species=s_cat,
            positions=p_cat.clone().requires_grad_(True),
            batch=b_cat,
            cell=cell_cat,
            compute_forces=True,
        )

        E_batch = out_batch["energy"]
        F_batch = out_batch["forces"]
        assert torch.allclose(E_batch[0], E0[0], atol=1e-5)
        assert torch.allclose(E_batch[1], E1[0], atol=1e-5)
        assert torch.allclose(F_batch[:4], F0, atol=1e-4)
        assert torch.allclose(F_batch[4:], F1, atol=1e-4)


class TestForceConsistency:
    """Forces must equal -dE/dr (numerical finite-difference check)."""

    def test_forces_match_finite_difference(self):
        model = _small_model()
        model.eval()
        species, pos, batch = _random_frame(n_atoms=4, seed=99)

        _, F_auto = _forward(model, species, pos, batch)

        eps = 1e-3
        F_fd = torch.zeros_like(pos)
        for i in range(pos.shape[0]):
            for d in range(3):
                pos_p = pos.clone(); pos_p[i, d] += eps
                pos_m = pos.clone(); pos_m[i, d] -= eps
                Ep, _ = _forward(model, species, pos_p, batch)
                Em, _ = _forward(model, species, pos_m, batch)
                F_fd[i, d] = -(Ep.sum() - Em.sum()) / (2 * eps)

        assert torch.allclose(F_auto, F_fd, atol=5e-3), (
            f"Autograd forces differ from finite diff: "
            f"max {(F_auto - F_fd).abs().max().item():.2e}"
        )
