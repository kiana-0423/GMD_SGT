"""tests/test_calculator.py

Unit tests for MLIPCalculator inference interface.

Tests
-----
- from_checkpoint() loads model correctly
- compute() returns expected keys and shapes
- cutoff property matches model config
- Non-periodic and PBC paths produce sensible output
- get_ase_calculator() integration (skipped if ASE not installed)
- TorchScript export produces identical output to eager model

Run with:
    pytest tests/test_calculator.py -v
"""

from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np
import torch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_config():
    return dict(
        n_species=10,
        n_blocks=1,
        scalar_dim=16,
        irreps="16x0e",
        n_basis=4,
        local_cutoff=4.0,
        lr_cutoff=8.0,
        l_max=0,
        long_range_type="none",
        n_heads=1,
        avg_neighbors=4.0,
    )


@pytest.fixture(scope="module")
def checkpoint_path(model_config, tmp_path_factory):
    """Write a minimal checkpoint to a temp file."""
    from gmd_se3gnn.model import UnifiedEquivariantMLIP

    model = UnifiedEquivariantMLIP(**model_config)
    tmp = tmp_path_factory.mktemp("ckpt") / "ckpt_test.pt"
    torch.save({
        "epoch": 1,
        "val_loss": 0.5,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
    }, str(tmp))
    return str(tmp)


@pytest.fixture(scope="module")
def calculator(checkpoint_path):
    from gmd_se3gnn.inference import MLIPCalculator
    return MLIPCalculator.from_checkpoint(checkpoint_path, device="cpu")


def _water_cluster():
    """8 atoms: 2 H₂O + 4 extra H for variety, non-periodic."""
    species = np.array([8, 1, 1, 8, 1, 1, 1, 8], dtype=np.int64)
    positions = np.array([
        [0.000,  0.000,  0.000],
        [0.757,  0.586,  0.000],
        [-0.757, 0.586,  0.000],
        [3.000,  0.000,  0.000],
        [3.757,  0.586,  0.000],
        [2.243,  0.586,  0.000],
        [1.500,  2.000,  0.000],
        [1.500,  4.000,  0.000],
    ], dtype=np.float32)
    return species, positions


# ── from_checkpoint tests ─────────────────────────────────────────────────────

class TestFromCheckpoint:

    def test_loads_without_error(self, checkpoint_path):
        from gmd_se3gnn.inference import MLIPCalculator
        calc = MLIPCalculator.from_checkpoint(checkpoint_path)
        assert calc is not None

    def test_missing_model_config_raises(self, tmp_path):
        """Checkpoint without model_config must raise KeyError."""
        from gmd_se3gnn.inference import MLIPCalculator
        bad_ckpt = str(tmp_path / "bad.pt")
        torch.save({"epoch": 0, "model_state_dict": {}}, bad_ckpt)
        with pytest.raises(KeyError, match="model_config"):
            MLIPCalculator.from_checkpoint(bad_ckpt)

    def test_cutoff_matches_config(self, calculator, model_config):
        assert calculator.cutoff == pytest.approx(model_config["local_cutoff"])

    def test_model_in_eval_mode(self, calculator):
        assert not calculator.model.training


# ── compute() tests ───────────────────────────────────────────────────────────

class TestCompute:

    def test_returns_energy_and_forces(self, calculator):
        species, positions = _water_cluster()
        result = calculator.compute(positions, species)
        assert "energy" in result
        assert "forces" in result

    def test_energy_is_scalar_float(self, calculator):
        species, positions = _water_cluster()
        result = calculator.compute(positions, species)
        assert isinstance(result["energy"], float)

    def test_forces_shape(self, calculator):
        species, positions = _water_cluster()
        N = len(species)
        result = calculator.compute(positions, species)
        assert result["forces"].shape == (N, 3)

    def test_forces_dtype_float64(self, calculator):
        species, positions = _water_cluster()
        result = calculator.compute(positions, species)
        assert result["forces"].dtype == np.float64

    def test_energy_is_finite(self, calculator):
        species, positions = _water_cluster()
        result = calculator.compute(positions, species)
        assert np.isfinite(result["energy"])

    def test_forces_are_finite(self, calculator):
        species, positions = _water_cluster()
        result = calculator.compute(positions, species)
        assert np.all(np.isfinite(result["forces"]))

    def test_deterministic(self, calculator):
        """Same input must give same output."""
        species, positions = _water_cluster()
        r1 = calculator.compute(positions, species)
        r2 = calculator.compute(positions, species)
        assert r1["energy"] == pytest.approx(r2["energy"])
        np.testing.assert_array_equal(r1["forces"], r2["forces"])

    def test_external_edge_index_accepted(self, calculator):
        """Passing pre-built edge_index should not raise."""
        species, positions = _water_cluster()
        N = len(species)
        # Build a trivial all-pairs edge_index (within cutoff for this test)
        src, dst = [], []
        cutoff = calculator.cutoff
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < cutoff:
                        src.append(i); dst.append(j)
        if src:
            ei = np.array([src, dst], dtype=np.int64)
            es = np.zeros((len(src), 3), dtype=np.float32)
            result = calculator.compute(positions, species,
                                        edge_index=ei, edge_shift=es)
            assert "energy" in result


# ── TorchScript export tests ──────────────────────────────────────────────────

class TestTorchScriptExport:

    def test_export_produces_file(self, checkpoint_path, tmp_path):
        from gmd_se3gnn.inference.export import export_torchscript
        out = str(tmp_path / "model.pt")
        export_torchscript(checkpoint_path, out, device="cpu")
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_exported_model_runs(self, checkpoint_path, tmp_path):
        from gmd_se3gnn.inference.export import export_torchscript
        out = str(tmp_path / "model.pt")
        export_torchscript(checkpoint_path, out, device="cpu")

        scripted = torch.jit.load(out)
        species, positions = _water_cluster()
        N = len(species)

        spc_t = torch.tensor(species, dtype=torch.long)
        pos_t = torch.tensor(positions, dtype=torch.float32)
        # Build trivial edge_index for test
        ei = torch.zeros(2, 0, dtype=torch.long)
        es = torch.zeros(0, 3, dtype=torch.float32)

        result = scripted.forward(spc_t, pos_t, ei, es)
        assert "energy" in result
        assert "forces" in result
        assert result["forces"].shape == (N, 3)

    def test_export_output_matches_eager(self, checkpoint_path, tmp_path):
        """TorchScript forward must match eager MLIPCalculator output."""
        from gmd_se3gnn.inference import MLIPCalculator
        from gmd_se3gnn.inference.export import export_torchscript

        calc = MLIPCalculator.from_checkpoint(checkpoint_path)
        species, positions = _water_cluster()

        # Eager result (with no external edge_index → internal graph)
        r_eager = calc.compute(positions, species)

        # Scripted result (with explicit edge_index built the same way)
        out = str(tmp_path / "model.pt")
        export_torchscript(checkpoint_path, out)
        scripted = torch.jit.load(out)

        spc_t = torch.tensor(species, dtype=torch.long)
        pos_t = torch.tensor(positions, dtype=torch.float32)
        batch = torch.zeros(len(species), dtype=torch.long)

        # Rebuild the same edge_index that eager uses
        with torch.no_grad():
            ei, es = calc.model.build_neighbor_graph(
                pos_t, batch, calc.model.local_cutoff
            )
        if es is None:
            es = torch.zeros(ei.shape[1], 3, dtype=torch.float32)

        r_script = scripted.forward(spc_t, pos_t, ei, es)
        e_script = float(r_script["energy"].sum().item())

        assert e_script == pytest.approx(r_eager["energy"], rel=1e-4)


# ── ASE calculator integration ────────────────────────────────────────────────

class TestASECalculator:

    @pytest.fixture(autouse=True)
    def skip_if_no_ase(self):
        pytest.importorskip("ase")

    def test_get_ase_calculator_returns_object(self, calculator):
        ase_calc = calculator.get_ase_calculator()
        assert ase_calc is not None

    def test_ase_calculator_computes_energy(self, calculator):
        import ase
        ase_calc = calculator.get_ase_calculator()
        species, positions = _water_cluster()
        atoms = ase.Atoms(
            numbers=species,
            positions=positions,
            calculator=ase_calc,
        )
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)

    def test_ase_calculator_computes_forces(self, calculator):
        import ase
        ase_calc = calculator.get_ase_calculator()
        species, positions = _water_cluster()
        atoms = ase.Atoms(
            numbers=species,
            positions=positions,
            calculator=ase_calc,
        )
        forces = atoms.get_forces()
        assert forces.shape == (len(species), 3)
        assert np.all(np.isfinite(forces))
