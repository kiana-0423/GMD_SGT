"""Behaviour without optional dependencies (e3nn).

The tests marked ``no_e3nn_env`` run when e3nn is genuinely missing. The
subprocess test hides e3nn from a fresh interpreter, so the dependency-minimal
path is exercised in every environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from gmd_sgt.models import AllegroStyleBackbone, UnifiedEquivariantMLIP
from gmd_sgt.models.dependencies import E3NN_AVAILABLE
from tests._helpers import evaluate, make_unified

REPO_ROOT = Path(__file__).resolve().parents[1]
no_e3nn_env = pytest.mark.skipif(E3NN_AVAILABLE, reason="e3nn is installed")


@no_e3nn_env
def test_default_unified_model_requires_e3nn():
    with pytest.raises(ImportError, match="e3nn"):
        UnifiedEquivariantMLIP()


@no_e3nn_env
def test_backbone_l_max_above_two_requires_e3nn():
    with pytest.raises(ImportError, match="e3nn"):
        AllegroStyleBackbone(l_max=3)


@no_e3nn_env
def test_scalar_fallback_rejects_mismatched_scalar_dim():
    with pytest.raises(ValueError, match="scalar_dim"):
        UnifiedEquivariantMLIP(scalar_dim=16, irreps="8x0e", n_heads=1)


@no_e3nn_env
def test_scalar_fallback_energy_depends_on_geometry():
    model = make_unified().double().eval()
    species = torch.tensor([1, 8, 1])
    batch = torch.zeros(3, dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    e1, f1 = evaluate(model, species, pos, batch)
    e2, _ = evaluate(model, species, pos * 1.3, batch)
    assert not torch.allclose(e1, e2)
    assert f1.abs().max() > 1e-4


_NO_E3NN_SCRIPT = textwrap.dedent(
    """
    import importlib.abc, sys

    class _BlockE3nn(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "e3nn" or name.startswith("e3nn."):
                raise ImportError("e3nn hidden for this test")
            return None

    sys.meta_path.insert(0, _BlockE3nn())

    import torch
    from gmd_sgt.models import AllegroStyleBackbone, UnifiedEquivariantMLIP
    from gmd_sgt.models.dependencies import E3NN_AVAILABLE

    assert not E3NN_AVAILABLE
    try:
        UnifiedEquivariantMLIP()
    except ImportError as exc:
        assert "e3nn" in str(exc)
    else:
        raise SystemExit("default (non-scalar) model was built without e3nn")

    torch.manual_seed(0)
    model = UnifiedEquivariantMLIP(n_species=10, n_blocks=2, scalar_dim=8, irreps="8x0e",
                                   n_basis=4, local_cutoff=3.0, long_range_type="none",
                                   n_heads=2).double().eval()
    with torch.no_grad():
        model.energy_head[-1].weight.normal_()
    species = torch.tensor([1, 8, 1])
    batch = torch.zeros(3, dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    out1 = model(species, pos.clone(), batch)
    out2 = model(species, 1.3 * pos, batch)
    assert not torch.allclose(out1["energy"], out2["energy"]), "energy ignores geometry"
    assert out1["forces"].abs().max() > 1e-4, "forces vanish"

    backbone = AllegroStyleBackbone(n_species=10, hidden_channels=8, num_layers=1, l_max=2)
    assert backbone(species, pos.float(), batch)["forces"].shape == (3, 3)
    try:
        AllegroStyleBackbone(l_max=3)
    except ImportError:
        pass
    else:
        raise SystemExit("l_max=3 backbone was built without e3nn")
    print("NO_E3NN_OK")
    """
)


def test_dependency_minimal_behaviour_in_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _NO_E3NN_SCRIPT],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NO_E3NN_OK" in proc.stdout


_PORTABILITY_SCRIPT = textwrap.dedent(
    """
    import importlib.abc, sys

    class _BlockE3nn(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "e3nn" or name.startswith("e3nn."):
                raise ImportError("e3nn hidden for this test")
            return None

    sys.meta_path.insert(0, _BlockE3nn())
    import torch
    from gmd_sgt.models import UnifiedEquivariantMLIP
    from gmd_sgt.models.dependencies import E3NN_AVAILABLE

    assert not E3NN_AVAILABLE
    payload = torch.load(sys.argv[1], weights_only=False)
    model = UnifiedEquivariantMLIP(**payload["config"]).double()
    model.load_state_dict(payload["state_dict"], strict=True)
    worst = 0.0
    for mode in ("eval", "train"):
        model.train(mode == "train")
        out = model(*payload["inputs"])
        ref = payload[mode]
        worst = max(worst, float((out["energy"] - ref["energy"]).abs().max()),
                    float((out["forces"] - ref["forces"]).abs().max()))
    print("MAX_DIFF", worst)
    """
)


@pytest.mark.skipif(not E3NN_AVAILABLE, reason="compares e3nn against the e3nn-free path")
def test_scalar_irreps_checkpoint_is_portable_without_e3nn(tmp_path):
    config = dict(n_species=10, n_blocks=2, scalar_dim=8, irreps="12x0e", n_basis=4,
                  local_cutoff=3.0, l_max=2, long_range_type="equivariant_attention", n_heads=2)
    torch.manual_seed(0)
    model = UnifiedEquivariantMLIP(**config).double()
    with torch.no_grad():
        model.energy_head[-1].weight.normal_()
    species = torch.tensor([1, 6, 8, 1, 6])
    positions = torch.randn(5, 3, dtype=torch.float64) * 1.2
    batch = torch.zeros(5, dtype=torch.long)
    model.train()
    model(species, positions.clone(), batch)  # non-trivial BatchNorm running stats

    payload = {"config": config, "inputs": (species, positions, batch)}
    state = {k: v.clone() for k, v in model.state_dict().items()}
    for mode in ("eval", "train"):
        model.load_state_dict(state)  # identical running stats before each mode
        model.train(mode == "train")
        out = model(species, positions.clone(), batch)
        payload[mode] = {"energy": out["energy"].detach(), "forces": out["forces"].detach()}
        assert out["forces"].abs().max() > 1e-4
    payload["state_dict"] = state
    torch.save(payload, tmp_path / "payload.pt")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _PORTABILITY_SCRIPT, str(tmp_path / "payload.pt")],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    max_diff = float(proc.stdout.split("MAX_DIFF")[1].split()[0])
    assert max_diff < 1e-10, max_diff
