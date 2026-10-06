"""Packaging metadata sanity checks (the full wheel install is exercised in CI/manually)."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject():
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def test_build_backend_is_importable(pyproject):
    backend = pyproject["build-system"]["build-backend"]
    module_name, _, attr = backend.partition(":")
    module = importlib.import_module(module_name)
    if attr:
        getattr(module, attr)
    assert hasattr(module, "build_wheel")


def test_console_scripts_live_inside_the_distributed_package(pyproject):
    includes = pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    for name, target in pyproject["project"]["scripts"].items():
        module_name, _, func = target.partition(":")
        top_level = module_name.split(".")[0]
        assert any(
            top_level == pattern.rstrip("*") for pattern in includes
        ), f"{name} -> {module_name} is not shipped in the wheel"
        assert callable(getattr(importlib.import_module(module_name), func))


def test_version_metadata_is_single_sourced(pyproject):
    import gmd_sgt

    assert "version" in pyproject["project"]["dynamic"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "gmd_sgt.__version__"
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    major_minor = ".".join(gmd_sgt.__version__.split(".")[:2])
    assert f"v{major_minor}" in readme.splitlines()[0]


def test_cli_help_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "gmd_sgt.cli", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--config" in proc.stdout
