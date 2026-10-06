"""Loading checkpoints written before the v0.3 fixes."""

from __future__ import annotations

import pytest
import torch

from gmd_sgt.models import get_model_class, load_model_from_checkpoint
from tests._helpers import make_unified, unified_config


def test_legacy_unified_checkpoint_loads_with_warning(tmp_path):
    config = unified_config(long_range_type="invariant_attention", lr_cutoff=12.0)
    model = make_unified(long_range_type="invariant_attention", lr_cutoff=12.0)
    state = model.state_dict()
    # Older versions registered an unused long-range Bessel basis buffer.
    state["radial_basis_lr.freq"] = torch.arange(1, 5, dtype=torch.float32)
    path = tmp_path / "legacy.pt"
    torch.save({"model_config": config, "model_state_dict": state}, path)  # no model_type/format

    with pytest.warns(UserWarning, match="predates checkpoint format"):
        checkpoint, loaded = load_model_from_checkpoint(path)
    assert type(loaded).__name__ == "UnifiedEquivariantMLIP"
    assert loaded.lr_cutoff == 12.0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_current_checkpoint_loads_without_warning(tmp_path, recwarn):
    model = make_unified()
    path = tmp_path / "current.pt"
    torch.save(
        {
            "model_type": "UnifiedEquivariantMLIP",
            "model_config": unified_config(),
            "model_state_dict": model.state_dict(),
            "format_version": 2,
        },
        path,
    )
    load_model_from_checkpoint(path)
    assert not [w for w in recwarn if "predates" in str(w.message)]


def test_unknown_model_type_is_rejected():
    with pytest.raises(KeyError, match="Unknown model_type"):
        get_model_class("NotAModel")
