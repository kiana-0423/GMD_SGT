"""Model registry and checkpoint loading helpers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .backbone_allegro_style import AllegroStyleBackbone
from .core import UnifiedEquivariantMLIP
from .gmd_sgt_model import GMDSGTModel

# Bumped when model semantics change in a way that alters predictions of
# previously trained weights. Version 2: smooth cutoff on every message,
# envelope-weighted attention, scalar-only feature initialisation, l_max
# consistent harmonics and corrected l=2 basis normalisation.
CHECKPOINT_FORMAT_VERSION = 2

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "UnifiedEquivariantMLIP": UnifiedEquivariantMLIP,
    "AllegroStyleBackbone": AllegroStyleBackbone,
    "GMDSGTModel": GMDSGTModel,
}


def get_model_class(model_type: str | None) -> type[nn.Module]:
    """Resolve a model class name from checkpoint metadata."""
    name = model_type or "UnifiedEquivariantMLIP"
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_type {name!r}; expected one of {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def instantiate_model(model_type: str | None, model_config: dict[str, Any]) -> nn.Module:
    """Instantiate a registered model from config."""
    model_cls = get_model_class(model_type)
    return model_cls(**model_config)


def load_model_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], nn.Module]:
    """Load checkpoint metadata and reconstruct the corresponding model."""
    checkpoint = torch.load(str(path), map_location=map_location, weights_only=False)
    if "model_config" not in checkpoint:
        raise KeyError(
            f"Checkpoint {path!r} is missing 'model_config'. "
            "Re-train with the current Trainer to produce a valid checkpoint."
        )

    model = instantiate_model(
        checkpoint.get("model_type", "UnifiedEquivariantMLIP"),
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if int(checkpoint.get("format_version", 1)) < CHECKPOINT_FORMAT_VERSION:
        warnings.warn(
            f"Checkpoint {path!r} predates checkpoint format {CHECKPOINT_FORMAT_VERSION}. "
            "Its weights load, but cutoff smoothing, attention normalisation and feature "
            "initialisation have since been corrected, so predictions differ from the "
            "original run; retrain or fine-tune before production use.",
            UserWarning,
            stacklevel=2,
        )
    return checkpoint, model
