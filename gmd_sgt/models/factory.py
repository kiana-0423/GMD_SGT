"""Model registry and checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .backbone_allegro_style import AllegroStyleBackbone
from .core import UnifiedEquivariantMLIP
from .gmd_sgt_model import GMDSGTModel

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "UnifiedEquivariantMLIP": UnifiedEquivariantMLIP,
    "AllegroStyleBackbone": AllegroStyleBackbone,
    "GMDSGTModel": GMDSGTModel,
}


def get_model_class(model_type: str | None) -> type[nn.Module]:
    """Resolve a model class name from checkpoint metadata."""
    return MODEL_REGISTRY.get(model_type or "UnifiedEquivariantMLIP", UnifiedEquivariantMLIP)


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
    return checkpoint, model
