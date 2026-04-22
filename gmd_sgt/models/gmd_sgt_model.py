"""Hybrid staged residual-learning MLIP model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .backbone_allegro_style import AllegroStyleBackbone
from .geometry import scatter_sum
from .gnn_correction import GNNCorrection
from .transformer_correction import TransformerCorrection


class GMDSGTModel(nn.Module):
    """Backbone + residual correction branches with conservative force output."""

    def __init__(
        self,
        backbone_config: Optional[Dict[str, object]] = None,
        use_gnn: bool = True,
        gnn_hidden_channels: Optional[int] = None,
        gnn_layers: int = 2,
        use_transformer: bool = False,
        transformer_hidden_channels: Optional[int] = None,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.0,
        lambda_gnn: float = 1.0,
        lambda_attn: float = 1.0,
    ):
        super().__init__()
        backbone_config = dict(backbone_config or {})
        self.backbone = AllegroStyleBackbone(**backbone_config)
        self.local_cutoff = self.backbone.local_cutoff
        self.lr_cutoff = self.backbone.lr_cutoff
        self.lambda_gnn = float(lambda_gnn)
        self.lambda_attn = float(lambda_attn)
        self.use_gnn = use_gnn
        self.use_transformer = use_transformer

        node_dim = self.backbone.hidden_channels
        edge_dim = self.backbone.n_basis + 1
        n_species = self.backbone.n_species

        self.gnn_correction = (
            GNNCorrection(
                n_species=n_species,
                input_channels=node_dim,
                hidden_channels=gnn_hidden_channels or node_dim,
                edge_dim=edge_dim,
                num_layers=gnn_layers,
            )
            if use_gnn
            else None
        )
        self.transformer_correction = (
            TransformerCorrection(
                n_species=n_species,
                input_channels=node_dim,
                hidden_channels=transformer_hidden_channels or node_dim,
                edge_dim=edge_dim,
                num_layers=transformer_layers,
                n_heads=transformer_heads,
                dropout=transformer_dropout,
            )
            if use_transformer
            else None
        )

    @classmethod
    def from_backbone_checkpoint(
        cls,
        checkpoint_path: str | Path,
        **kwargs,
    ) -> "GMDSGTModel":
        """Instantiate a residual model using a Stage 1 backbone checkpoint."""
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        backbone_config = checkpoint["model_config"]
        model = cls(backbone_config=backbone_config, **kwargs)
        model.load_backbone_checkpoint(str(checkpoint_path))
        return model

    def load_backbone_checkpoint(self, checkpoint_path: str | Path, strict: bool = True) -> None:
        """Load Stage 1 backbone weights into the embedded backbone module."""
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        model_type = checkpoint.get("model_type", "AllegroStyleBackbone")
        if model_type != "AllegroStyleBackbone":
            raise ValueError(
                f"Expected AllegroStyleBackbone checkpoint, received {model_type}"
            )
        self.backbone.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def semi_freeze_backbone(self) -> None:
        """Freeze backbone except the energy readout head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.readout.parameters():
            param.requires_grad = True

    def forward(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_shift: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        neighbor_list: Optional[dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]] = None,
        compute_forces: bool = True,
        compute_stress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del compute_stress

        if compute_forces and not positions.requires_grad:
            positions = positions.requires_grad_(True)

        backbone_out = self.backbone(
            species=species,
            positions=positions,
            batch=batch,
            edge_index=edge_index,
            edge_shift=edge_shift,
            cell=cell,
            neighbor_list=neighbor_list,
            compute_forces=False,
            compute_stress=False,
        )

        atomic_backbone = backbone_out["atomic_energies"]
        edge_features = torch.cat(
            [
                backbone_out["edge_rbf"],
                backbone_out["distances"].unsqueeze(-1),
            ],
            dim=-1,
        )
        coordination = backbone_out["coordination"]
        edge_index = backbone_out["edge_index"]
        n_graphs = int(batch.max().item()) + 1

        delta_atomic_gnn = torch.zeros_like(atomic_backbone)
        delta_atomic_attn = torch.zeros_like(atomic_backbone)

        if self.gnn_correction is not None:
            delta_atomic_gnn = self.gnn_correction(
                node_features=backbone_out["node_features"],
                species=species,
                edge_index=edge_index,
                edge_features=edge_features,
                coordination=coordination,
            )

        if self.transformer_correction is not None:
            delta_atomic_attn = self.transformer_correction(
                node_features=backbone_out["node_features"],
                species=species,
                edge_index=edge_index,
                edge_features=edge_features,
                coordination=coordination,
            )

        atomic_total = (
            atomic_backbone
            + self.lambda_gnn * delta_atomic_gnn
            + self.lambda_attn * delta_atomic_attn
        )
        energy_total = scatter_sum(atomic_total.unsqueeze(-1), batch, n_graphs).squeeze(-1)
        energy_backbone = scatter_sum(
            atomic_backbone.unsqueeze(-1),
            batch,
            n_graphs,
        ).squeeze(-1)
        delta_energy_gnn = scatter_sum(
            delta_atomic_gnn.unsqueeze(-1),
            batch,
            n_graphs,
        ).squeeze(-1)
        delta_energy_attn = scatter_sum(
            delta_atomic_attn.unsqueeze(-1),
            batch,
            n_graphs,
        ).squeeze(-1)

        output: Dict[str, torch.Tensor] = {
            "energy": energy_total,
            "energy_backbone": energy_backbone,
            "delta_energy_gnn": delta_energy_gnn,
            "delta_energy_attn": delta_energy_attn,
            "atomic_energies": atomic_total,
            "atomic_backbone": atomic_backbone,
            "delta_atomic_gnn": delta_atomic_gnn,
            "delta_atomic_attn": delta_atomic_attn,
        }

        if compute_forces:
            grad = torch.autograd.grad(
                outputs=[energy_total.sum()],
                inputs=[positions],
                create_graph=self.training,
                retain_graph=False,
                allow_unused=True,
            )[0]
            output["forces"] = -grad if grad is not None else torch.zeros_like(positions)

        return output
