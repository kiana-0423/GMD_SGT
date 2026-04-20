"""Energy + force (+ stress) loss functions."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class EnergyForceLoss(nn.Module):
    """
    Combined energy + force (+ optional stress) loss.

    L = w_E * L_energy + w_F * L_force + w_S * L_stress

    Energy loss uses per-atom normalisation to remove system-size dependence.
    Force loss uses RMSE by default (more sensitive to large errors).

    Parameters
    ----------
    w_energy    : weight for energy term
    w_force     : weight for force term
    w_stress    : weight for stress term (set 0 to disable)
    energy_loss : 'mae' | 'mse' | 'huber'
    force_loss  : 'rmse' | 'mae' | 'huber'
    huber_delta : delta for Huber loss
    """

    def __init__(
        self,
        w_energy: float = 1.0,
        w_force: float = 1.0,
        w_stress: float = 0.01,
        energy_loss: str = "mae",
        force_loss: str = "rmse",
        huber_delta: float = 0.01,
    ):
        super().__init__()
        self.w_energy = w_energy
        self.w_force = w_force
        self.w_stress = w_stress
        self.energy_loss = energy_loss
        self.force_loss = force_loss
        self.huber_delta = huber_delta

    @staticmethod
    def _mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs().mean()

    @staticmethod
    def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target) ** 2).mean()

    @staticmethod
    def _rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target) ** 2).mean().sqrt()

    def _huber(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.huber_loss(pred, target, delta=self.huber_delta)

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        n_atoms_per_graph: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        pred               : model output dict (energy, forces, stress)
        target             : ground-truth dict (energy, forces, stress)
        n_atoms_per_graph  : [n_graphs]  number of atoms per structure

        Returns
        -------
        total_loss : scalar tensor
        loss_dict  : {'energy': ..., 'force': ..., 'stress': ...}
        """
        loss_dict: Dict[str, torch.Tensor] = {}
        total = torch.zeros(1, device=pred["energy"].device)

        # Energy loss — per-atom normalised
        e_pred = pred["energy"] / n_atoms_per_graph.float()
        e_ref = target["energy"] / n_atoms_per_graph.float()
        _efn = {"mae": self._mae, "mse": self._mse, "huber": self._huber}
        e_loss = _efn[self.energy_loss](e_pred, e_ref)
        loss_dict["energy"] = e_loss
        total = total + self.w_energy * e_loss

        # Force loss
        if "forces" in pred and "forces" in target:
            _ffn = {"rmse": self._rmse, "mae": self._mae, "huber": self._huber}
            f_loss = _ffn[self.force_loss](pred["forces"], target["forces"])
            loss_dict["force"] = f_loss
            total = total + self.w_force * f_loss

        # Stress loss
        if (
            self.w_stress > 0
            and "stress" in pred
            and pred["stress"] is not None
            and "stress" in target
        ):
            s_loss = self._mae(pred["stress"], target["stress"])
            loss_dict["stress"] = s_loss
            total = total + self.w_stress * s_loss

        return total.squeeze(), loss_dict
