"""Trainer: training loop, logging, checkpointing."""

from __future__ import annotations

import csv
import math
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Type

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from gmd_sgt.models.factory import CHECKPOINT_FORMAT_VERSION

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _loader_has_batches(loader: Optional[DataLoader]) -> bool:
    if loader is None:
        return False
    try:
        return len(loader) > 0
    except TypeError:  # iterable without __len__
        return True


_RESUME_IGNORED_CONFIG_KEYS = frozenset({"atomic_energies"})


def _config_differences(saved, requested, prefix: str = "") -> list:
    """Human-readable differences between two model configs (see from_checkpoint)."""
    if isinstance(saved, dict) and isinstance(requested, dict):
        diffs = []
        for key in sorted(set(saved) | set(requested), key=str):
            if key in _RESUME_IGNORED_CONFIG_KEYS:
                continue
            name = f"{prefix}{key}"
            if key not in saved or key not in requested:
                diffs.append(f"{name}: present in only one config")
            else:
                diffs.extend(_config_differences(saved[key], requested[key], name + "."))
        return diffs
    if saved != requested:
        return [f"{prefix.rstrip('.')}: checkpoint={saved!r}, requested={requested!r}"]
    return []


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before stopping.
        Set to 0 to disable.
    """

    def __init__(self, patience: int = 50):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if self.patience == 0:
            return False
        if val_loss < self.best - 1e-8:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Full training loop with CSV logging, checkpointing, and early stopping.

    Parameters
    ----------
    model:
        The model to train.
    model_config:
        The *dict* of keyword arguments used to construct ``model``.
        Saved into every checkpoint so the model can be reconstructed.
    loss_fn:
        Callable that returns (total_loss, loss_dict).
    train_loader, val_loader:
        DataLoaders for train / validation splits.
    lr:
        Peak learning rate for AdamW.
    lr_min:
        Minimum LR at end of cosine decay.
    weight_decay:
        AdamW weight decay.
    max_grad_norm:
        Gradient clipping norm (0 = disabled).
    n_epochs:
        Total training epochs.
    warmup_steps:
        Linear warmup phase length in optimiser steps.
    device:
        Torch device string (cpu / cuda / mps).
    output_dir:
        Directory for checkpoints and CSV log.
    patience:
        EarlyStopping patience (0 = disabled).

    Validation and checkpoint selection
    -----------------------------------
    When ``val_loader`` yields at least one batch, ``ckpt_best.pt`` and early
    stopping follow the mean validation loss. When it is ``None`` or empty
    (e.g. ``val_fraction=0`` or a tiny dataset), validation is skipped, a
    warning is emitted, and selection follows the mean *training* loss of
    the epoch instead. The metric used is stored in every checkpoint under
    ``selection_metric``.

    Stress is predicted (``compute_stress=True``) exactly when the loss
    function has ``w_stress > 0``.
    """

    CSV_HEADER = [
        "epoch", "train_energy", "train_force", "train_total",
        "val_energy", "val_force", "val_total", "lr", "epoch_time_s",
    ]

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        lr: float = 3e-4,
        lr_min: float = 3e-6,
        weight_decay: float = 1e-5,
        max_grad_norm: float = 1.0,
        n_epochs: int = 500,
        warmup_steps: int = 1000,
        device: str = "cpu",
        output_dir: str = "outputs/run",
        patience: int = 50,
        _start_epoch: int = 0,
        _optim_state: Optional[dict] = None,
        _sched_state: Optional[dict] = None,
        _best_val: float = float("inf"),
        _early_stopping_counter: int = 0,
        _resumed_from_best: bool = False,
    ):
        self.model = model
        self.model_config = model_config
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.compute_stress = float(getattr(loss_fn, "w_stress", 0.0)) > 0.0
        self.has_validation = _loader_has_batches(val_loader)
        self.selection_metric = "val_total" if self.has_validation else "train_total"
        if not self.has_validation:
            warnings.warn(
                "Validation set is empty; checkpoint selection and early stopping "
                "use the training loss instead.",
                UserWarning,
                stacklevel=2,
            )

        # Optimiser
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if _optim_state:
            self.optimizer.load_state_dict(_optim_state)

        # Scheduler: linear warmup → cosine
        total_steps = n_epochs * len(train_loader)
        warmup = LinearLR(self.optimizer, start_factor=1e-4, end_factor=1.0,
                          total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(total_steps - warmup_steps, 1),
                                   eta_min=lr_min)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine],
                                      milestones=[warmup_steps])
        if _sched_state:
            self.scheduler.load_state_dict(_sched_state)

        self.early_stopping = EarlyStopping(patience=patience)
        self.best_val = _best_val
        # Restore early-stopping internal state so counter and best threshold
        # are continuous across resume boundaries.
        self.early_stopping.best = _best_val
        self.early_stopping.counter = _early_stopping_counter
        self.start_epoch = _start_epoch
        # When resuming from the best checkpoint into a new output directory,
        # that checkpoint is the best model until training improves on it.
        self._resumed_from_best = _resumed_from_best

        # CSV log
        self.csv_path = self.output_dir / "training_log.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.CSV_HEADER)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Train for n_epochs epochs starting from start_epoch."""
        if self._resumed_from_best and not (self.output_dir / "ckpt_best.pt").exists():
            self.save_checkpoint(self.start_epoch, self.best_val, tag="best")
        for epoch in range(self.start_epoch + 1, self.n_epochs + 1):
            t0 = time.time()
            tr = self._train_epoch()
            if "total" not in tr:
                raise RuntimeError("Training loader produced no batches")
            va = self._val_epoch() if self.has_validation else {}
            elapsed = time.time() - t0

            self._write_csv(epoch, tr, va, elapsed)
            self._maybe_print(epoch, tr, va, elapsed)

            # Best checkpoint (validation loss, or training loss without validation)
            val_total = va["total"] if self.has_validation else tr["total"]
            if not math.isfinite(val_total):
                raise FloatingPointError(f"Non-finite loss {val_total} at epoch {epoch}")
            # Update early stopping first so every checkpoint written for this
            # epoch stores the state *after* this epoch (resuming from it then
            # continues exactly like the uninterrupted run).
            should_stop = self.early_stopping.step(val_total)
            if val_total < self.best_val:
                self.best_val = val_total
                self.save_checkpoint(epoch, val_total, tag="best")

            # Periodic checkpoint every 50 epochs
            if epoch % 50 == 0:
                self.save_checkpoint(epoch, val_total)

            if should_stop:
                print(f"[EarlyStopping] No improvement for "
                      f"{self.early_stopping.patience} epochs. Stopping at epoch {epoch}.")
                break

        print("Training complete. Best val loss:", round(self.best_val, 6))

    def save_checkpoint(self, epoch: int, val_loss: float, tag: str = "") -> None:
        """Save model + optimizer + scheduler + model_config to disk."""
        fname = f"ckpt_{tag}.pt" if tag else f"ckpt_epoch{epoch:04d}.pt"
        ckpt = {
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val": self.best_val,
            "early_stopping_counter": self.early_stopping.counter,
            "model_type": type(self.model).__name__,
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "selection_metric": self.selection_metric,
            "model_config": self.model_config,          # ← critical for inference reload
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(ckpt, self.output_dir / fname)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_cls: Optional[Type[nn.Module]] = None,
        **trainer_kwargs,
    ) -> "Trainer":
        """Resume training from a saved checkpoint.

        Parameters
        ----------
        checkpoint_path:
            Path to a .pt checkpoint saved by :meth:`save_checkpoint`.
        model_cls:
            The model class to instantiate (e.g. ``UnifiedEquivariantMLIP``).
            Must match the checkpoint's ``model_type`` when that is recorded.
        **trainer_kwargs:
            Arguments forwarded to :class:`Trainer`. Training settings
            (``output_dir``, ``n_epochs``, ``lr``, loaders, ...) may differ
            from the original run.

        Model configuration rules
        -------------------------
        The model is rebuilt from the checkpoint's ``model_config`` and its
        weights, so that configuration is authoritative and is what later
        checkpoints record. A ``model_config`` passed here is only a
        consistency check: it must equal the checkpoint's, except for
        ``atomic_energies`` (at any nesting level), whose values come from the
        checkpoint's ``atomic_energies_ref`` buffer and are kept as saved.
        Any other difference (cutoff, widths, ...) raises ``ValueError``
        before training starts, because it cannot be applied to the trained
        weights.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = ckpt["model_config"]
        saved_type = ckpt.get("model_type")
        if model_cls is None:
            from gmd_sgt.models import get_model_class

            model_cls = get_model_class(saved_type)
        elif saved_type is not None and model_cls.__name__ != saved_type:
            raise ValueError(
                f"Checkpoint {checkpoint_path!r} holds a {saved_type}, "
                f"cannot resume it as {model_cls.__name__}"
            )

        requested = trainer_kwargs.pop("model_config", None)
        if requested is not None:
            differences = _config_differences(model_config, requested)
            if differences:
                raise ValueError(
                    f"Cannot resume {checkpoint_path!r} with a different model configuration "
                    f"({'; '.join(differences)}). The checkpoint's weights define the model; "
                    "start a new run (or fine-tune from a new model) to change it."
                )

        model = model_cls(**model_config)
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(
            model_config=model_config,
            model=model,
            _start_epoch=ckpt["epoch"],
            _optim_state=ckpt.get("optimizer_state_dict"),
            _sched_state=ckpt.get("scheduler_state_dict"),
            _best_val=ckpt.get("best_val", float("inf")),
            _early_stopping_counter=ckpt.get("early_stopping_counter", 0),
            _resumed_from_best=(
                "best_val" in ckpt and ckpt.get("val_loss") == ckpt["best_val"]
            ),
            **trainer_kwargs,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        n_batches = 0

        iterable = self.train_loader
        if _HAS_TQDM:
            iterable = _tqdm(iterable, desc="  train", leave=False)

        for batch in iterable:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            self.optimizer.zero_grad()

            pred = self._predict(batch)
            loss, ld = self.loss_fn(pred, batch, batch.get("n_atoms"))
            loss.backward()

            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()

            for k, v in ld.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            totals["total"] = totals.get("total", 0.0) + float(loss)
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def _val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        n_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            # torch.enable_grad() is required because compute_forces=True uses
            # torch.autograd.grad() internally; @no_grad would suppress the graph.
            with torch.enable_grad():
                pred = self._predict(batch)
                loss, ld = self.loss_fn(pred, batch, batch.get("n_atoms"))
            for k, v in ld.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            totals["total"] = totals.get("total", 0.0) + float(loss)
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def _predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Optional arguments are only passed when used, so custom models with
        # the original (species, positions, batch, cell, compute_forces)
        # signature keep working.
        kwargs = {}
        if "pbc" in batch:
            kwargs["pbc"] = batch["pbc"]
        if self.compute_stress:
            kwargs["compute_stress"] = True
        return self.model(
            species=batch["species"],
            positions=batch["positions"],
            batch=batch["batch"],
            cell=batch.get("cell"),
            compute_forces=True,
            **kwargs,
        )

    def _write_csv(self, epoch: int, tr: dict, va: dict, elapsed: float) -> None:
        row = [
            epoch,
            tr.get("energy", ""),
            tr.get("force",  ""),
            tr.get("total",  ""),
            va.get("energy", ""),
            va.get("force",  ""),
            va.get("total",  ""),
            self.optimizer.param_groups[0]["lr"],
            round(elapsed, 2),
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _maybe_print(self, epoch: int, tr: dict, va: dict, elapsed: float) -> None:
        if epoch % 10 != 0:
            return
        lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:5d} | "
            f"train total={tr.get('total', 0):.4f} "
            f"(E={tr.get('energy', 0):.4f} F={tr.get('force', 0):.4f}) | "
            f"val total={va.get('total', 0):.4f} "
            f"(E={va.get('energy', 0):.4f} F={va.get('force', 0):.4f}) | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )
