"""Custom Trainer with MoE auxiliary loss support.

Extends standard training loop to handle:
  - Combined language-modelling loss + MoE load-balancing loss
  - Logging per-expert utilisation statistics
  - Staged freezing/unfreezing of model components
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SigLlamaTrainer:
    """Training orchestrator for SigLlama.

    Parameters
    ----------
    model : nn.Module
        The SigLlama model.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader or None
        Validation data loader.
    optimizer : torch.optim.Optimizer
        Optimiser instance.
    scheduler : Any or None
        Learning rate scheduler.
    moe_aux_weight : float
        Weight for the MoE auxiliary loss term.
    device : str
        Target device.
    max_epochs : int
        Maximum training epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        moe_aux_weight: float = 0.01,
        device: str = "cuda",
        max_epochs: int = 10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.moe_aux_weight = moe_aux_weight
        self.device = device
        self.max_epochs = max_epochs

    def train(self) -> None:
        """Run the full training loop."""
        # TODO: implement
        #   for epoch in range(self.max_epochs):
        #       self._train_epoch(epoch)
        #       self._validate(epoch)
        raise NotImplementedError

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError

    def _validate(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError

    def _compute_loss(
        self, lm_loss: torch.Tensor, aux_loss: torch.Tensor
    ) -> torch.Tensor:
        """Combine language modelling loss with MoE auxiliary loss."""
        return lm_loss + self.moe_aux_weight * aux_loss
