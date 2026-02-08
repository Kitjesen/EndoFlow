"""
Base trainer class — common training loop utilities.
"""

import torch
import time
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Abstract base class for model trainers.

    Subclasses must implement:
      - _build_dataset()
      - _train_one_epoch()
      - _validate_one_epoch()
    """

    def __init__(self, model, checkpoint_path, lr=1e-4, weight_decay=1e-4):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = torch.nn.MSELoss()

    @abstractmethod
    def _build_dataset(self, **kwargs):
        """Create and return (train_loader, val_loader)."""
        ...

    @abstractmethod
    def _train_one_epoch(self, loader):
        """Run one training epoch. Return dict of loss values."""
        ...

    @abstractmethod
    def _validate_one_epoch(self, loader):
        """Run one validation epoch. Return dict of loss values."""
        ...

    def save_checkpoint(self, epoch):
        """Save model state dict."""
        path = f"{self.checkpoint_path}/model_{epoch}.pth"
        torch.save(self.model.state_dict(), path)
        return path
