"""Classifier architectures and training.

Use MPS on Apple Silicon when available.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LeNet(nn.Module):
    """Small CNN for MNIST (28x28 grayscale). Expose penultimate features for Mahalanobis."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 128) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits z(x)."""
        raise NotImplementedError

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate features phi(x) used by the Mahalanobis score."""
        raise NotImplementedError


class SmallResNet(nn.Module):
    """Compact ResNet for CIFAR-10 (32x32 RGB)."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 256) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def get_device() -> torch.device:
    """Prefer MPS, then CUDA, then CPU."""
    raise NotImplementedError


def train_model(model, train_loader, design_loader, cfg: dict) -> dict:
    """Train with cross-entropy; early-stop on D_design. Return history/logits.

    TODO: standard supervised loop; save best checkpoint to checkpoints/.
    """
    raise NotImplementedError


def predict_logits(model, loader) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, labels) for all points in a loader, in model.eval() mode."""
    raise NotImplementedError
