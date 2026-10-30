"""Data loading and the leakage-safe four-way split.

Splits (disjoint):
    D_train  -> fit the classifier
    D_design -> early stopping, temperature T, RAPS lambda/k_reg, gamma, method selection
    D_cal    -> compute the conformal quantile q_hat  (NEVER used for tuning)
    D_test   -> final reported metrics only
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Splits:
    train: torch.utils.data.Dataset
    design: torch.utils.data.Dataset
    calib: torch.utils.data.Dataset
    test: torch.utils.data.Dataset


def build_transforms(dataset: str) -> tuple[Callable, Callable]:
    """Return (train_transform, eval_transform). Normalisation stats fit on D_train only.

    TODO: compute/confirm per-channel mean/std; CIFAR-10 augmentation (crop, flip) train-only.
    """
    raise NotImplementedError


def load_in_distribution(dataset: str, root: str = "data") -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return (train_set, test_set) via torchvision. dataset in {mnist, cifar10}."""
    raise NotImplementedError


def load_ood(dataset: str, parent: str, root: str = "data") -> torch.utils.data.Dataset:
    """Return an OOD test set. FashionMNIST/KMNIST for mnist; SVHN/CIFAR-100 for cifar10."""
    raise NotImplementedError


def four_way_split(
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    fractions: dict[str, float],
    seed: int,
) -> Splits:
    """Create disjoint train/design/calib (+ untouched test) with a fixed seed.

    TODO: use a torch.Generator seeded with `seed`; assert index sets are pairwise disjoint.
    """
    raise NotImplementedError
