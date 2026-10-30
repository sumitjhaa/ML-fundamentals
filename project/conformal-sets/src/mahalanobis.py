"""Mahalanobis distance in feature space (Lee et al., 2018).

Fit per-class Gaussians with one shared covariance Sigma on D_train penultimate features.
"""
from __future__ import annotations

import torch


def fit_class_gaussians(features: torch.Tensor, labels: torch.Tensor, num_classes: int, reg: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mu, sigma_inv).

    mu:        (K, D) class means
    sigma_inv: (D, D) inverse of the shared (tied) covariance, regularised by `reg` on the diagonal.
    """
    raise NotImplementedError


def mahalanobis_per_class(features: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor) -> torch.Tensor:
    """Return (n, K) matrix of d_M(x_i, y) for every class y."""
    raise NotImplementedError
