"""Conformity scores. Convention: smaller = more conforming.

All scores return a matrix S of shape (n, K) where S[i, y] = s(x_i, y).
"""
from __future__ import annotations

import torch


def naive_scores(probs: torch.Tensor) -> torch.Tensor:
    """s(x, y) = 1 - p(y|x).  Shape (n, K)."""
    raise NotImplementedError


def aps_scores(probs: torch.Tensor, randomised: bool = True, seed: int = 0) -> torch.Tensor:
    """Adaptive Prediction Sets (Romano et al., 2020).

    Sort descending p_(1) >= ... >= p_(K).
    randomised:      s(x,y) = sum_{j<k_y} p_(j) + u * p_(k_y),  u ~ U(0,1)
    non-randomised:  s(x,y) = sum_{j<k_y} p_(j) + p_(k_y)
    where k_y is the rank of y.
    """
    raise NotImplementedError


def raps_scores(probs: torch.Tensor, lam: float, k_reg: int, randomised: bool = True) -> torch.Tensor:
    """Regularised APS: s = s_APS + lam * max(k_y - k_reg, 0)."""
    raise NotImplementedError


def mahalanobis_scores(features: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor) -> torch.Tensor:
    """d_M(x, y) = sqrt((phi(x) - mu_y)^T Sigma^-1 (phi(x) - mu_y)).  Shape (n, K)."""
    raise NotImplementedError


def distance_augmented_scores(
    base_scores: torch.Tensor,
    d_m_normalised: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Proposed score: s_gamma(x, y) = base_scores(x, y) + gamma * d_m_normalised(x, y).

    gamma = 0 recovers the base score (clean ablation).
    """
    raise NotImplementedError


def normalise_distance(d_m: torch.Tensor, ref_mean: float, ref_std: float) -> torch.Tensor:
    """Standardise d_M using design-split statistics: (d_M - mean) / std."""
    raise NotImplementedError
