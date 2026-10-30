"""Conformal prediction: quantile threshold and set construction.

Theorem (marginal coverage): under exchangeability of (X_i, Y_i),
    1 - alpha  <=  P(Y in C(X))  <=  1 - alpha + 1/(n+1).
"""
from __future__ import annotations

import math

import torch


def conformal_quantile(cal_scores: torch.Tensor, alpha: float) -> float:
    """Split-conformal threshold.

    k = ceil((n+1) * (1 - alpha));  q_hat = k-th smallest of the n calibration scores.
    Return +inf if k > n.

    TODO: use the correct order statistic (1-indexed k). Guard against k <= 0.
    """
    raise NotImplementedError


def build_sets(score_matrix: torch.Tensor, q_hat: float) -> torch.Tensor:
    """Return a boolean membership matrix C (n, K): C[i, y] = (S[i, y] <= q_hat)."""
    raise NotImplementedError


def mondrian_quantiles(cal_scores: torch.Tensor, cal_labels: torch.Tensor, alpha: float, num_classes: int) -> torch.Tensor:
    """Class-conditional (Mondrian) thresholds q_hat_c, one per class.

    For class c, compute the conformal quantile from calibration points whose label is c.
    Gives P(Y in C(X) | Y = c) >= 1 - alpha per class.
    """
    raise NotImplementedError


def build_sets_mondrian(score_matrix: torch.Tensor, q_hats: torch.Tensor) -> torch.Tensor:
    """C[i, y] = (S[i, y] <= q_hats[y])."""
    raise NotImplementedError


def coverage(membership: torch.Tensor, labels: torch.Tensor) -> float:
    """Marginal empirical coverage: mean over i of C[i, labels[i]]."""
    raise NotImplementedError


def class_conditional_coverage(membership: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return per-class empirical coverage, shape (K,)."""
    raise NotImplementedError
