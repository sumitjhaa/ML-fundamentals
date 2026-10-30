"""Evaluation: coverage, efficiency, class-conditional coverage, OOD behaviour."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Metrics:
    coverage: float
    mean_size: float
    median_size: float
    p90_size: float
    singleton_rate: float
    empty_rate: float
    worst_class_gap: float


def set_sizes(membership: torch.Tensor) -> torch.Tensor:
    """Return per-example set size = row sums of the boolean membership matrix."""
    raise NotImplementedError


def worst_class_gap(class_coverage: torch.Tensor, alpha: float) -> float:
    """max_c |cov_c - (1 - alpha)| across classes."""
    raise NotImplementedError


def evaluate_scores(membership: torch.Tensor, labels: torch.Tensor, alpha: float, num_classes: int) -> Metrics:
    """Compute the full Metrics bundle for one method/alpha."""
    raise NotImplementedError


def subgroup_coverage(membership: torch.Tensor, labels: torch.Tensor, confidence: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    """Coverage as a function of model-confidence decile."""
    raise NotImplementedError


def ood_report(membership: torch.Tensor, alpha: float) -> dict[str, float]:
    """Coverage/set-size statistics on an OOD set (exchangeability violated -> report, do not claim)."""
    raise NotImplementedError
