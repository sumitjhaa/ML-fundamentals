"""Temperature scaling (Guo et al., 2017).

Fit T > 0 on D_design by minimising NLL. Preserves argmax; rescales score magnitudes.
"""
from __future__ import annotations

import torch


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the scalar temperature T via LBFGS on logits (design split only).

    p_T(y|x) = softmax(z(x) / T)
    """
    raise NotImplementedError


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return softmax(logits / T)."""
    raise NotImplementedError


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """ECE: sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|."""
    raise NotImplementedError
