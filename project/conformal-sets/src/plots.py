"""Figures for the paper.

All plots write to results/ as PDF/PNG.
"""
from __future__ import annotations

import pandas as pd


def plot_coverage_vs_size(df: pd.DataFrame, out_path: str) -> None:
    """Efficiency frontier: mean set size (x) vs empirical coverage (y), one line per score."""
    raise NotImplementedError


def plot_size_histogram(sizes: dict[str, "object"], out_path: str) -> None:
    """Overlaid histograms of set size per method."""
    raise NotImplementedError


def plot_class_coverage(class_cov: dict[str, "object"], alpha: float, out_path: str) -> None:
    """Per-class coverage bars with the nominal 1-alpha line; highlights the worst class gap."""
    raise NotImplementedError


def plot_reliability(probs, labels, out_path: str, n_bins: int = 15) -> None:
    """Reliability diagram before/after temperature scaling."""
    raise NotImplementedError
