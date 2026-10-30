#!/usr/bin/env python3
"""Sweeps: alpha x gamma grid and seed aggregation.

Usage:
    python scripts/sweep.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())

    # TODO(Phase 3): loop over cfg['alphas'] x cfg['gamma_grid'] x cfg['seeds'];
    # record coverage, mean size, worst-class gap; aggregate mean +/- std into results/sweep.csv
    raise NotImplementedError("Implement after Phase 2")


if __name__ == "__main__":
    main()
