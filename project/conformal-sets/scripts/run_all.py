#!/usr/bin/env python3
"""End-to-end entry point: regenerates every figure and table into results/.

Usage:
    python scripts/run_all.py --config configs/default.yaml
    python scripts/run_all.py --config configs/default.yaml --dry-run
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
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())

    if args.dry_run:
        print("Plan:")
        print(f"  datasets : {cfg['datasets']}")
        print(f"  seeds    : {cfg['seeds']}")
        print(f"  alphas   : {cfg['alphas']}")
        print(f"  scores   : {cfg['scores']}")
        print(f"  gammas   : {cfg['gamma_grid']}")
        print("OK (dry run)")
        return

    # TODO(Phase 1): data + splits + baseline + naive conformal + coverage sanity.
    # TODO(Phase 2): APS, RAPS, temperature scaling, efficiency table/plots.
    # TODO(Phase 3): Mahalanobis, s_gamma, Mondrian, conditional coverage, OOD, sweeps.
    # TODO(Phase 4): write results/ tables + figures consumed by the paper.
    raise NotImplementedError("Implement phases in order; see PLAN.md")


if __name__ == "__main__":
    main()
