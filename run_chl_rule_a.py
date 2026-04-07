#!/usr/bin/env python3
"""Export shallow Chl data under Rule A(p) (calendar-month coverage threshold)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.chl_rule_a_months import run_rule_a_export


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--p",
        type=float,
        default=0.8,
        help="Month coverage threshold: n_days_with_data / n_days_required >= p (default: 0.8).",
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    run_rule_a_export(base / "data", base / "processed" / "chl_shallow", p=args.p)
    print(f"Wrote Rule-A(p={args.p}) outputs under {base / 'processed' / 'chl_shallow'}")


if __name__ == "__main__":
    main()
