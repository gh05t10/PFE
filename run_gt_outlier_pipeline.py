#!/usr/bin/env python3
"""Run outlier labeling + optional winsorization on Rule-A shallow Chl timeseries.

Hampel uses a window in **samples**, so it must be derived from the data frequency.
This script takes ``--freq`` and converts ``--hampel-hours`` into a sample window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.chl_gt_outlier import GtOutlierExportConfig, run_gt_outlier_export
from src.resample_config import get_resample_freq

if __name__ == "__main__":
    base = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(description="Outlier flags + weights for shallow Chl GT.")
    p.add_argument("--freq", default=None, help="sampling freq used for Hampel window (default: env or 30min)")
    p.add_argument("--hampel-hours", type=float, default=24.0, help="Hampel window span in hours")
    p.add_argument("--hampel-n-sigma", type=float, default=3.0)
    p.add_argument("--winsor-inner", action="store_true")
    args = p.parse_args()

    freq = get_resample_freq(cli=args.freq)
    import pandas as pd

    step = pd.Timedelta(freq)
    if step.total_seconds() <= 0:
        raise SystemExit(f"Bad --freq: {freq}")
    hampel_window = int(round((args.hampel_hours * 3600.0) / step.total_seconds()))
    hampel_window = max(7, hampel_window | 1)  # odd, >=7

    cfg = GtOutlierExportConfig(
        input_csv=base / "processed" / "chl_shallow" / "chl_shallow_rule_a_timeseries.csv",
        out_dir=base / "processed" / "chl_shallow",
        hampel_window=hampel_window,
        hampel_n_sigma=args.hampel_n_sigma,
        winsor_inner=bool(args.winsor_inner),
    )
    run_gt_outlier_export(cfg)
    print(
        f"Wrote chl_gt_outlier_flags.csv, chl_shallow_transformer_gt.csv "
        f"and related outputs under {cfg.out_dir}"
    )
