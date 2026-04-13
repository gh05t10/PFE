#!/usr/bin/env python3
"""
Time-based train/val/test split + per-channel z-score (fit on train only).

Reads ``processed/chl_shallow/resampled_<slug>/soft_sensor_resampled.csv`` (from
``run_unified_resample.py``) and writes ``train.csv``, ``val.csv``, ``test.csv`` plus
``scaler_params.json`` and ``split_manifest.json``.

Default split (exclusive bounds):
  - train: DateTime < 2019-01-01
  - val:   2019-01-01 <= DateTime < 2021-01-01
  - test:  DateTime >= 2021-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.resample_config import freq_slug, get_resample_freq
from src.time_split_normalize import SplitNormalizeConfig, run_split_and_normalize


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Train-only z-score + time splits for resampled soft-sensor data.")
    p.add_argument("--freq", default=None, help="resample slug folder (default: PFE_RESAMPLE_FREQ or 30min)")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="override path to soft_sensor_resampled.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: resampled_<slug>/normalized_split)",
    )
    p.add_argument(
        "--train-end",
        default="2019-01-01",
        help="exclusive upper bound for train (pandas-parsable datetime)",
    )
    p.add_argument(
        "--val-end",
        default="2021-01-01",
        help="exclusive upper bound for val / start of test",
    )
    args = p.parse_args()

    freq = get_resample_freq(cli=args.freq)
    slug = freq_slug(freq)
    in_csv = args.input_csv or (base / "processed" / "chl_shallow" / f"resampled_{slug}" / "soft_sensor_resampled.csv")
    out_dir = args.out_dir or (base / "processed" / "chl_shallow" / f"resampled_{slug}" / "normalized_split")

    train_end = pd.Timestamp(args.train_end)
    val_end = pd.Timestamp(args.val_end)
    if train_end >= val_end:
        raise SystemExit("--train-end must be strictly before --val-end")

    cfg = SplitNormalizeConfig(
        input_csv=in_csv,
        out_dir=out_dir,
        train_end=train_end,
        val_end=val_end,
        resample_freq=freq,
    )
    m = run_split_and_normalize(cfg)
    print(f"Wrote splits under {out_dir}")
    print(f"rows: {m['n_rows']}")


if __name__ == "__main__":
    main()
