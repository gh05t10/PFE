#!/usr/bin/env python3
"""
Full-tier EDA after normalized (and optionally windowed) preprocess outputs.

Writes under ``normalized_split/eda/`` by default:
  - ``eda_summary.json`` — missing + describe per split
  - ``missing_rates.csv``
  - ``windowed_summary.json`` — if ``--windowed-dir`` points to NPZ bundle
  - ``figures/hist_*_chl_z.png`` — if ``--plots`` and matplotlib installed
  - ``README_eda.txt``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eda_report import run_full_eda
from src.resample_config import freq_slug, get_resample_freq


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="EDA summary for normalized / windowed pipeline outputs.")
    p.add_argument("--freq", default=None)
    p.add_argument(
        "--rule-a",
        action="store_true",
        help="use resampled_<slug>_ruleA/normalized_split",
    )
    p.add_argument(
        "--normalized-dir",
        type=Path,
        default=None,
        help="folder with train/val/test.csv (post z-score)",
    )
    p.add_argument(
        "--windowed-dir",
        type=Path,
        default=None,
        help="folder with train.npz / val.npz / test.npz (optional)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <normalized-dir>/eda",
    )
    p.add_argument("--plots", action="store_true", help="save Chl_z histograms (needs matplotlib)")
    args = p.parse_args()

    freq = get_resample_freq(cli=args.freq)
    slug = freq_slug(freq)
    ra = "_ruleA" if args.rule_a else ""
    norm_dir = args.normalized_dir or (
        base / "processed" / "chl_shallow" / f"resampled_{slug}{ra}" / "normalized_split"
    )
    out_dir = args.out_dir or (norm_dir / "eda")
    win_dir = args.windowed_dir
    if win_dir is None:
        cand = sorted(norm_dir.glob("windowed_L*_H*_S*"))
        if len(cand) == 1:
            win_dir = cand[0]
        elif len(cand) > 1:
            win_dir = cand[-1]

    run_full_eda(norm_dir, win_dir, out_dir, with_plots=args.plots)
    print(f"EDA written to {out_dir}")


if __name__ == "__main__":
    main()
