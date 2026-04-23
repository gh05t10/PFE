#!/usr/bin/env python3
"""
Build a unified resampled panel (5 soft-sensor inputs + Chl GT) on a configurable grid.

Default frequency: ``30min`` (see ``src/resample_config.py``), overridable via
``--freq`` or environment ``PFE_RESAMPLE_FREQ`` (e.g. ``10min`` for a later experiment).

Outputs under ``processed/chl_shallow/resampled_<slug>/`` so different grids do not overwrite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.resample_config import DEFAULT_RESAMPLE_FREQ, ENV_RESAMPLE_FREQ, freq_slug, get_resample_freq
from src.unified_resample import UnifiedResampleConfig, run_unified_resample


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Resample soft-sensor + Chl to a unified grid.")
    p.add_argument(
        "--freq",
        default=None,
        help=f"pandas offset (default: env {ENV_RESAMPLE_FREQ} or {DEFAULT_RESAMPLE_FREQ})",
    )
    p.add_argument(
        "--agg",
        choices=("median", "mean"),
        default="median",
        help="aggregation inside each resample bin (default: median)",
    )
    p.add_argument("--data-dir", type=Path, default=base / "data")
    p.add_argument(
        "--out-root",
        type=Path,
        default=base / "processed" / "chl_shallow",
        help="parent directory; a subfolder resampled_<slug> is created",
    )
    p.add_argument(
        "--rule-a",
        action="store_true",
        help="keep only timestamps in Rule A(p) months (Chl coverage); aligns with monthly GT",
    )
    p.add_argument(
        "--rule-a-p",
        type=float,
        default=0.8,
        help="Rule A(p) threshold when --rule-a (default 0.8)",
    )
    p.add_argument(
        "--gt-weights-csv",
        type=Path,
        default=None,
        help="optional path to chl_shallow_transformer_gt.csv to carry weight_chl_gt into the resampled panel",
    )
    args = p.parse_args()

    freq = get_resample_freq(cli=args.freq)
    slug = freq_slug(freq)
    suffix = "_ruleA" if args.rule_a else ""
    out_dir = args.out_root / f"resampled_{slug}{suffix}"
    cfg = UnifiedResampleConfig(
        data_dir=args.data_dir,
        out_dir=out_dir,
        freq=freq,
        agg=args.agg,
        rule_a=args.rule_a,
        rule_a_p=args.rule_a_p,
        gt_weights_csv=args.gt_weights_csv,
    )
    path = run_unified_resample(cfg)
    print(f"freq={freq} → {path}")
    print(f"meta: {out_dir / 'resample_meta.txt'}")


if __name__ == "__main__":
    main()
