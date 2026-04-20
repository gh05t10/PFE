#!/usr/bin/env python3
"""
Evaluate naive baselines on the same windowed NPZ splits as GRU.

  - mean_train: constant = mean(y_z on train windows)
  - persistence: y_hat_z = chl_z_at_window_end (requires rebuilt NPZ with that key)

Writes ``baseline_metrics.json`` next to the window directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval_baselines import run_all_baselines
from src.resample_config import freq_slug, get_resample_freq


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Naive baselines vs windowed NPZ.")
    p.add_argument("--freq", default=None)
    p.add_argument("--rule-a", action="store_true")
    p.add_argument("--window-dir", type=Path, default=None)
    args = p.parse_args()

    freq = get_resample_freq(cli=args.freq)
    slug = freq_slug(freq)
    ra = "_ruleA" if args.rule_a else ""
    norm_dir = base / "processed" / "chl_shallow" / f"resampled_{slug}{ra}" / "normalized_split"
    if args.window_dir is None:
        cand = sorted(norm_dir.glob("windowed_L*_H*_S*"))
        if not cand:
            raise SystemExit(f"No windowed_* under {norm_dir}")
        window_dir = cand[-1]
    else:
        window_dir = args.window_dir

    train_npz = window_dir / "train.npz"
    val_npz = window_dir / "val.npz"
    test_npz = window_dir / "test.npz"
    scaler_json = norm_dir / "scaler_params.json"

    for f in (train_npz, val_npz, scaler_json):
        if not f.is_file():
            raise SystemExit(f"Missing {f}")

    out = run_all_baselines(train_npz, val_npz, test_npz, scaler_json)
    out["window_dir"] = str(window_dir.resolve())
    out_path = window_dir / "baseline_metrics.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
