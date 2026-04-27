#!/usr/bin/env python3
"""
Build windowed dataset (minimal preprocess end-state) from ``normalized_split/*.csv``.

Writes one ``.npz`` + sidecar ``.json`` per split under
``.../normalized_<slug>/windowed_L{L}_H{T}_P{H}_S{S}/``.

Defaults: L=96 (~2 days @ 30min), horizon_steps=0, pred_len=48 (1 day), stride=1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.resample_config import freq_slug, get_resample_freq
from src.window_dataset import WindowDatasetConfig, build_windows_to_npz


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Windowed X/mask/y from normalized split CSVs.")
    p.add_argument("--freq", default=None, help="resample slug (default: env or 30min)")
    p.add_argument(
        "--rule-a",
        action="store_true",
        help="use resampled_<slug>_ruleA/normalized_split",
    )
    p.add_argument(
        "--normalized-dir",
        type=Path,
        default=None,
        help="folder containing train.csv / val.csv / test.csv",
    )
    p.add_argument("--context-len", type=int, default=96, help="window length in rows (Δt steps)")
    p.add_argument(
        "--horizon-steps",
        type=int,
        default=0,
        help="shift applied before the future horizon (0=starts at one-step-ahead; 1=skip one step, etc.)",
    )
    p.add_argument(
        "--pred-len",
        type=int,
        default=48,
        help="number of future steps in the prediction horizon (e.g. 48 = 1 day @ 30min)",
    )
    p.add_argument("--stride", type=int, default=1, help="sliding window stride in rows")
    p.add_argument(
        "--max-gap-steps",
        type=int,
        default=None,
        help="if set, drop windows where any consecutive DateTime gap exceeds this many steps",
    )
    p.add_argument(
        "--keep-nan-targets",
        action="store_true",
        help="if set, keep samples with NaN y_z (y_mask False); default skips them",
    )
    args = p.parse_args()

    freq = get_resample_freq(cli=args.freq)
    slug = freq_slug(freq)
    ra = "_ruleA" if args.rule_a else ""
    norm_dir = args.normalized_dir or (
        base / "processed" / "chl_shallow" / f"resampled_{slug}{ra}" / "normalized_split"
    )
    if not norm_dir.is_dir():
        raise SystemExit(f"Missing normalized split dir: {norm_dir}")

    out_root = norm_dir / f"windowed_L{args.context_len}_H{args.horizon_steps}_P{args.pred_len}_S{args.stride}"
    out_root.mkdir(parents=True, exist_ok=True)

    pack = {
        "step": "window_dataset",
        "resample_freq": freq,
        "resample_slug": slug,
        "normalized_split_dir": str(norm_dir.resolve()),
        "split_manifest": str((norm_dir / "split_manifest.json").resolve()),
        "scaler_params": str((norm_dir / "scaler_params.json").resolve()),
    }
    try:
        pack["split_manifest_body"] = json.loads((norm_dir / "split_manifest.json").read_text(encoding="utf-8"))
    except OSError:
        pass

    for sp in ("train", "val", "test"):
        csv_path = norm_dir / f"{sp}.csv"
        if not csv_path.is_file():
            raise SystemExit(f"Missing {csv_path}")
        npz_path = out_root / f"{sp}.npz"
        cfg = WindowDatasetConfig(
            split_csv=csv_path,
            out_npz=npz_path,
            context_len=args.context_len,
            horizon_steps=args.horizon_steps,
            pred_len=args.pred_len,
            stride=args.stride,
            skip_nan_target=not args.keep_nan_targets,
            max_gap_steps=args.max_gap_steps,
        )
        build_windows_to_npz(cfg, {**pack, "split": sp})
        print(
            f"Wrote {npz_path} ({cfg.context_len=}, {cfg.horizon_steps=}, {cfg.pred_len=}, {cfg.stride=})"
        )


if __name__ == "__main__":
    main()
