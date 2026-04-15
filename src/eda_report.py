"""
EDA outputs for normalized splits and optional windowed datasets (JSON + CSV).

No plotting dependency required; optional PNG histograms if matplotlib is installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .soft_sensor_columns import FEATURE_COLS, TARGET_COL


def _missing_fraction(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.isna().mean()) if len(x) else 1.0


def summarize_normalized_split(
    norm_dir: Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, Any]:
    """Missing rates + describe stats for raw and _z columns."""
    cols_of_interest = list(FEATURE_COLS) + [TARGET_COL]
    zcols = [f"{c}_z" for c in cols_of_interest]
    nobs_cols = [f"n_obs_{c}" for c in cols_of_interest]

    out: dict[str, Any] = {"normalized_dir": str(norm_dir.resolve()), "splits": {}}

    for sp in splits:
        path = norm_dir / f"{sp}.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path, parse_dates=["DateTime"], low_memory=False)
        split_info: dict[str, Any] = {
            "n_rows": int(len(df)),
            "datetime_min": str(df["DateTime"].min()) if len(df) else None,
            "datetime_max": str(df["DateTime"].max()) if len(df) else None,
        }
        miss = {}
        for c in cols_of_interest + zcols:
            if c in df.columns:
                miss[c] = _missing_fraction(df[c])
        for c in nobs_cols:
            if c in df.columns:
                nn = pd.to_numeric(df[c], errors="coerce")
                miss[c] = float((nn.fillna(0) == 0).mean())
        split_info["missing_fraction"] = miss

        desc = {}
        for c in cols_of_interest + zcols:
            if c in df.columns:
                desc[c] = pd.to_numeric(df[c], errors="coerce").describe().to_dict()
        split_info["describe_raw_z"] = desc
        out["splits"][sp] = split_info

    return out


def summarize_windowed_npz(window_dir: Path) -> dict[str, Any]:
    """Load train/val/test.npz and report shapes + mask coverage."""
    out: dict[str, Any] = {"window_dir": str(window_dir.resolve()), "splits": {}}
    for sp in ("train", "val", "test"):
        npz_path = window_dir / f"{sp}.npz"
        if not npz_path.is_file():
            continue
        data = np.load(npz_path, allow_pickle=False)
        X_m = data["X_mask"]
        y_m = data["y_mask"]
        out["splits"][sp] = {
            "n_samples": int(data["X_z"].shape[0]),
            "context_len": int(data["X_z"].shape[1]),
            "n_features": int(data["X_z"].shape[2]),
            "mean_X_valid_per_window": float(np.mean(X_m)),
            "mean_y_valid": float(np.mean(y_m)),
        }
    return out


def write_missing_rates_csv(norm_dir: Path, out_csv: Path) -> None:
    rows: list[dict[str, Any]] = []
    cols_of_interest = list(FEATURE_COLS) + [TARGET_COL]
    for sp in ("train", "val", "test"):
        path = norm_dir / f"{sp}.csv"
        if not path.is_file():
            continue
        header = pd.read_csv(path, nrows=0).columns.tolist()
        want = [c for c in cols_of_interest if c in header]
        nobs = [f"n_obs_{c}" for c in cols_of_interest if f"n_obs_{c}" in header]
        df = pd.read_csv(path, usecols=["DateTime"] + want + nobs, low_memory=False)
        for c in want:
            rows.append({"split": sp, "column": c, "missing_fraction": _missing_fraction(df[c])})
        for nc in nobs:
            nn = pd.to_numeric(df[nc], errors="coerce").fillna(0)
            rows.append({"split": sp, "column": nc, "missing_fraction": float((nn == 0).mean())})

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def maybe_plot_histograms(norm_dir: Path, fig_dir: Path) -> list[str]:
    """Returns list of written PNG paths; empty if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for sp in ("train", "val", "test"):
        path = norm_dir / f"{sp}.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path, low_memory=False)
        ycol = f"{TARGET_COL}_z"
        if ycol not in df.columns:
            continue
        s = pd.to_numeric(df[ycol], errors="coerce").dropna()
        if s.empty:
            continue
        plt.figure(figsize=(6, 3))
        plt.hist(s, bins=80, color="steelblue", alpha=0.85)
        plt.title(f"{sp}: {ycol} (n={len(s)})")
        plt.xlabel(ycol)
        plt.ylabel("count")
        plt.tight_layout()
        fp = fig_dir / f"hist_{sp}_chl_z.png"
        plt.savefig(fp, dpi=120)
        plt.close()
        written.append(str(fp.resolve()))
    return written


def run_full_eda(
    norm_dir: Path,
    window_dir: Path | None,
    out_dir: Path,
    *,
    with_plots: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_summary = summarize_normalized_split(norm_dir)
    norm_summary["eda_version"] = 1
    (out_dir / "eda_summary.json").write_text(
        json.dumps(norm_summary, indent=2, default=str),
        encoding="utf-8",
    )

    write_missing_rates_csv(norm_dir, out_dir / "missing_rates.csv")

    written_plots: list[str] = []
    if with_plots:
        written_plots = maybe_plot_histograms(norm_dir, out_dir / "figures")

    if window_dir and window_dir.is_dir():
        wsum = summarize_windowed_npz(window_dir)
        wsum["figures_chl_z"] = written_plots
        (out_dir / "windowed_summary.json").write_text(json.dumps(wsum, indent=2, default=str), encoding="utf-8")
    elif written_plots:
        (out_dir / "figures_index.json").write_text(json.dumps({"figures_chl_z": written_plots}, indent=2), encoding="utf-8")

    readme = [
        "EDA bundle (full tier)",
        f"- normalized_dir: {norm_dir}",
        f"- eda_summary.json: missing + describe per split",
        f"- missing_rates.csv: long-form missing fractions",
    ]
    if window_dir:
        readme.append(f"- windowed_summary.json: NPZ shapes/masks from {window_dir}")
    (out_dir / "README_eda.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    return out_dir


__all__ = ["summarize_normalized_split", "summarize_windowed_npz", "run_full_eda"]
