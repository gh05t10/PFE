"""
Sliding windows on resampled + normalized CSVs: X (L×5) + masks + scalar Chl target z-score.

Rows must be **time-sorted** on a **uniform** resample grid (e.g. 30 min); ``context_len`` is in rows.

- ``horizon_steps=0``: target ``y_z`` at the **last** timestep of the context window.
- ``horizon_steps=h>=1``: target at **h rows after** the context end (``horizon_steps=1`` = one step ahead).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .soft_sensor_columns import FEATURE_COLS, TARGET_COL


def z_feature_names(suffix: str = "_z") -> tuple[str, ...]:
    return tuple(f"{c}{suffix}" for c in FEATURE_COLS)


def target_z_name(suffix: str = "_z") -> str:
    return f"{TARGET_COL}{suffix}"


@dataclass(frozen=True)
class WindowDatasetConfig:
    split_csv: Path
    out_npz: Path
    context_len: int
    horizon_steps: int
    stride: int
    skip_nan_target: bool


def _build_one_split(
    df: pd.DataFrame,
    *,
    context_len: int,
    horizon_steps: int,
    stride: int,
    skip_nan_target: bool,
) -> dict[str, np.ndarray]:
    fz = list(z_feature_names())
    yz = target_z_name()
    for c in fz + [yz]:
        if c not in df.columns:
            raise KeyError(f"Missing column {c!r} — run normalize split first.")

    n = len(df)
    dt = pd.to_datetime(df["DateTime"], errors="coerce")

    # Window rows [i, i+L); target row index = i + L - 1 + horizon_steps; require target < n
    imax = n - context_len - horizon_steps
    if imax < 0:
        raise ValueError(
            f"Series too short for context_len={context_len} and horizon_steps={horizon_steps}: n={n}"
        )

    rows: list[int] = []
    for i in range(0, imax + 1, stride):
        tgt = i + context_len - 1 + horizon_steps
        yv = df[yz].iloc[tgt]
        if skip_nan_target and (pd.isna(yv) or not np.isfinite(float(yv))):
            continue
        rows.append(i)

    if not rows:
        raise RuntimeError("No valid windows — check data, horizon, or skip_nan_target.")

    n_samples = len(rows)
    L = context_len
    X_z = np.full((n_samples, L, len(FEATURE_COLS)), np.nan, dtype=np.float32)
    X_mask = np.zeros((n_samples, L, len(FEATURE_COLS)), dtype=np.bool_)
    y_z = np.full((n_samples,), np.nan, dtype=np.float32)
    y_mask = np.ones((n_samples,), dtype=np.bool_)
    # Chl_z at the last timestep of the input window (for persistence baseline vs one-step-ahead target).
    chl_z_at_window_end = np.full((n_samples,), np.nan, dtype=np.float32)
    context_end = np.empty(n_samples, dtype="datetime64[ns]")
    target_time = np.empty(n_samples, dtype="datetime64[ns]")

    for k, i in enumerate(rows):
        sl = df.iloc[i : i + L]
        xz = sl[fz].to_numpy(dtype=np.float64, copy=False)
        X_z[k] = xz.astype(np.float32)
        fin = np.isfinite(xz)
        X_mask[k] = fin
        tgt = i + L - 1 + horizon_steps
        yv = df[yz].iloc[tgt]
        y_z[k] = float(yv) if np.isfinite(yv) else np.nan
        if not np.isfinite(yv):
            y_mask[k] = False
        end_chl = df[yz].iloc[i + L - 1]
        chl_z_at_window_end[k] = float(end_chl) if np.isfinite(end_chl) else np.nan
        context_end[k] = np.datetime64(dt.iloc[i + L - 1])
        target_time[k] = np.datetime64(dt.iloc[tgt])

    return {
        "X_z": X_z,
        "X_mask": X_mask,
        "y_z": y_z,
        "y_mask": y_mask,
        "chl_z_at_window_end": chl_z_at_window_end,
        "context_end_time": context_end,
        "target_time": target_time,
    }


def build_windows_to_npz(cfg: WindowDatasetConfig, manifest_extra: dict[str, Any]) -> None:
    df = pd.read_csv(cfg.split_csv, parse_dates=["DateTime"], low_memory=False)
    df = df.sort_values("DateTime").reset_index(drop=True)

    arrs = _build_one_split(
        df,
        context_len=cfg.context_len,
        horizon_steps=cfg.horizon_steps,
        stride=cfg.stride,
        skip_nan_target=cfg.skip_nan_target,
    )

    cfg.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.out_npz, **arrs)
    man = {
        **manifest_extra,
        "split_csv": str(cfg.split_csv.resolve()),
        "out_npz": str(cfg.out_npz.resolve()),
        "n_samples": int(arrs["X_z"].shape[0]),
        "context_len": cfg.context_len,
        "horizon_steps": cfg.horizon_steps,
        "stride": cfg.stride,
        "feature_cols": list(FEATURE_COLS),
        "target_col_z": target_z_name(),
        "skip_nan_target": cfg.skip_nan_target,
        "shape_X_z": list(arrs["X_z"].shape),
    }
    meta_path = cfg.out_npz.with_suffix(".json")
    meta_path.write_text(json.dumps(man, indent=2), encoding="utf-8")


__all__ = [
    "WindowDatasetConfig",
    "z_feature_names",
    "target_z_name",
    "build_windows_to_npz",
]
