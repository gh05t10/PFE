"""
Sliding windows on resampled + normalized CSVs: X (L×5) + masks + Chl targets (scalar and/or multi-step).

Rows must be **time-sorted** on a **uniform** resample grid (e.g. 30 min); ``context_len`` is in rows.

- ``horizon_steps=0``: target ``y_z`` at the **last** timestep of the context window.
- ``horizon_steps=h>=1``: target at **h rows after** the context end (``horizon_steps=1`` = one step ahead).

Multi-step extension (for LTSF-style vector forecasting):
- ``pred_len>=1``: export ``Y_z`` with shape (n_samples, pred_len), where the first element corresponds
  to one-step-ahead relative to the context end (i.e. lead=1), shifted by ``horizon_steps``.
  Concretely, for a window ending at index e=i+L-1, we define:
    start = e + horizon_steps + 1
    Y_z[k, j] = df[yz].iloc[start + j] for j=0..pred_len-1

Backward compatibility:
- ``y_z`` (scalar) is always exported as the first step of ``Y_z`` when ``pred_len>=1``.
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
    pred_len: int = 1
    max_gap_steps: int | None = None


def _build_one_split(
    df: pd.DataFrame,
    *,
    context_len: int,
    horizon_steps: int,
    pred_len: int,
    stride: int,
    skip_nan_target: bool,
    max_gap_steps: int | None,
) -> dict[str, np.ndarray]:
    fz = list(z_feature_names())
    yz = target_z_name()
    for c in fz + [yz]:
        if c not in df.columns:
            raise KeyError(f"Missing column {c!r} — run normalize split first.")

    n = len(df)
    dt = pd.to_datetime(df["DateTime"], errors="coerce")
    # infer step size from median delta
    dts = dt.to_numpy(dtype="datetime64[ns]")
    diffs = (dts[1:] - dts[:-1]).astype("timedelta64[ns]")
    # ignore non-positive/NaT
    diffs_ns = diffs.astype("int64")
    diffs_ns = diffs_ns[(diffs_ns > 0) & np.isfinite(diffs_ns)]
    step_ns = int(np.median(diffs_ns)) if diffs_ns.size else 0

    # Optional GT weights passed through normalize split.
    # Prefer resampled weights (aligned to the same grid), else fall back to raw weight if present.
    wcol = "weight_chl_gt_resampled"
    if wcol not in df.columns:
        wcol = "weight_chl_gt" if "weight_chl_gt" in df.columns else ""

    if pred_len < 1:
        raise ValueError(f"pred_len must be >= 1, got {pred_len}")

    # Window rows [i, i+L); window end index e = i+L-1.
    # Scalar target index (legacy): e + horizon_steps
    # Vector target start index: e + horizon_steps + 1; need start+pred_len-1 < n
    imax = n - context_len - horizon_steps - pred_len
    if imax < 0:
        raise ValueError(
            f"Series too short for context_len={context_len}, horizon_steps={horizon_steps}, pred_len={pred_len}: n={n}"
        )

    rows: list[int] = []
    for i in range(0, imax + 1, stride):
        e = i + context_len - 1
        start = e + horizon_steps + 1
        y0 = df[yz].iloc[start]
        if skip_nan_target and (pd.isna(y0) or not np.isfinite(float(y0))):
            continue
        if max_gap_steps is not None and step_ns > 0:
            # Ensure no large gaps inside [i, start+pred_len)
            end2 = start + pred_len - 1
            seg = dts[i : end2 + 1]
            seg_d = (seg[1:] - seg[:-1]).astype("timedelta64[ns]").astype("int64")
            if seg_d.size and np.nanmax(seg_d) > (max_gap_steps * step_ns):
                continue
        rows.append(i)

    if not rows:
        raise RuntimeError("No valid windows — check data, horizon, or skip_nan_target.")

    n_samples = len(rows)
    L = context_len
    X_z = np.full((n_samples, L, len(FEATURE_COLS)), np.nan, dtype=np.float32)
    X_mask = np.zeros((n_samples, L, len(FEATURE_COLS)), dtype=np.bool_)
    # Slide 10: calibration branch = 5 features + Chl_z at each timestep in the window (training / teacher).
    X6_z = np.full((n_samples, L, 6), np.nan, dtype=np.float32)
    X6_mask = np.zeros((n_samples, L, 6), dtype=np.bool_)
    # Scalar legacy target: first step of the future sequence (one-step-ahead after window end).
    y_z = np.full((n_samples,), np.nan, dtype=np.float32)
    y_mask = np.ones((n_samples,), dtype=np.bool_)
    y_w = np.ones((n_samples,), dtype=np.float32)
    # Multi-step future targets (z-space)
    Y_z = np.full((n_samples, pred_len), np.nan, dtype=np.float32)
    Y_mask = np.zeros((n_samples, pred_len), dtype=np.bool_)
    Y_w = np.ones((n_samples, pred_len), dtype=np.float32)
    # Chl_z at the last timestep of the input window (for persistence baseline vs one-step-ahead target).
    chl_z_at_window_end = np.full((n_samples,), np.nan, dtype=np.float32)
    context_end = np.empty(n_samples, dtype="datetime64[ns]")
    target_time = np.empty(n_samples, dtype="datetime64[ns]")  # first-step target time (y_z)
    target_times = np.empty((n_samples, pred_len), dtype="datetime64[ns]")

    for k, i in enumerate(rows):
        sl = df.iloc[i : i + L]
        xz = sl[fz].to_numpy(dtype=np.float64, copy=False)
        X_z[k] = xz.astype(np.float32)
        fin = np.isfinite(xz)
        X_mask[k] = fin
        chl_seq = sl[yz].to_numpy(dtype=np.float64, copy=False)
        X6_z[k, :, :5] = xz.astype(np.float32)
        X6_z[k, :, 5] = chl_seq.astype(np.float32)
        chl_ok = np.isfinite(chl_seq)
        X6_mask[k, :, :5] = fin
        X6_mask[k, :, 5] = chl_ok
        e = i + L - 1
        start = e + horizon_steps + 1
        future = df[yz].iloc[start : start + pred_len].to_numpy(dtype=np.float64, copy=False)
        if future.shape[0] != pred_len:
            # Should not happen due to imax constraint, but keep safe.
            continue
        Y_z[k] = np.where(np.isfinite(future), future, np.nan).astype(np.float32)
        Y_mask[k] = np.isfinite(future)
        if wcol:
            fw = (
                pd.to_numeric(df[wcol].iloc[start : start + pred_len], errors="coerce")
                .to_numpy(dtype=np.float64, copy=False)
            )
            fw = np.where(np.isfinite(fw), fw, 0.0)
            fw = np.clip(fw, 0.0, 1.0)
            Y_w[k] = fw.astype(np.float32)
            y_w[k] = float(Y_w[k, 0])
        # Legacy scalar target = first future step
        y0 = future[0]
        y_z[k] = float(y0) if np.isfinite(y0) else np.nan
        if not np.isfinite(y0):
            y_mask[k] = False
        end_chl = df[yz].iloc[i + L - 1]
        chl_z_at_window_end[k] = float(end_chl) if np.isfinite(end_chl) else np.nan
        context_end[k] = np.datetime64(dt.iloc[i + L - 1])
        target_time[k] = np.datetime64(dt.iloc[start])
        target_times[k] = dt.iloc[start : start + pred_len].to_numpy(dtype="datetime64[ns]", copy=False)

    return {
        "X_z": X_z,
        "X_mask": X_mask,
        "X6_z": X6_z,
        "X6_mask": X6_mask,
        "y_z": y_z,
        "y_mask": y_mask,
        "y_w": y_w,
        "Y_z": Y_z,
        "Y_mask": Y_mask,
        "Y_w": Y_w,
        "chl_z_at_window_end": chl_z_at_window_end,
        "context_end_time": context_end,
        "target_time": target_time,
        "target_times": target_times,
    }


def build_windows_to_npz(cfg: WindowDatasetConfig, manifest_extra: dict[str, Any]) -> None:
    df = pd.read_csv(cfg.split_csv, parse_dates=["DateTime"], low_memory=False)
    df = df.sort_values("DateTime").reset_index(drop=True)

    arrs = _build_one_split(
        df,
        context_len=cfg.context_len,
        horizon_steps=cfg.horizon_steps,
        pred_len=cfg.pred_len,
        stride=cfg.stride,
        skip_nan_target=cfg.skip_nan_target,
        max_gap_steps=cfg.max_gap_steps,
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
        "pred_len": cfg.pred_len,
        "stride": cfg.stride,
        "max_gap_steps": cfg.max_gap_steps,
        "feature_cols": list(FEATURE_COLS),
        "target_col_z": target_z_name(),
        "skip_nan_target": cfg.skip_nan_target,
        "shape_X_z": list(arrs["X_z"].shape),
        "shape_X6_z": list(arrs["X6_z"].shape),
        "shape_Y_z": list(arrs["Y_z"].shape),
    }
    meta_path = cfg.out_npz.with_suffix(".json")
    meta_path.write_text(json.dumps(man, indent=2), encoding="utf-8")


__all__ = [
    "WindowDatasetConfig",
    "z_feature_names",
    "target_z_name",
    "build_windows_to_npz",
]
