"""
Monthly preprocessing aligned with Aquaculture-report17_1.pdf (soft sensor + long-horizon forecast).

Slide mapping
-------------
- *Common measurement parameters* + *Meteorological parameters* → monthly aggregates of
  air pressure, wind, humidity, rain, PAR, temperature, etc.
- *Specific measurement parameters* (water) → shallow/deep water quality, including
  ChlRFUShallow_RFU as **ground truth** for the soft-sensor target.
- *Hard calibration* (example: 14 days prediction / 1 day calibration) → operational idea:
  periodically align model or sensor to truth; in preprocessing we expose **trailing
  high-frequency calibration statistics** (optional) and **month-of-year anomalies**
  to mimic *compensation* against drift (soft calibration theme).

Horizon ≥ 1 month
-----------------
- Work in **calendar-month** resolution: target = monthly Chl (two-stage: daily mean →
  monthly mean of dailies) to avoid oversampling bias.
- For supervised forecasting with h ≥ 1, **do not** use the same month's Chl as input;
  build **lagged** targets/features in modelling code from this panel (see
  ``add_monthly_lags``).
- **Meteorological + water covariates** are synchronized to the same month index so
  multivariate models match the slide's “common + specific” structure.

This module complements ``chl_shallow_pipeline`` (target-only path); it does not import
``data_visualization``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .chl_shallow_pipeline import (
    EXCLUDE_FLAGS,
    TARGET_COL,
    daily_table,
    load_trimmed_chl_frames,
    monthly_from_daily,
    PipelineConfig,
)

# Slide “common” + meteorological (subset; extend as needed)
COMMON_MET_COLS: tuple[str, ...] = (
    "AirTemp_C",
    "BarometricPress_kPa",
    "RelativeHum_%",
    "WindSp_km/h",
    "WindDir_Degree",
    "DailyRain_mm",
    "PARAir_umol/s/m2",
)

# Water / lake “specific” (excludes Chl target from *inputs* in ``wide_monthly_panel``)
WATER_SPECIFIC_COLS: tuple[str, ...] = (
    "TempShallow_C",
    "pHShallow",
    "ODOShallow_mg/L",
    "ODOSatShallow_%",
    "SpCondShallow_uS/cm",
    "TurbShallow_NTU+",
    "PARW1_umol/s/m2",
    "PARW2_umol/s/m2",
    "BGPCShallowRFU_RFU",
)


def _read_yearly_frames(data_dir: Path, value_cols: Iterable[str]) -> pd.DataFrame:
    """Concatenate all preprocessed CSVs with DateTime + requested value + flag columns."""
    paths = sorted(data_dir.glob("BPBuoyData_*_Preprocessed.csv"))
    if not paths:
        raise FileNotFoundError(f"No BPBuoyData_*_Preprocessed.csv under {data_dir}")

    frames: list[pd.DataFrame] = []
    for p in paths:
        need = ["DateTime"] + list(value_cols)
        flag_cols = [f"{c}_Flag" for c in value_cols]
        # Only request flags that exist in file
        header = pd.read_csv(p, nrows=0).columns.tolist()
        usecols = [c for c in need + flag_cols if c in header]
        df = pd.read_csv(p, usecols=usecols, low_memory=False)
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df = df.dropna(subset=["DateTime"])
        df["source_file"] = p.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("DateTime").drop_duplicates(subset=["DateTime"], keep="first")


def _mask_valid_for_column(df: pd.DataFrame, col: str) -> pd.Series:
    """True where *col* is usable (numeric, not flagged B7/C/M if flag column exists)."""
    fc = f"{col}_Flag"
    if fc in df.columns:
        bad = df[fc].map(lambda x: str(x).strip() if pd.notna(x) else None).isin(EXCLUDE_FLAGS)
    else:
        bad = False
    y = pd.to_numeric(df[col], errors="coerce")
    ok = (~bad) & y.notna()
    return ok


def _column_to_monthly(
    df: pd.DataFrame,
    col: str,
    min_days: int,
) -> pd.Series:
    """Two-stage monthly mean for one variable; index = month-end."""
    ok = _mask_valid_for_column(df, col)
    sub = df.loc[ok, ["DateTime", col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=[col])
    sub = sub.set_index("DateTime").sort_index()
    if sub.empty:
        return pd.Series(dtype=float)
    daily = daily_table(sub.rename(columns={col: TARGET_COL}))
    daily = daily.rename(columns={"chl_rfu_daily_mean": f"{col}_daily_mean"})
    cfg = PipelineConfig(
        data_dir=Path("."),
        out_dir=Path("."),
        monthly_method="two_stage",
        min_samples_per_month=0,
        min_days_with_data_per_month=min_days,
    )
    # Reuse monthly_from_daily shape by temp rename
    tmp = daily.rename(columns={f"{col}_daily_mean": "chl_rfu_daily_mean"})
    m = monthly_from_daily(tmp, cfg)
    s = m["chl_rfu_monthly_mean"].rename(col)
    return s


def wide_monthly_panel(
    data_dir: Path,
    *,
    min_days_with_data_per_month: int = 3,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    One row per calendar month (month-end index), columns = monthly means of dailies.

    *include_target*: include ``ChlRFUShallow_RFU`` as a column (ground truth for soft
    sensor). For **inputs** to a model predicting month *t+h*, strip same-month Chl or
    use ``add_monthly_lags`` only.
    """
    cols = list(COMMON_MET_COLS) + list(WATER_SPECIFIC_COLS)
    if include_target and TARGET_COL not in cols:
        cols = cols + [TARGET_COL]

    df = _read_yearly_frames(data_dir, cols)
    monthly_cols: dict[str, pd.Series] = {}
    for c in cols:
        if c not in df.columns:
            continue
        monthly_cols[c] = _column_to_monthly(df, c, min_days_with_data_per_month)

    if TARGET_COL not in monthly_cols or monthly_cols[TARGET_COL].empty:
        raise RuntimeError("Could not build monthly target Chl series.")

    ref_idx = monthly_cols[TARGET_COL].index.sort_values()
    panel = pd.DataFrame(index=ref_idx)
    for c, s in monthly_cols.items():
        panel[c] = s.reindex(ref_idx)
    panel["year"] = panel.index.year
    panel["month"] = panel.index.month
    panel["year_month"] = panel.index.strftime("%Y-%m")
    return panel


def monthly_climatology_anomaly(
    monthly_chl: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Soft-calibration-style drift handling: anomaly vs multi-year mean for that month.

    Returns (climatology_by_month, anomaly) indexed like *monthly_chl*.

    **Leakage note:** if ``anomaly`` is used as a model input, recompute climatology
    using **training months only**, or use rolling/expanding statistics (not implemented
    here) so validation/test months do not influence the reference mean.
    """
    clim_series = monthly_chl.groupby(monthly_chl.index.month).transform("mean")
    anomaly = monthly_chl - clim_series
    return clim_series, anomaly


def trailing_window_highfreq_stats(
    clean_chl_indexed: pd.DataFrame,
    *,
    days: int = 14,
) -> pd.DataFrame:
    """
    Slide-inspired *calibration window*: for each month-end *t*, summarize Chl RFU over
    the previous *days* calendar days of **high-frequency** data (mean, std, n).

    Use as optional bias features or post-hoc calibration hooks (not a leak if only
    past days relative to month label are used — here we use days strictly before month
    boundary if we anchor at month end; adjust in modelling if your label is month start).
    """
    s = clean_chl_indexed[TARGET_COL].sort_index()
    out_rows = []
    for period_end in s.resample("ME").mean().index:
        start = period_end - pd.Timedelta(days=days)
        win = s.loc[(s.index > start) & (s.index <= period_end)]
        out_rows.append(
            {
                "month_end": period_end,
                f"chl_trailing_{days}d_mean": win.mean(),
                f"chl_trailing_{days}d_std": win.std(),
                f"chl_trailing_{days}d_n": len(win),
            }
        )
    return pd.DataFrame(out_rows).set_index("month_end")


def add_monthly_lags(
    panel: pd.DataFrame,
    col: str,
    lags: Iterable[int],
) -> pd.DataFrame:
    """Add columns ``{col}_lag{k}`` for integer month lags (k >= 1 for h>=1 forecasting)."""
    out = panel.copy()
    for k in lags:
        out[f"{col}_lag{k}"] = out[col].shift(k)
    return out


def run_full_preprocess(
    data_dir: Path,
    out_dir: Path,
    *,
    min_days: int = 3,
    calibration_trailing_days: int = 14,
) -> None:
    """Write panel + optional trailing stats + lagged Chl for forecasting experiments."""
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = wide_monthly_panel(data_dir, min_days_with_data_per_month=min_days)
    clim, anom = monthly_climatology_anomaly(panel[TARGET_COL])
    panel["chl_climatology"] = clim
    panel["chl_monthly_anomaly"] = anom

    clean_ts = load_trimmed_chl_frames(data_dir)
    trail = trailing_window_highfreq_stats(clean_ts, days=calibration_trailing_days)
    merged = panel.join(trail, how="left")

    # Lags for horizon >= 1 month (do not use lag0 as sole Chl input for future month)
    merged = add_monthly_lags(merged, TARGET_COL, lags=(1, 2, 3, 6, 12))

    merged.index.name = "month_end"
    panel.index.name = "month_end"
    merged.to_csv(out_dir / "monthly_multivariate_panel.csv")
    panel.to_csv(out_dir / "monthly_multivariate_panel_no_lags.csv")


__all__ = [
    "COMMON_MET_COLS",
    "WATER_SPECIFIC_COLS",
    "wide_monthly_panel",
    "monthly_climatology_anomaly",
    "trailing_window_highfreq_stats",
    "add_monthly_lags",
    "run_full_preprocess",
]
