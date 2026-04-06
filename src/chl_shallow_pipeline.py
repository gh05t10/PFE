"""
Shallow chlorophyll (ChlRFUShallow_RFU) preprocessing for monthly lake chlorophyll targets.

Context (aligned with aquaculture / eutrophic-lake forecasting use cases):
  - High-frequency buoy RFU is aggregated to **calendar months** as a proxy for
    "monthly chlorophyll" when lab µg/L are unavailable or as a parallel target.
  - **Two-stage aggregation** (daily → monthly) avoids bias when sampling density
    varies within the month (more samples on some days than others).

Input data: `data/BPBuoyData_*_Preprocessed.csv` where rows flagged B7, C, or M
  already have missing Chl; retaining only finite Chl is equivalent to trimming
  those QC codes for this dataset.

This module does not depend on `data_visualization.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

EXCLUDE_FLAGS = frozenset({"B7", "C", "M"})
TARGET_COL = "ChlRFUShallow_RFU"
FLAG_COL = "ChlRFUShallow_RFU_Flag"


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path
    out_dir: Path
    monthly_method: Literal["two_stage", "direct_monthly_mean"]
    min_samples_per_month: int
    min_days_with_data_per_month: int


def _normalize_flag(val: object) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s or None


def load_trimmed_chl_frames(data_dir: Path) -> pd.DataFrame:
    """Load all preprocessed yearly CSVs and keep Chl rows usable as ground truth."""
    frames: list[pd.DataFrame] = []
    paths = sorted(data_dir.glob("BPBuoyData_*_Preprocessed.csv"))
    if not paths:
        raise FileNotFoundError(f"No BPBuoyData_*_Preprocessed.csv under {data_dir}")

    usecols = ["DateTime", TARGET_COL, FLAG_COL]
    for p in paths:
        df = pd.read_csv(p, usecols=usecols, low_memory=False)
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df = df.dropna(subset=["DateTime"])
        flags = df[FLAG_COL].map(_normalize_flag)
        bad = flags.isin(EXCLUDE_FLAGS)
        df = df.loc[~bad].copy()
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df = df.dropna(subset=[TARGET_COL])
        df["source_file"] = p.name
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("DateTime").drop_duplicates(subset=["DateTime"], keep="first")
    out = out.set_index("DateTime").sort_index()
    return out


def daily_table(ts: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day: mean RFU and sample count (noise reduction vs raw 15-min)."""
    g = ts[TARGET_COL].groupby(pd.Grouper(freq="D"))
    daily = g.agg(["mean", "count"]).rename(columns={"mean": "chl_rfu_daily_mean", "count": "n_subhourly"})
    daily = daily.dropna(subset=["chl_rfu_daily_mean"])
    return daily


def monthly_from_daily(daily: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Monthly statistics from daily table (recommended default)."""
    m = daily["chl_rfu_daily_mean"].groupby(pd.Grouper(freq="ME"))
    agg = m.agg(["mean", "median", "std", "count"]).rename(
        columns={
            "mean": "chl_rfu_monthly_mean",
            "median": "chl_rfu_monthly_median_daily",
            "std": "chl_rfu_monthly_std_of_daily_means",
            "count": "n_days_in_month",
        }
    )
    agg = agg.dropna(subset=["chl_rfu_monthly_mean"])
    agg = agg[agg["n_days_in_month"] >= cfg.min_days_with_data_per_month]
    agg["year"] = agg.index.year
    agg["month"] = agg.index.month
    agg["year_month"] = agg.index.strftime("%Y-%m")
    return agg


def monthly_direct(ts: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Single-stage monthly mean of all sub-hourly points (can overweight dense periods)."""
    g = ts[TARGET_COL].groupby(pd.Grouper(freq="ME"))
    agg = g.agg(["mean", "median", "std", "count"]).rename(
        columns={
            "mean": "chl_rfu_monthly_mean",
            "median": "chl_rfu_monthly_median",
            "std": "chl_rfu_monthly_std",
            "count": "n_samples_in_month",
        }
    )
    agg = agg.dropna(subset=["chl_rfu_monthly_mean"])
    agg = agg[agg["n_samples_in_month"] >= cfg.min_samples_per_month]
    agg["year"] = agg.index.year
    agg["month"] = agg.index.month
    agg["year_month"] = agg.index.strftime("%Y-%m")
    return agg


def run_pipeline(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      clean_ts — trimmed high-frequency series (index: DateTime)
      daily — daily aggregates
      monthly — monthly ground-truth table for modelling
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    clean_ts = load_trimmed_chl_frames(cfg.data_dir)
    daily = daily_table(clean_ts)

    if cfg.monthly_method == "two_stage":
        monthly = monthly_from_daily(daily, cfg)
    elif cfg.monthly_method == "direct_monthly_mean":
        monthly = monthly_direct(clean_ts, cfg)
    else:
        raise ValueError(cfg.monthly_method)

    # Persist
    ts_path = cfg.out_dir / "chl_shallow_clean_timeseries.csv"
    clean_ts.reset_index().to_csv(ts_path, index=False)

    daily_path = cfg.out_dir / "chl_shallow_daily.csv"
    daily_out = daily.reset_index()
    daily_out.rename(columns={daily_out.columns[0]: "date"}, inplace=True)
    daily_out.to_csv(daily_path, index=False)

    monthly_path = cfg.out_dir / "chl_shallow_monthly_ground_truth.csv"
    monthly.to_csv(monthly_path)

    return clean_ts, daily, monthly


def summarize_quality(clean_ts: pd.DataFrame, monthly: pd.DataFrame) -> str:
    lines = [
        f"Trimmed rows (finite Chl, flags not in {sorted(EXCLUDE_FLAGS)}): {len(clean_ts):,}",
        f"Datetime span: {clean_ts.index.min()} → {clean_ts.index.max()}",
        f"Monthly rows (after thresholds): {len(monthly)}",
    ]
    return "\n".join(lines)
