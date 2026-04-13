"""
Multi-horizon supervision (≤ 31 days) from shallow Chl series.

Why daily resolution?
  High-frequency sampling differs by year (e.g. 10 vs 15 minutes) and can be irregular.
  Aggregating to **one value per calendar day** gives comparable targets across years before
  defining horizons in **whole days**.

Each row is one (anchor_date, horizon_days) pair with a supervised target on target_date.
Model input windows are **not** built here — only the **label side** for multi-horizon training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .chl_shallow_pipeline import TARGET_COL

# Horizons (days). Max 31 caps “within one calendar month” span in day units.
DEFAULT_HORIZON_DAYS: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 14, 21, 28, 30)


@dataclass(frozen=True)
class HorizonTableConfig:
    input_csv: Path
    out_csv: Path
    horizon_days: tuple[int, ...] = DEFAULT_HORIZON_DAYS
    max_horizon_days: int = 31
    weight_col: str | None = "weight_chl_gt"


def _load_series(cfg: HorizonTableConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_csv, parse_dates=["DateTime"], low_memory=False)
    df = df.sort_values("DateTime").drop_duplicates(subset=["DateTime"])
    if cfg.weight_col and cfg.weight_col in df.columns:
        w = df[cfg.weight_col].astype(float).fillna(0.0)
    else:
        w = pd.Series(1.0, index=df.index)
    df["_w"] = w
    df = df.loc[df["_w"] > 0].copy()
    return df


def daily_chl_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day: median/mean Chl and sample counts (after weight filter)."""
    df = df.copy()
    df["date"] = df["DateTime"].dt.normalize()
    g = df.groupby("date", sort=True)
    med = g[TARGET_COL].median()
    mean = g[TARGET_COL].mean()
    n = g[TARGET_COL].count()
    out = pd.DataFrame({"chl_daily_median": med, "chl_daily_mean": mean, "n_samples": n})
    return out


def build_horizon_supervision(
    daily: pd.DataFrame,
    *,
    horizon_days: Iterable[int],
    max_horizon_days: int = 31,
    target_col: str = "chl_daily_median",
) -> pd.DataFrame:
    """
    For each day *anchor* with finite *target_col*, and each *h* in horizon_days (1..max),
    add a row if *anchor + h days* exists and has finite target_col.
    """
    s = daily[target_col].copy()
    s = s[s.notna()]
    rows: list[dict] = []
    for anchor in s.index:
        y0 = s.loc[anchor]
        if not np.isfinite(y0):
            continue
        for h in horizon_days:
            if h < 1 or h > max_horizon_days:
                continue
            tgt_date = anchor + pd.Timedelta(days=int(h))
            if tgt_date not in s.index:
                continue
            y1 = s.loc[tgt_date]
            if not np.isfinite(y1):
                continue
            rows.append(
                {
                    "anchor_date": anchor.date().isoformat(),
                    "horizon_days": int(h),
                    "target_date": tgt_date.date().isoformat(),
                    "y_chl": float(y1),
                    "y_anchor_chl": float(y0),
                    "target_col_used": target_col,
                }
            )
    return pd.DataFrame(rows)


def run_horizon_table_export(cfg: HorizonTableConfig) -> pd.DataFrame:
    df = _load_series(cfg)
    daily = daily_chl_table(df)
    horizons = tuple(h for h in cfg.horizon_days if 1 <= h <= cfg.max_horizon_days)
    sup = build_horizon_supervision(
        daily,
        horizon_days=horizons,
        max_horizon_days=cfg.max_horizon_days,
        target_col="chl_daily_median",
    )
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    daily.reset_index().to_csv(cfg.out_csv.parent / "chl_daily_for_horizons.csv", index=False)
    sup.to_csv(cfg.out_csv, index=False)
    summary = [
        f"input: {cfg.input_csv}",
        f"daily rows: {len(daily)}",
        f"supervision rows: {len(sup)}",
        f"horizons (days): {horizons}",
    ]
    (cfg.out_csv.parent / "horizon_supervision_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return sup


__all__ = [
    "DEFAULT_HORIZON_DAYS",
    "HorizonTableConfig",
    "daily_chl_table",
    "build_horizon_supervision",
    "run_horizon_table_export",
]
