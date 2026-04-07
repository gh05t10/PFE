"""
Outlier handling for shallow Chl ground truth on high-frequency series.

Design notes (see ``literature/outlier_handling_shallow_chl.md``):
  - **Hampel filter**: robust local spike detection (median/MAD), aligned with
    continuity-based reasoning in aquaculture/WQ TS literature.
  - **Month-stratified Tukey**: Q1/Q3 computed **per calendar month (1–12)** across
    years on **daily median** series to reduce false positives during bloom seasons.
  - **Global Tukey** (optional): matches the reporting style in *Scientific Data*
    style papers (inner 1.5×IQR, outer 3×IQR) for benchmarking only — not ideal alone.

Outputs are **labels + optional winsorization**; we avoid silently deleting samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .chl_shallow_pipeline import TARGET_COL


def _rolling_median_mad(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized rolling median; rolling MAD around rolling median (numpy only)."""
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=max(3, window // 4)).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=max(3, window // 4)).median()
    return med.to_numpy(), mad.to_numpy()


def hampel_flags(
    values: np.ndarray,
    *,
    window: int = 97,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """
    Hampel-style spike flags: |x - med| > n_sigma * 1.4826 * MAD (per window).

    *window*: in samples (~97 ≈ 24h at 15-min spacing).
    """
    x = np.asarray(values, dtype=float)
    med, mad = _rolling_median_mad(x, window)
    sigma = 1.4826 * np.where(np.isfinite(mad) & (mad > 1e-12), mad, np.nan)
    resid = np.abs(x - med)
    with np.errstate(invalid="ignore"):
        bad = resid > (n_sigma * sigma)
    return np.where(np.isfinite(bad), bad, False)


def tukey_fences(y: np.ndarray, *, k: float) -> tuple[float, float]:
    q1, q3 = np.nanpercentile(y, [25, 75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return float(np.nanmin(y)), float(np.nanmax(y))
    return q1 - k * iqr, q3 + k * iqr


def daily_median_series(ts: pd.DataFrame, col: str = TARGET_COL) -> pd.Series:
    s = ts[col].astype(float)
    g = s.groupby(ts.index.normalize())
    return g.median()


def month_stratified_tukey_on_daily(
    ts: pd.DataFrame,
    *,
    inner_k: float = 1.5,
    outer_k: float = 3.0,
    col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    For each calendar month-of-year, flag daily medians outside Tukey fences.
    Returns a dataframe indexed by **date** with columns:
    ``tukey_daily_potential``, ``tukey_daily_possible`` (bool).
    """
    d = daily_median_series(ts, col)
    pot = pd.Series(False, index=d.index)
    pos = pd.Series(False, index=d.index)
    for m in range(1, 13):
        sub = d[d.index.month == m]
        vals = sub.values
        vals = vals[np.isfinite(vals)]
        if vals.size < 10:
            continue
        lo_i, hi_i = tukey_fences(vals, k=inner_k)
        lo_o, hi_o = tukey_fences(vals, k=outer_k)
        pot.loc[sub.index] = (sub.values < lo_i) | (sub.values > hi_i)
        pos.loc[sub.index] = (sub.values < lo_o) | (sub.values > hi_o)
    return pd.DataFrame({"daily_median_chl": d, "tukey_daily_potential": pot, "tukey_daily_possible": pos})


def map_daily_flag_to_samples(ts: pd.DataFrame, daily_flags: pd.DataFrame, col: str = TARGET_COL) -> pd.Series:
    """Broadcast daily boolean to each high-frequency timestamp by calendar date."""
    br = daily_flags["tukey_daily_potential"].copy()
    br.index = pd.to_datetime(br.index).normalize()
    mapped = br.reindex(ts.index.normalize()).fillna(False).values
    return pd.Series(mapped, index=ts.index, dtype=bool)


def global_tukey_on_values(values: np.ndarray, *, inner_k: float = 1.5, outer_k: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    y = values[np.isfinite(values)]
    if y.size == 0:
        z = np.zeros_like(values, dtype=bool)
        return z, z
    lo_i, hi_i = tukey_fences(y, k=inner_k)
    lo_o, hi_o = tukey_fences(y, k=outer_k)
    v = np.asarray(values, dtype=float)
    pot = (v < lo_i) | (v > hi_i)
    pos = (v < lo_o) | (v > hi_o)
    pot &= np.isfinite(v)
    pos &= np.isfinite(v)
    return pot, pos


@dataclass(frozen=True)
class GtOutlierExportConfig:
    input_csv: Path
    out_dir: Path
    hampel_window: int
    hampel_n_sigma: float
    winsor_inner: bool


def run_gt_outlier_export(cfg: GtOutlierExportConfig) -> None:
    df = pd.read_csv(cfg.input_csv, parse_dates=["DateTime"], low_memory=False)
    df = df.sort_values("DateTime").drop_duplicates(subset=["DateTime"])
    df = df.set_index("DateTime")

    y = df[TARGET_COL].astype(float).values
    h_flag = hampel_flags(y, window=cfg.hampel_window, n_sigma=cfg.hampel_n_sigma)

    g_pot, g_pos = global_tukey_on_values(y)

    daily_t = month_stratified_tukey_on_daily(df)
    m_flag = map_daily_flag_to_samples(df, daily_t)

    # Conservative combined spike: local Hampel AND (optional) month-daily Tukey
    combined = h_flag & m_flag

    out = df[[c for c in df.columns if c != TARGET_COL]].copy()
    out[TARGET_COL] = df[TARGET_COL]
    out["flag_hampel_spike"] = h_flag
    out["flag_tukey_global_potential"] = g_pot
    out["flag_tukey_global_possible"] = g_pos
    out["flag_tukey_monthly_daily_potential"] = m_flag
    out["flag_combined_conservative"] = combined

    w = df[TARGET_COL].astype(float).copy()
    if cfg.winsor_inner and np.isfinite(y).any():
        lo_i, hi_i = tukey_fences(y[np.isfinite(y)], k=1.5)
        wvals = w.values
        wvals = np.where(np.isfinite(wvals) & (wvals < lo_i), lo_i, wvals)
        wvals = np.where(np.isfinite(wvals) & (wvals > hi_i), hi_i, wvals)
        w = pd.Series(wvals, index=w.index)
    out[f"{TARGET_COL}_winsor_global_inner"] = w

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(cfg.out_dir / "chl_gt_outlier_flags.csv", index=False)
    daily_t.to_csv(cfg.out_dir / "chl_daily_median_tukey_by_month.csv")

    summary = [
        f"input: {cfg.input_csv}",
        f"samples: {len(out):,}",
        f"Hampel window={cfg.hampel_window}, n_sigma={cfg.hampel_n_sigma}",
        f"count flag_hampel_spike: {int(h_flag.sum())}",
        f"count flag_tukey_global_potential: {int(g_pot.sum())}",
        f"count flag_tukey_monthly_daily_potential: {int(m_flag.sum())}",
        f"count combined (Hampel & monthly-daily Tukey): {int(combined.sum())}",
    ]
    (cfg.out_dir / "chl_gt_outlier_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


__all__ = [
    "hampel_flags",
    "month_stratified_tukey_on_daily",
    "run_gt_outlier_export",
    "GtOutlierExportConfig",
]
