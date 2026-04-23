"""
Unified resampling of soft-sensor inputs (5 channels) and shallow Chl ground truth.

- **X** and **y** share the same time grid and aggregation rule; counts per bin document coverage.
- Invalid QC flags (B7/C/M) become NaN before resampling — no neighbour imputation.
- Output folder includes the frequency slug so 30min vs 10min runs stay separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .chl_shallow_pipeline import EXCLUDE_FLAGS, load_trimmed_chl_frames
from .chl_rule_a_months import audit_rule_a
from .resample_config import freq_slug, validate_freq
from .soft_sensor_columns import FEATURE_COLS, TARGET_COL, flag_columns


def filter_rows_to_rule_a_months(df: pd.DataFrame, data_dir: Path, *, p: float = 0.8) -> pd.DataFrame:
    """
    Keep only rows whose calendar month passes Rule A(p) on **shallow Chl** coverage.

    Use when soft-sensor training should align with **monthly Chl GT** eligibility
    (same months as ``chl_shallow_rule_a_timeseries.csv``). Default pipeline keeps
    all open-water timestamps so **X** remains dense when Chl months are sparse.
    """
    ts = load_trimmed_chl_frames(data_dir)
    audit = audit_rule_a(ts, p=p)
    passed = audit.loc[audit["rule_a_pass"], ["year", "month"]]
    good = set(zip(passed["year"], passed["month"]))
    if not good:
        return df.iloc[0:0].copy()
    dt = pd.to_datetime(df["DateTime"], errors="coerce")
    ok = dt.map(lambda t: (t.year, t.month) in good if pd.notna(t) else False)
    return df.loc[ok].copy()


def _normalize_flag(val: object) -> str | None:
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s or None


def _mask_valid_for_column(df: pd.DataFrame, col: str) -> pd.Series:
    fc = f"{col}_Flag"
    if fc in df.columns:
        bad = df[fc].map(_normalize_flag).isin(EXCLUDE_FLAGS)
    else:
        bad = False
    y = pd.to_numeric(df[col], errors="coerce")
    return (~bad) & y.notna()


def load_raw_soft_sensor_long(data_dir: Path) -> pd.DataFrame:
    """Load all yearly preprocessed CSVs; columns = DateTime + features + target + flags."""
    paths = sorted(data_dir.glob("BPBuoyData_*_Preprocessed.csv"))
    if not paths:
        raise FileNotFoundError(f"No BPBuoyData_*_Preprocessed.csv under {data_dir}")

    need = ["DateTime"] + list(FEATURE_COLS) + [TARGET_COL]
    flag_need = flag_columns()
    frames: list[pd.DataFrame] = []
    for p in paths:
        header = pd.read_csv(p, nrows=0).columns.tolist()
        usecols = [c for c in need + flag_need if c in header]
        df = pd.read_csv(p, usecols=usecols, low_memory=False)
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df = df.dropna(subset=["DateTime"])
        df["source_file"] = p.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("DateTime").drop_duplicates(subset=["DateTime"], keep="first")
    return out


def _apply_valid_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column with NaN where QC excludes the row."""
    s = pd.to_numeric(df[col], errors="coerce")
    ok = _mask_valid_for_column(df, col)
    return s.where(ok)


@dataclass(frozen=True)
class UnifiedResampleConfig:
    data_dir: Path
    out_dir: Path
    freq: str
    agg: Literal["median", "mean"] = "median"
    rule_a: bool = False
    rule_a_p: float = 0.8
    gt_weights_csv: Path | None = None


def resample_panel(df: pd.DataFrame, *, freq: str, agg: Literal["median", "mean"] = "median") -> pd.DataFrame:
    """
    Resample FEATURE_COLS + TARGET_COL to a regular grid.

    Returns a DataFrame with DatetimeIndex (bin **right** labels, **right**-closed intervals).
    """
    validate_freq(freq)
    d = df.sort_values("DateTime").drop_duplicates(subset=["DateTime"], keep="first")
    work = pd.DataFrame(index=pd.DatetimeIndex(d["DateTime"]))
    for col in list(FEATURE_COLS) + [TARGET_COL]:
        if col not in d.columns:
            raise KeyError(f"Missing column {col!r} in input frame.")
        work[col] = _apply_valid_numeric(d, col).values
    work = work.sort_index()
    work = work[~work.index.duplicated(keep="first")]

    rule = agg
    values = work.resample(freq, label="right", closed="right").agg(rule)
    counts = work.resample(freq, label="right", closed="right").count()
    counts = counts.rename(columns={c: f"n_obs_{c}" for c in counts.columns})
    out = pd.concat([values, counts], axis=1)
    out.index.name = "DateTime"
    return out


def _load_gt_weights(path: Path) -> pd.DataFrame:
    """
    Load shallow Chl GT weights table (e.g. ``chl_shallow_transformer_gt.csv``).
    Expected columns: DateTime, weight_chl_gt (and optionally weight_chl_gt_conservative).
    """
    g = pd.read_csv(path, parse_dates=["DateTime"], low_memory=False)
    g = g.sort_values("DateTime").drop_duplicates(subset=["DateTime"], keep="first")
    cols = [c for c in g.columns if c.startswith("weight_chl_gt")]
    if not cols:
        raise ValueError(f"No weight_chl_gt* columns in {path}")
    g = g[["DateTime", *cols]].copy()
    for c in cols:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    return g


def resample_gt_weights(gt: pd.DataFrame, *, freq: str, agg: Literal["median", "mean"] = "mean") -> pd.DataFrame:
    """
    Resample weights onto the same grid as values.

    We aggregate weights by **mean** within bin (i.e. expected reliability mass in that bin).
    This preserves partial coverage in a bin when some raw points are flagged as spikes.
    """
    validate_freq(freq)
    d = gt.sort_values("DateTime").drop_duplicates(subset=["DateTime"], keep="first")
    work = pd.DataFrame(index=pd.DatetimeIndex(d["DateTime"]))
    wcols = [c for c in d.columns if c.startswith("weight_chl_gt")]
    for c in wcols:
        work[c] = pd.to_numeric(d[c], errors="coerce").values
    work = work.sort_index()
    w = work.resample(freq, label="right", closed="right").agg(agg)
    w = w.rename(columns={c: f"{c}_resampled" for c in w.columns})
    w.index.name = "DateTime"
    return w


def run_unified_resample(cfg: UnifiedResampleConfig) -> Path:
    """Write ``soft_sensor_resampled.csv`` + ``resample_meta.txt`` under ``out_dir``."""
    freq = validate_freq(cfg.freq)
    slug = freq_slug(freq)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_soft_sensor_long(cfg.data_dir)
    if cfg.rule_a:
        raw = filter_rows_to_rule_a_months(raw, cfg.data_dir, p=cfg.rule_a_p)
    panel = resample_panel(raw, freq=freq, agg=cfg.agg)
    if cfg.gt_weights_csv is not None and Path(cfg.gt_weights_csv).is_file():
        gt = _load_gt_weights(Path(cfg.gt_weights_csv))
        w = resample_gt_weights(gt, freq=freq, agg="mean")
        panel = panel.join(w, how="left")

    csv_path = out_dir / "soft_sensor_resampled.csv"
    panel.reset_index().to_csv(csv_path, index=False)

    meta = out_dir / "resample_meta.txt"
    lines = [
        f"resample_freq={freq}",
        f"freq_slug={slug}",
        f"aggregation={cfg.agg}",
        f"rule_a_filter={cfg.rule_a}",
        f"rule_a_p={cfg.rule_a_p}",
        f"gt_weights_csv={str(cfg.gt_weights_csv) if cfg.gt_weights_csv is not None else ''}",
        f"feature_cols={list(FEATURE_COLS)}",
        f"target_col={TARGET_COL}",
        f"n_rows={len(panel)}",
        f"datetime_min={panel.index.min()}",
        f"datetime_max={panel.index.max()}",
    ]
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return csv_path


__all__ = [
    "UnifiedResampleConfig",
    "filter_rows_to_rule_a_months",
    "load_raw_soft_sensor_long",
    "resample_panel",
    "run_unified_resample",
]
