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

from .chl_shallow_pipeline import EXCLUDE_FLAGS
from .resample_config import freq_slug, validate_freq
from .soft_sensor_columns import FEATURE_COLS, TARGET_COL, flag_columns


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


def run_unified_resample(cfg: UnifiedResampleConfig) -> Path:
    """Write ``soft_sensor_resampled.csv`` + ``resample_meta.txt`` under ``out_dir``."""
    freq = validate_freq(cfg.freq)
    slug = freq_slug(freq)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_soft_sensor_long(cfg.data_dir)
    panel = resample_panel(raw, freq=freq, agg=cfg.agg)

    csv_path = out_dir / "soft_sensor_resampled.csv"
    panel.reset_index().to_csv(csv_path, index=False)

    meta = out_dir / "resample_meta.txt"
    lines = [
        f"resample_freq={freq}",
        f"freq_slug={slug}",
        f"aggregation={cfg.agg}",
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
    "load_raw_soft_sensor_long",
    "resample_panel",
    "run_unified_resample",
]
