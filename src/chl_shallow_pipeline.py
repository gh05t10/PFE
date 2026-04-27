"""
Utilities for shallow chlorophyll (`ChlRFUShallow_RFU`) ground truth loading.

Used by ``unified_resample`` (QC flags → NaN), ``chl_rule_a_months`` (month eligibility;
no monthly forecasting series here), and GT outlier scripts.

Rows flagged B7, C, or M are excluded upstream of modelling/resampling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXCLUDE_FLAGS = frozenset({"B7", "C", "M"})
TARGET_COL = "ChlRFUShallow_RFU"
FLAG_COL = "ChlRFUShallow_RFU_Flag"


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
