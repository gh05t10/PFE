"""
Time-ordered splits + per-channel z-score using **train-only** statistics.

Split rule (exclusive upper bounds):
  - train: ``DateTime < train_end``
  - val:   ``train_end <= DateTime < val_end``
  - test:  ``DateTime >= val_end``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .channel_scaler import PerChannelStandardScalers, fit_per_channel_scalers, transform_frame
from .resample_config import freq_slug
from .soft_sensor_columns import FEATURE_COLS, TARGET_COL


def assign_split(
    dt: pd.Series,
    *,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> pd.Series:
    """Return 'train' | 'val' | 'test' for each timestamp."""
    t = pd.to_datetime(dt, utc=False)
    out = pd.Series(index=dt.index, dtype="object")
    out.loc[t < train_end] = "train"
    out.loc[(t >= train_end) & (t < val_end)] = "val"
    out.loc[t >= val_end] = "test"
    return out


@dataclass(frozen=True)
class SplitNormalizeConfig:
    input_csv: Path
    out_dir: Path
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    resample_freq: str  # metadata only (echoed in manifest)


def run_split_and_normalize(cfg: SplitNormalizeConfig) -> dict[str, Any]:
    """
    Read resampled panel, split, fit scalers on train, write train/val/test CSV + JSON manifest.

    Output columns: original value columns + ``*_z`` z-scored columns; ``split`` column;
    ``n_obs_*`` passthrough if present.
    """
    df = pd.read_csv(cfg.input_csv, parse_dates=["DateTime"], low_memory=False)
    if df.empty:
        raise ValueError(f"Empty input: {cfg.input_csv}")

    df["split"] = assign_split(df["DateTime"], train_end=cfg.train_end, val_end=cfg.val_end)
    train_df = df.loc[df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("Train split is empty — check train_end / date range.")

    scalers = fit_per_channel_scalers(train_df, FEATURE_COLS, TARGET_COL)
    df_z = transform_frame(df, scalers, FEATURE_COLS, TARGET_COL)

    # Pass-through any GT weight columns (already in [0,1]) so training can weight losses.
    weight_cols = [c for c in df.columns if c.startswith("weight_chl_gt")]
    weight_resampled_cols = [c for c in df.columns if c.startswith("weight_chl_gt") and c.endswith("_resampled")]
    passthrough = sorted(set(weight_cols + weight_resampled_cols))
    for c in passthrough:
        if c in df_z.columns:
            continue
        df_z[c] = pd.to_numeric(df[c], errors="coerce")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        part = df_z.loc[df_z["split"] == sp].copy()
        part.drop(columns=["split"], errors="ignore").to_csv(cfg.out_dir / f"{sp}.csv", index=False)

    manifest = {
        "input_csv": str(cfg.input_csv.resolve()),
        "resample_freq": cfg.resample_freq,
        "resample_slug": freq_slug(cfg.resample_freq),
        "split_rule": {
            "train": f"DateTime < {cfg.train_end.isoformat()}",
            "val": f"{cfg.train_end.isoformat()} <= DateTime < {cfg.val_end.isoformat()}",
            "test": f"DateTime >= {cfg.val_end.isoformat()}",
        },
        "train_end": cfg.train_end.isoformat(),
        "val_end": cfg.val_end.isoformat(),
        "n_rows": {"train": int((df["split"] == "train").sum()), "val": int((df["split"] == "val").sum()), "test": int((df["split"] == "test").sum())},
        "feature_cols": list(FEATURE_COLS),
        "target_col": TARGET_COL,
        "z_suffix": "_z",
        "weight_cols_passthrough": passthrough,
    }
    (cfg.out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    full = scalers.to_json_dict()
    full["split"] = manifest["split_rule"]
    full["train_end"] = manifest["train_end"]
    full["val_end"] = manifest["val_end"]
    full["resample_freq"] = cfg.resample_freq
    (cfg.out_dir / "scaler_params.json").write_text(json.dumps(full, indent=2), encoding="utf-8")

    return manifest


__all__ = ["assign_split", "SplitNormalizeConfig", "run_split_and_normalize"]
