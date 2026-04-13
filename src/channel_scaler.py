"""
Per-channel z-score (train-only fit). Separate statistics for features vs target.

Use ``fit_standard`` on training columns (finite values only), then ``transform_column`` /
``inverse_transform_column`` for each series. Persist with ``to_json`` / ``from_json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-8


@dataclass
class ChannelStats:
    mean: float
    std: float

    def transform(self, x: pd.Series | np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=np.float64)
        out = (a - self.mean) / self.std
        return out

    def inverse_transform(self, z: pd.Series | np.ndarray) -> np.ndarray:
        a = np.asarray(z, dtype=np.float64)
        return a * self.std + self.mean


@dataclass
class PerChannelStandardScalers:
    """One scaler per feature column + one for the target."""

    features: dict[str, ChannelStats] = field(default_factory=dict)
    target: ChannelStats | None = None
    target_name: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "method": "standard_per_channel",
            "features": {k: {"mean": v.mean, "std": v.std} for k, v in self.features.items()},
            "target": (
                {"name": self.target_name, "mean": self.target.mean, "std": self.target.std}
                if self.target is not None
                else None
            ),
        }

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> PerChannelStandardScalers:
        if d.get("method") != "standard_per_channel":
            raise ValueError(f"Unknown scaler method {d.get('method')!r}")
        feats = {
            k: ChannelStats(mean=float(v["mean"]), std=float(v["std"]))
            for k, v in d["features"].items()
        }
        tgt = None
        name = ""
        if d.get("target"):
            tgt = ChannelStats(mean=float(d["target"]["mean"]), std=float(d["target"]["std"]))
            name = str(d["target"]["name"])
        return cls(features=feats, target=tgt, target_name=name)

    @classmethod
    def load_json(cls, path: Path) -> PerChannelStandardScalers:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json_dict(d)


def fit_channel_stats(s: pd.Series) -> ChannelStats:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        raise ValueError("Cannot fit scaler: no finite values in series.")
    mu = float(x.mean())
    sig = float(x.std(ddof=0))
    if not np.isfinite(sig) or sig < EPS:
        sig = 1.0
    return ChannelStats(mean=mu, std=sig)


def fit_per_channel_scalers(
    train_df: pd.DataFrame,
    feature_cols: tuple[str, ...],
    target_col: str,
) -> PerChannelStandardScalers:
    """Fit μ, σ on **train** rows only (finite values per column)."""
    feats: dict[str, ChannelStats] = {}
    for c in feature_cols:
        if c not in train_df.columns:
            raise KeyError(f"Missing feature column {c!r}")
        feats[c] = fit_channel_stats(train_df[c])
    if target_col not in train_df.columns:
        raise KeyError(f"Missing target column {target_col!r}")
    tgt = fit_channel_stats(train_df[target_col])
    return PerChannelStandardScalers(features=feats, target=tgt, target_name=target_col)


def transform_frame(
    df: pd.DataFrame,
    scalers: PerChannelStandardScalers,
    feature_cols: tuple[str, ...],
    target_col: str,
    *,
    suffix_z: str = "_z",
) -> pd.DataFrame:
    """
    Add z-scored columns: ``{feat}{suffix_z}``, ``{target}{suffix_z}``; keep originals.

    NaN stays NaN through transform (no imputation).
    """
    out = df.copy()
    for c in feature_cols:
        st = scalers.features[c]
        raw = pd.to_numeric(out[c], errors="coerce")
        out[f"{c}{suffix_z}"] = np.where(raw.notna(), st.transform(raw), np.nan)
    if scalers.target is None:
        raise ValueError("scalers.target is missing")
    raw_y = pd.to_numeric(out[target_col], errors="coerce")
    out[f"{target_col}{suffix_z}"] = np.where(raw_y.notna(), scalers.target.transform(raw_y), np.nan)
    return out


def inverse_transform_target(z: np.ndarray | pd.Series, scalers: PerChannelStandardScalers) -> np.ndarray:
    if scalers.target is None:
        raise ValueError("No target scaler")
    return scalers.target.inverse_transform(z)


__all__ = [
    "EPS",
    "ChannelStats",
    "PerChannelStandardScalers",
    "fit_channel_stats",
    "fit_per_channel_scalers",
    "transform_frame",
    "inverse_transform_target",
]
