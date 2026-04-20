"""
Naive baselines on the same windowed samples as GRU (same NPZ, same scaler).

- **mean_train**: predict constant = mean(y_z on train) for every sample.
- **persistence**: predict y_hat_z = chl_z_at_window_end (last known Chl in window before target).

Metrics in Chl RFU via inverse transform from scaler_params.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_scaler_target(path: Path) -> tuple[float, float]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    mu = float(d["target"]["mean"])
    std = float(d["target"]["std"])
    return mu, std


def z_to_rf(z: np.ndarray, mu: float, std: float) -> np.ndarray:
    return z * std + mu


def rmse_mae_rf(y_true_rf: np.ndarray, y_pred_rf: np.ndarray) -> dict[str, float]:
    e = y_true_rf - y_pred_rf
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "n": int(len(e)),
    }


def eval_mean_train_baseline(
    train_npz: Path,
    val_or_test_npz: Path,
    scaler_json: Path,
) -> dict[str, Any]:
    tr = np.load(train_npz, allow_pickle=False)
    va = np.load(val_or_test_npz, allow_pickle=False)
    y_tr = tr["y_z"].astype(np.float64)
    m_tr = tr["y_mask"].astype(bool) if "y_mask" in tr.files else np.ones(len(y_tr), dtype=bool)
    finite_tr = m_tr & np.isfinite(y_tr)
    mean_z = float(np.nanmean(y_tr[finite_tr]))

    y = va["y_z"].astype(np.float64)
    m = va["y_mask"].astype(bool) if "y_mask" in va.files else np.ones(len(y), dtype=bool)
    valid = m & np.isfinite(y)
    pred_z = np.full_like(y, mean_z)
    mu, std = load_scaler_target(scaler_json)
    out = rmse_mae_rf(z_to_rf(y[valid], mu, std), z_to_rf(pred_z[valid], mu, std))
    out["method"] = "mean_train_z"
    out["mean_z_used"] = mean_z
    return out


def eval_persistence_baseline(
    val_or_test_npz: Path,
    scaler_json: Path,
) -> dict[str, Any]:
    va = np.load(val_or_test_npz, allow_pickle=False)
    if "chl_z_at_window_end" not in va.files:
        return {
            "method": "persistence_window_end",
            "error": "NPZ missing chl_z_at_window_end — re-run run_build_window_dataset.py",
        }
    y = va["y_z"].astype(np.float64)
    pred_z = va["chl_z_at_window_end"].astype(np.float64)
    m = va["y_mask"].astype(bool) if "y_mask" in va.files else np.ones(len(y), dtype=bool)
    valid = m & np.isfinite(y) & np.isfinite(pred_z)
    if valid.sum() == 0:
        return {"method": "persistence_window_end", "error": "no valid samples"}
    mu, std = load_scaler_target(scaler_json)
    out = rmse_mae_rf(z_to_rf(y[valid], mu, std), z_to_rf(pred_z[valid], mu, std))
    out["method"] = "persistence_window_end"
    return out


def run_all_baselines(
    train_npz: Path,
    val_npz: Path,
    test_npz: Path | None,
    scaler_json: Path,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scaler_json": str(scaler_json.resolve()),
        "val": {
            "mean_train": eval_mean_train_baseline(train_npz, val_npz, scaler_json),
            "persistence": eval_persistence_baseline(val_npz, scaler_json),
        },
    }
    if test_npz is not None and Path(test_npz).is_file():
        out["test"] = {
            "mean_train": eval_mean_train_baseline(train_npz, test_npz, scaler_json),
            "persistence": eval_persistence_baseline(test_npz, scaler_json),
        }
    return out


__all__ = [
    "eval_mean_train_baseline",
    "eval_persistence_baseline",
    "run_all_baselines",
    "rmse_mae_rf",
]
