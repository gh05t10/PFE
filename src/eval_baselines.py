"""
Naive baselines on the same windowed samples as GRU (same NPZ, same scaler).

- **mean_train**: predict constant = mean(y_z on train) for every sample.
- **persistence**: predict y_hat_z = chl_z_at_window_end (last known Chl in window before target).

Metrics in Chl RFU via inverse transform from scaler_params.json.

Multi-step support:
- If ``Y_z`` / ``Y_mask`` exist, evaluate metrics on the flattened valid elements across horizon.
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


def _load_y_any(npz: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (y, mask).

    - Multi-step: y shape (N, H) from Y_z and Y_mask
    - Legacy:     y shape (N,) from y_z and y_mask
    """
    if "Y_z" in npz.files:
        y = npz["Y_z"].astype(np.float64)
        m = npz["Y_mask"].astype(bool) if "Y_mask" in npz.files else np.isfinite(y)
        return y, m & np.isfinite(y)
    y = npz["y_z"].astype(np.float64)
    m = npz["y_mask"].astype(bool) if "y_mask" in npz.files else np.ones(len(y), dtype=bool)
    return y, m & np.isfinite(y)


def _flatten_valid(y: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Works for both 1D and 2D (numpy boolean indexing flattens).
    return y[valid], pred[valid]


def eval_mean_train_baseline(
    train_npz: Path,
    val_or_test_npz: Path,
    scaler_json: Path,
) -> dict[str, Any]:
    tr = np.load(train_npz, allow_pickle=False)
    va = np.load(val_or_test_npz, allow_pickle=False)
    y_tr, m_tr = _load_y_any(tr)
    mean_z = float(np.nanmean(y_tr[m_tr]))

    y, m = _load_y_any(va)
    pred_z = np.full_like(y, mean_z)
    valid = m & np.isfinite(pred_z)
    mu, std = load_scaler_target(scaler_json)
    yt, yp = _flatten_valid(y, pred_z, valid)
    out = rmse_mae_rf(z_to_rf(yt, mu, std), z_to_rf(yp, mu, std))
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
    y, m = _load_y_any(va)
    end = va["chl_z_at_window_end"].astype(np.float64)
    if y.ndim == 1:
        pred_z = end
        valid = m & np.isfinite(y) & np.isfinite(pred_z)
    else:
        pred_z = np.repeat(end[:, None], y.shape[1], axis=1)
        valid = m & np.isfinite(y) & np.isfinite(pred_z)
    if valid.sum() == 0:
        return {"method": "persistence_window_end", "error": "no valid samples"}
    mu, std = load_scaler_target(scaler_json)
    yt, yp = _flatten_valid(y, pred_z, valid)
    out = rmse_mae_rf(z_to_rf(yt, mu, std), z_to_rf(yp, mu, std))
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
