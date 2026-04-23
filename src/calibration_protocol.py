from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _flatten_valid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    m = mask.astype(bool) & np.isfinite(y_true) & np.isfinite(y_pred)
    if weights is not None:
        w = np.where(np.isfinite(weights), weights, 0.0)
        m &= w > 0
        return y_true[m], y_pred[m], w[m]
    return y_true[m], y_pred[m], None


def weighted_bias(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None) -> float:
    """Return weighted mean (y_true - y_pred)."""
    e = y_true - y_pred
    if w is None:
        return float(np.mean(e)) if e.size else 0.0
    den = float(np.sum(w))
    if den <= 1e-12:
        return 0.0
    return float(np.sum(e * w) / den)


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None) -> dict[str, float]:
    e = y_pred - y_true
    if e.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0}
    if w is None:
        return {
            "rmse": float(np.sqrt(np.mean(e**2))),
            "mae": float(np.mean(np.abs(e))),
            "n": int(e.size),
        }
    den = float(np.sum(w))
    if den <= 1e-12:
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0}
    mse = float(np.sum((e**2) * w) / den)
    mae = float(np.sum(np.abs(e) * w) / den)
    return {"rmse": float(np.sqrt(mse)), "mae": mae, "n": int(e.size)}


@dataclass
class CalibrationConfig:
    pred_len: int = 672  # 14 days @ 30min
    calib_len: int = 48  # 1 day @ 30min
    update: Literal["bias_mean"] = "bias_mean"
    apply_space: Literal["z"] = "z"  # additive bias in z-space


@dataclass
class CalibrationReport:
    cfg: CalibrationConfig
    baseline: dict[str, float]
    calibrated: dict[str, float]
    per_block: list[dict[str, float]]


def simulate_daily_calibration(
    *,
    y_true: np.ndarray,  # (N, H)
    y_pred: np.ndarray,  # (N, H)
    mask: np.ndarray,  # (N, H)
    weights: np.ndarray | None,  # (N, H)
    cfg: CalibrationConfig,
) -> CalibrationReport:
    """
    Simulate: for each sample window, walk forward by calib_len blocks:
      - evaluate current predictions with current bias
      - then update bias using GT over that block (if available)
      - apply updated bias to subsequent blocks

    This matches slide-8 intuition: predict continuously, hard-calibrate every period.
    """
    H = int(y_true.shape[1])
    if H != cfg.pred_len:
        raise ValueError(f"pred_len mismatch: cfg.pred_len={cfg.pred_len}, but y_true has H={H}")
    if cfg.calib_len < 1 or cfg.calib_len > H:
        raise ValueError(cfg.calib_len)
    n_blocks = int(np.ceil(H / cfg.calib_len))

    # Baseline (no calibration)
    yt0, yp0, w0 = _flatten_valid(y_true, y_pred, mask, weights)
    baseline = rmse_mae(yt0, yp0, w0)

    # Calibrated
    y_pred_cal = np.array(y_pred, copy=True)
    per_block: list[dict[str, float]] = []

    for i in range(y_true.shape[0]):
        bias = 0.0
        for b in range(n_blocks):
            s = b * cfg.calib_len
            e = min(H, (b + 1) * cfg.calib_len)

            # apply current bias to this block
            y_pred_cal[i, s:e] = y_pred[i, s:e] + bias

            # compute update based on this block GT
            m = mask[i, s:e].astype(bool)
            yt = y_true[i, s:e]
            yp = y_pred_cal[i, s:e]
            ww = None
            if weights is not None:
                ww = weights[i, s:e]
            yt_f, yp_f, w_f = _flatten_valid(yt, yp, m, ww)
            blk = rmse_mae(yt_f, yp_f, w_f)
            blk["block"] = int(b)
            blk["start"] = int(s)
            blk["end"] = int(e)
            per_block.append(blk)

            if cfg.update == "bias_mean":
                bias += weighted_bias(yt_f, yp_f, w_f)
            else:
                raise ValueError(cfg.update)

    yt1, yp1, w1 = _flatten_valid(y_true, y_pred_cal, mask, weights)
    calibrated = rmse_mae(yt1, yp1, w1)

    return CalibrationReport(cfg=cfg, baseline=baseline, calibrated=calibrated, per_block=per_block)


__all__ = [
    "CalibrationConfig",
    "CalibrationReport",
    "simulate_daily_calibration",
]

