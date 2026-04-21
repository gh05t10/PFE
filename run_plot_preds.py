#!/usr/bin/env python3
"""
Plot saved predictions produced by `run_train_slide.py --save-preds`.

Input: NPZ with keys:
- y_true_rf, y_pred_rf
- (optional) target_time, context_end_time as datetime64

Outputs PNGs into --out-dir (default: alongside NPZ).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1)


def _maybe_sort_by_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_time: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if target_time is None or target_time.size == 0:
        return y_true, y_pred, target_time
    t = _as_1d(target_time)
    order = np.argsort(t)
    return y_true[order], y_pred[order], t[order]


def main() -> None:
    p = argparse.ArgumentParser(description="Plot y_true vs y_pred from preds_*.npz.")
    p.add_argument(
        "--preds",
        type=Path,
        required=True,
        help="Path to preds_val.npz or preds_test.npz produced by run_train_slide.py --save-preds",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: preds file directory)")
    p.add_argument("--prefix", default=None, help="Filename prefix (default: NPZ stem)")
    args = p.parse_args()

    preds_path: Path = args.preds
    if not preds_path.is_file():
        raise SystemExit(f"Missing preds file: {preds_path}")

    out_dir = args.out_dir or preds_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or preds_path.stem

    d = np.load(preds_path, allow_pickle=False)
    if "y_true_rf" not in d.files or "y_pred_rf" not in d.files:
        raise SystemExit(f"{preds_path} missing y_true_rf/y_pred_rf. Re-run training with --save-preds.")

    y_true = _as_1d(d["y_true_rf"]).astype(np.float64)
    y_pred = _as_1d(d["y_pred_rf"]).astype(np.float64)
    if y_true.size != y_pred.size:
        raise SystemExit(f"Size mismatch: y_true_rf({y_true.size}) vs y_pred_rf({y_pred.size})")

    target_time = d["target_time"] if "target_time" in d.files else None
    y_true, y_pred, t_sorted = _maybe_sort_by_time(y_true, y_pred, target_time)
    resid = y_pred - y_true

    # Import matplotlib lazily so users without it still can read errors early.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if y_true.size else float("nan")
    mae = float(np.mean(np.abs(y_pred - y_true))) if y_true.size else float("nan")

    # 1) Time series (if timestamps exist) or index plot.
    fig, ax = plt.subplots(figsize=(12, 4))
    x = t_sorted if (t_sorted is not None and t_sorted.size) else np.arange(y_true.size)
    ax.plot(x, y_true, label="y_true (RFU)", linewidth=1.0)
    ax.plot(x, y_pred, label="y_pred (RFU)", linewidth=1.0, alpha=0.9)
    ax.set_title(f"{prefix} — time series (RMSE={rmse:.4f}, MAE={mae:.4f})")
    ax.set_xlabel("target_time" if (t_sorted is not None and t_sorted.size) else "sample index")
    ax.set_ylabel("Chl (RFU)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_timeseries.png", dpi=160)
    plt.close(fig)

    # 2) Residual over time/index.
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, resid, linewidth=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"{prefix} — residual (y_pred - y_true)")
    ax.set_xlabel("target_time" if (t_sorted is not None and t_sorted.size) else "sample index")
    ax.set_ylabel("Residual (RFU)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_residual.png", dpi=160)
    plt.close(fig)

    # 3) Scatter y_true vs y_pred.
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.35, edgecolors="none")
    mn = float(np.nanmin([y_true.min(initial=np.nan), y_pred.min(initial=np.nan)]))
    mx = float(np.nanmax([y_true.max(initial=np.nan), y_pred.max(initial=np.nan)]))
    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
        ax.plot([mn, mx], [mn, mx], color="black", linewidth=1.0, label="y=x")
    ax.set_title(f"{prefix} — scatter (RMSE={rmse:.4f})")
    ax.set_xlabel("y_true (RFU)")
    ax.set_ylabel("y_pred (RFU)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_scatter.png", dpi=180)
    plt.close(fig)

    print(f"Saved plots to {out_dir} with prefix {prefix!r}")


if __name__ == "__main__":
    main()

