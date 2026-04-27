#!/usr/bin/env python3
"""
Generate key visualizations for the project (forecast examples + error curves).

Outputs (PDF + PNG) under: figures/out/
  - fig_forecast_examples_test.pdf/png
  - fig_rmse_by_horizon_test.pdf/png
  - fig_block_rmse_test.pdf/png  (RMSE by **forecast day**: 1 day = 48 steps @30min)
  - daily_horizon_rmse_{split}.json — same curves as numeric table for slides
  - fig_training_curve_slide.pdf/png

Designed to be slide-friendly and reproducible.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_scaler(scaler_json: Path) -> tuple[float, float]:
    d = json.loads(scaler_json.read_text(encoding="utf-8"))
    mu = float(d["target"]["mean"])
    std = float(d["target"]["std"])
    return mu, std


def _unz(z: np.ndarray, mu: float, std: float) -> np.ndarray:
    return z * std + mu


def _json_float_list(arr: np.ndarray) -> list[float | None]:
    """JSON-safe: NaN/inf become null."""
    out: list[float | None] = []
    for x in np.asarray(arr).ravel():
        xf = float(x)
        if not np.isfinite(xf):
            out.append(None)
        else:
            out.append(xf)
    return out


def _torch_load_trusted(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _apply_bias_mean_calibration(
    *,
    y_true: np.ndarray,  # (H,)
    y_pred: np.ndarray,  # (H,)
    mask: np.ndarray,  # (H,)
    weights: np.ndarray | None,  # (H,)
    calib_len: int,
) -> np.ndarray:
    """
    Per-sample version of src.calibration_protocol.simulate_daily_calibration.
    Additive bias in z-space, updated per block with weighted mean error.
    """
    H = int(y_true.shape[0])
    y_cal = np.array(y_pred, copy=True)
    bias = 0.0
    n_blocks = int(np.ceil(H / calib_len))
    for b in range(n_blocks):
        s = b * calib_len
        e = min(H, (b + 1) * calib_len)
        y_cal[s:e] = y_pred[s:e] + bias
        m = mask[s:e].astype(bool) & np.isfinite(y_true[s:e]) & np.isfinite(y_cal[s:e])
        if weights is not None:
            w = np.where(np.isfinite(weights[s:e]), weights[s:e], 0.0)
            m &= w > 0
            den = float(np.sum(w[m]))
            if den > 1e-12:
                bias += float(np.sum((y_true[s:e][m] - y_cal[s:e][m]) * w[m]) / den)
        else:
            if m.any():
                bias += float(np.mean(y_true[s:e][m] - y_cal[s:e][m]))
    return y_cal


def _weighted_rmse_per_step(
    y_true: np.ndarray,  # (N,H)
    y_pred: np.ndarray,  # (N,H)
    mask: np.ndarray,  # (N,H)
    weights: np.ndarray | None,  # (N,H)
) -> np.ndarray:
    m = mask.astype(bool) & np.isfinite(y_true) & np.isfinite(y_pred)
    if weights is None:
        out = np.full((y_true.shape[1],), np.nan, dtype=np.float64)
        for t in range(y_true.shape[1]):
            mt = m[:, t]
            if not mt.any():
                continue
            e = y_pred[mt, t] - y_true[mt, t]
            out[t] = float(np.sqrt(np.mean(e**2)))
        return out
    w = np.where(np.isfinite(weights), weights, 0.0)
    m &= w > 0
    out = np.full((y_true.shape[1],), np.nan, dtype=np.float64)
    for t in range(y_true.shape[1]):
        mt = m[:, t]
        if not mt.any():
            continue
        wt = w[mt, t].astype(np.float64)
        den = float(np.sum(wt))
        if den <= 1e-12:
            continue
        e = (y_pred[mt, t] - y_true[mt, t]).astype(np.float64)
        out[t] = float(np.sqrt(np.sum((e**2) * wt) / den))
    return out


def _block_rmse(
    y_true: np.ndarray,  # (N,H)
    y_pred: np.ndarray,  # (N,H)
    mask: np.ndarray,  # (N,H)
    weights: np.ndarray | None,  # (N,H)
    calib_len: int,
) -> np.ndarray:
    H = int(y_true.shape[1])
    n_blocks = int(np.ceil(H / calib_len))
    rmse = np.full((n_blocks,), np.nan, dtype=np.float64)
    for b in range(n_blocks):
        s = b * calib_len
        e = min(H, (b + 1) * calib_len)
        yt = y_true[:, s:e]
        yp = y_pred[:, s:e]
        m = mask[:, s:e].astype(bool) & np.isfinite(yt) & np.isfinite(yp)
        if weights is None:
            if not m.any():
                continue
            err = yp[m] - yt[m]
            rmse[b] = float(np.sqrt(np.mean(err**2)))
        else:
            w = np.where(np.isfinite(weights[:, s:e]), weights[:, s:e], 0.0)
            m &= w > 0
            if not m.any():
                continue
            den = float(np.sum(w[m]))
            if den <= 1e-12:
                continue
            err = (yp - yt).astype(np.float64)
            rmse[b] = float(np.sqrt(np.sum((err[m] ** 2) * w[m]) / den))
    return rmse


def main() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
        }
    )
    sns.set_style("whitegrid")

    repo = Path(__file__).resolve().parents[1]
    # Allow "import src.*" when running from repo root or elsewhere.
    sys.path.insert(0, str(repo))
    window_dir = (
        repo
        / "processed/chl_shallow/resampled_30min/normalized_split/windowed_L96_H0_P672_S48"
    )
    out_dir = repo / "figures/out"
    _ensure_dir(out_dir)

    split = "test"
    npz_path = window_dir / f"{split}.npz"
    ckpt_path = window_dir / "checkpoints_slide/best_slide.pt"
    scaler_json = repo / "processed/chl_shallow/resampled_30min/normalized_split/scaler_params.json"

    mu, std = _load_scaler(scaler_json)

    # Load window arrays
    d = np.load(npz_path, allow_pickle=True)
    X5 = d["X_z"].astype(np.float32)
    X5m = d["X_mask"].astype(bool)
    X6 = d["X6_z"].astype(np.float32)
    X6m = d["X6_mask"].astype(bool)
    Y = d["Y_z"].astype(np.float32)
    Ym = d["Y_mask"].astype(bool)
    Yw = d.get("Y_w")
    if Yw is not None:
        Yw = Yw.astype(np.float32)
    tts = d.get("target_times")
    c_end = d.get("context_end_time")

    H = int(Y.shape[1])
    calib_len = 48

    # Compute persistence (z-space)
    chl_end = d["chl_z_at_window_end"].astype(np.float32)
    Y_persist = np.repeat(chl_end[:, None], H, axis=1)

    # Teacher preds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = _torch_load_trusted(ckpt_path, device=torch.device("cpu"))
    from src.models_slide import SlidePatchCrossAttn, SlidePatchCrossAttnConfig

    cfg = SlidePatchCrossAttnConfig(**ck["model_config"])
    model = SlidePatchCrossAttn(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    with torch.no_grad():
        xb5 = torch.from_numpy(X5).to(device)
        xm5 = torch.from_numpy(X5m).to(device)
        xb6 = torch.from_numpy(X6).to(device)
        xm6 = torch.from_numpy(X6m).to(device)
        pred = model(xb5, xm5, xb6, xm6).detach().cpu().numpy().astype(np.float32)
    if pred.ndim == 1:
        pred = pred[:, None]
    Y_teacher = pred

    # Calibrated per-sample
    Y_teacher_cal = np.zeros_like(Y_teacher)
    Y_persist_cal = np.zeros_like(Y_persist)
    for i in range(Y.shape[0]):
        w_i = None if Yw is None else Yw[i]
        Y_teacher_cal[i] = _apply_bias_mean_calibration(
            y_true=Y[i], y_pred=Y_teacher[i], mask=Ym[i], weights=w_i, calib_len=calib_len
        )
        Y_persist_cal[i] = _apply_bias_mean_calibration(
            y_true=Y[i], y_pred=Y_persist[i], mask=Ym[i], weights=w_i, calib_len=calib_len
        )

    # -------- Figure 1: forecast examples (original units) --------
    # pick samples with most valid GT points
    valid_counts = Ym.sum(axis=1)
    idx = np.argsort(-valid_counts)[:3].tolist()

    fig, axes = plt.subplots(nrows=len(idx), ncols=1, figsize=(6.75, 6.5), sharex=True)
    if len(idx) == 1:
        axes = [axes]
    for ax, i in zip(axes, idx, strict=True):
        tt = np.arange(H) if tts is None else np.asarray(tts[i]).astype("datetime64[ns]")
        # mask for plotting GT
        m = Ym[i].astype(bool) & np.isfinite(Y[i])
        y_true_u = _unz(Y[i], mu, std)
        y_t_u = _unz(Y_teacher[i], mu, std)
        y_tcal_u = _unz(Y_teacher_cal[i], mu, std)
        y_p_u = _unz(Y_persist[i], mu, std)
        y_pcal_u = _unz(Y_persist_cal[i], mu, std)

        x = tt
        ax.plot(x[m], y_true_u[m], color="#264653", label="GT (Chl)", zorder=4)
        ax.plot(x, y_p_u, color="#B0BEC5", label="Persistence", alpha=0.9)
        ax.plot(x, y_pcal_u, color="#2A9D8F", label="Persistence + calib", alpha=0.95)
        ax.plot(x, y_t_u, color="#E9C46A", label="Teacher", alpha=0.9)
        ax.plot(x, y_tcal_u, color="#E76F51", label="Teacher + calib", alpha=0.95)

        title = f"Test sample {i} | valid={int(valid_counts[i])}/{H}"
        if c_end is not None:
            title += f" | context_end={np.asarray(c_end[i])}"
        ax.set_title(title)
        ax.set_ylabel("Chl (RFU)")

    axes[-1].set_xlabel("Target time" if tts is not None else "Horizon step (30min)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_forecast_examples_test.pdf")
    fig.savefig(out_dir / "fig_forecast_examples_test.png", dpi=300)
    plt.close(fig)

    # -------- Figure 2: RMSE by horizon step (z-space) --------
    rmse_teacher = _weighted_rmse_per_step(Y, Y_teacher, Ym, Yw)
    rmse_teacher_cal = _weighted_rmse_per_step(Y, Y_teacher_cal, Ym, Yw)
    rmse_persist = _weighted_rmse_per_step(Y, Y_persist, Ym, Yw)
    rmse_persist_cal = _weighted_rmse_per_step(Y, Y_persist_cal, Ym, Yw)

    x_days = (np.arange(H) * 0.5) / 24.0  # 30min -> hours -> days
    fig, ax = plt.subplots(figsize=(6.75, 2.8))
    ax.plot(x_days, rmse_persist, color="#B0BEC5", label="Persistence", alpha=0.95)
    ax.plot(x_days, rmse_persist_cal, color="#2A9D8F", label="Persistence + calib", alpha=0.95)
    ax.plot(x_days, rmse_teacher, color="#E9C46A", label="Teacher", alpha=0.95)
    ax.plot(x_days, rmse_teacher_cal, color="#E76F51", label="Teacher + calib", alpha=0.95)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("RMSE (z)")
    ax.set_title("Test RMSE by horizon (weighted, masked)")
    ax.legend(ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_rmse_by_horizon_test.pdf")
    fig.savefig(out_dir / "fig_rmse_by_horizon_test.png", dpi=300)
    plt.close(fig)

    # -------- Figure 3: RMSE by calibration block (day) --------
    blk_teacher = _block_rmse(Y, Y_teacher, Ym, Yw, calib_len=calib_len)
    blk_teacher_cal = _block_rmse(Y, Y_teacher_cal, Ym, Yw, calib_len=calib_len)
    blk_persist = _block_rmse(Y, Y_persist, Ym, Yw, calib_len=calib_len)
    blk_persist_cal = _block_rmse(Y, Y_persist_cal, Ym, Yw, calib_len=calib_len)

    days = np.arange(len(blk_teacher)) + 1
    fig, ax = plt.subplots(figsize=(6.75, 2.8))
    ax.plot(days, blk_persist, marker="o", color="#B0BEC5", label="Persistence")
    ax.plot(days, blk_persist_cal, marker="o", color="#2A9D8F", label="Persistence + calib")
    ax.plot(days, blk_teacher, marker="o", color="#E9C46A", label="Teacher")
    ax.plot(days, blk_teacher_cal, marker="o", color="#E76F51", label="Teacher + calib")
    ax.set_xticks(days)
    ax.set_xlabel("Forecast day index k (each day = 48 steps @30 min)")
    ax.set_ylabel("RMSE (z)")
    ax.set_title("Test RMSE by forecast day (48 steps/day; pooled weighted RMSE)")
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_block_rmse_test.pdf")
    fig.savefig(out_dir / "fig_block_rmse_test.png", dpi=300)
    plt.close(fig)

    # -------- Table: same per-day RMSE as JSON (option "1": daily reporting, 30-min targets) --------
    daily_payload = {
        "schema": "forecast_day_k_starts_at_day_1_ahead",
        "resolution": "30min",
        "steps_per_calendar_day": calib_len,
        "pred_len_steps": H,
        "num_forecast_days": int(len(blk_teacher)),
        "note": (
            "Targets remain 30-minute Chl (z). Each forecast_day_index k aggregates errors "
            "over steps [ (k-1)*steps_per_calendar_day , min(k*steps_per_calendar_day, H) ). "
            "Aligned with calibration blocks of length calib_len."
        ),
        "rmse_z": {
            "persistence": _json_float_list(blk_persist),
            "persistence_calibrated": _json_float_list(blk_persist_cal),
            "teacher": _json_float_list(blk_teacher),
            "teacher_calibrated": _json_float_list(blk_teacher_cal),
        },
        "forecast_day_index": list(range(1, len(blk_teacher) + 1)),
    }
    (out_dir / f"daily_horizon_rmse_{split}.json").write_text(
        json.dumps(daily_payload, indent=2),
        encoding="utf-8",
    )

    # -------- Figure 4: training curve (from metrics CSV) --------
    metrics_csv = window_dir / "checkpoints_slide/metrics_slide.csv"
    if metrics_csv.exists():
        import csv

        rows = []
        with metrics_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        if rows:
            epoch = np.array([int(r["epoch"]) for r in rows])
            train_loss = np.array([float(r["train_loss_z"]) for r in rows])
            val_rmse = np.array([float(r["val_rmse_chl"]) for r in rows])

            fig, ax1 = plt.subplots(figsize=(3.25, 2.5))
            ax1.plot(epoch, train_loss, marker="o", color="#264653", label="train_loss_z")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Train loss (z)")

            ax2 = ax1.twinx()
            ax2.plot(epoch, val_rmse, marker="s", color="#E76F51", label="val_rmse_chl")
            ax2.set_ylabel("Val RMSE (z)")
            ax1.set_title("Slide teacher training (short run)")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            fig.tight_layout()
            fig.savefig(out_dir / "fig_training_curve_slide.pdf")
            fig.savefig(out_dir / "fig_training_curve_slide.png", dpi=300)
            plt.close(fig)

    # Save a small JSON manifest for convenience
    manifest = {
        "window_dir": str(window_dir),
        "split": split,
        "ckpt": str(ckpt_path),
        "scaler_json": str(scaler_json),
        "model_config": asdict(cfg),
        "outputs": sorted(
            [p.name for p in out_dir.glob("fig_*.pdf")]
            + [p.name for p in out_dir.glob("daily_horizon_rmse_*.json")]
        ),
    }
    (out_dir / "viz_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()

