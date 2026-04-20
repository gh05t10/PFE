#!/usr/bin/env python3
"""
Minimal GRU baseline training on windowed NPZ dataset.

Uses:
  - X_z, X_mask, y_z, y_mask from windowed_L*_H*_S*/{train,val}.npz
  - scaler_params.json for reporting RMSE/MAE in original Chl RFU units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import WindowNPZDataset
from src.eval_baselines import run_all_baselines
from src.models_baseline import GRUBaseline, GRUBaselineConfig
from src.resample_config import freq_slug, get_resample_freq


def inverse_target_from_json(z: np.ndarray, scaler_json: Path) -> np.ndarray:
    d = json.loads(Path(scaler_json).read_text(encoding="utf-8"))
    mu = float(d["target"]["mean"])
    std = float(d["target"]["std"])
    return z * std + mu


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, scaler_json: Path) -> dict[str, float]:
    model.eval()
    se = []
    ae = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            x_mask = batch["x_mask"].to(device)
            y = batch["y"].to(device)
            y_mask = batch["y_mask"].to(device)

            # keep only valid targets
            valid = y_mask & torch.isfinite(y)
            if valid.sum() == 0:
                continue
            y = y[valid]
            pred = model(x, x_mask)[valid]

            # back to numpy for inverse transform
            y_np = y.detach().cpu().numpy()
            p_np = pred.detach().cpu().numpy()
            y_raw = inverse_target_from_json(y_np, scaler_json)
            p_raw = inverse_target_from_json(p_np, scaler_json)
            se.append((p_raw - y_raw) ** 2)
            ae.append(np.abs(p_raw - y_raw))

    if not se:
        return {"rmse": float("nan"), "mae": float("nan")}
    se_all = np.concatenate(se)
    ae_all = np.concatenate(ae)
    return {"rmse": float(np.sqrt(se_all.mean())), "mae": float(ae_all.mean())}


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Baseline GRU training on windowed NPZ data.")
    p.add_argument("--freq", default=None, help="resample slug (default: env or 30min)")
    p.add_argument("--rule-a", action="store_true", help="use resampled_<slug>_ruleA/normalized_split")
    p.add_argument(
        "--window-dir",
        type=Path,
        default=None,
        help="folder with train.npz / val.npz (default: auto under normalized_split)",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="where to save best_gru.pt (default: <window-dir>/checkpoints)",
    )
    p.add_argument(
        "--skip-baselines",
        action="store_true",
        help="do not write baseline_metrics.json before training",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    freq = get_resample_freq(cli=args.freq)
    slug = freq_slug(freq)
    ra = "_ruleA" if args.rule_a else ""
    norm_dir = base / "processed" / "chl_shallow" / f"resampled_{slug}{ra}" / "normalized_split"
    if args.window_dir is None:
        # pick the last windowed_* directory if many exist
        cand = sorted(norm_dir.glob("windowed_L*_H*_S*"))
        if not cand:
            raise SystemExit(f"No windowed_* dir under {norm_dir}; run run_build_window_dataset.py first.")
        window_dir = cand[-1]
    else:
        window_dir = args.window_dir

    train_npz = window_dir / "train.npz"
    val_npz = window_dir / "val.npz"
    test_npz = window_dir / "test.npz"
    scaler_json = norm_dir / "scaler_params.json"

    if not train_npz.is_file() or not val_npz.is_file():
        raise SystemExit(f"Missing train/val NPZ in {window_dir}")
    if not scaler_json.is_file():
        raise SystemExit(f"Missing scaler_params.json in {norm_dir}")

    if not args.skip_baselines:
        bl = run_all_baselines(train_npz, val_npz, test_npz if test_npz.is_file() else None, scaler_json)
        bl_path = window_dir / "baseline_metrics.json"
        bl_path.write_text(json.dumps(bl, indent=2, default=str), encoding="utf-8")
        print("=== Naive baselines (RFU) ===")
        print(json.dumps(bl, indent=2, default=str))
        print(f"(saved {bl_path})\n")

    ckpt_dir = args.checkpoint_dir or (window_dir / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_gru.pt"

    train_ds = WindowNPZDataset(train_npz)
    val_ds = WindowNPZDataset(val_npz)
    test_ds = WindowNPZDataset(test_npz) if test_npz.is_file() else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if test_ds is not None else None
    )

    cfg = GRUBaselineConfig()
    model = GRUBaseline(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            x_mask = batch["x_mask"].to(device)
            y = batch["y"].to(device)
            y_mask = batch["y_mask"].to(device)

            valid = y_mask & torch.isfinite(y)
            if valid.sum() == 0:
                continue

            opt.zero_grad()
            pred = model(x, x_mask)
            loss = loss_fn(pred[valid], y[valid])
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        metrics = eval_epoch(model, val_loader, device, scaler_json)
        print(
            f"Epoch {epoch:02d}: train_MSE_z={avg_loss:.4f}, "
            f"val_RMSE_Chl={metrics['rmse']:.4f}, val_MAE_Chl={metrics['mae']:.4f}"
        )

        if np.isfinite(metrics["rmse"]) and metrics["rmse"] < best_val_rmse:
            best_val_rmse = metrics["rmse"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": {"hidden_dim": cfg.hidden_dim, "num_layers": cfg.num_layers},
                    "val_rmse_chl_rf": best_val_rmse,
                    "val_mae_chl_rf": metrics["mae"],
                    "window_dir": str(window_dir.resolve()),
                    "scaler_json": str(scaler_json.resolve()),
                },
                best_path,
            )

    print(f"\nBest checkpoint: epoch {best_epoch}, val_RMSE_Chl={best_val_rmse:.4f} → {best_path}")

    if test_loader is not None and best_path.is_file():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_m = eval_epoch(model, test_loader, device, scaler_json)
        summary = {
            "best_epoch": best_epoch,
            "val_rmse_chl": best_val_rmse,
            "val_mae_chl": checkpoint.get("val_mae_chl_rf"),
            "test_rmse_chl": test_m["rmse"],
            "test_mae_chl": test_m["mae"],
            "checkpoint": str(best_path.resolve()),
        }
        summ_path = ckpt_dir / "gru_eval_summary.json"
        summ_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"Test (best val): RMSE_Chl={test_m['rmse']:.4f}, MAE_Chl={test_m['mae']:.4f} → {summ_path}"
        )


if __name__ == "__main__":
    main()

