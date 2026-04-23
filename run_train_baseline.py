#!/usr/bin/env python3
"""
GRU baseline training on windowed NPZ with common training utilities:

  - reproducibility (seed), AdamW + weight decay, gradient clipping
  - LR scheduler (plateau / cosine / none)
  - early stopping on val RMSE (RFU)
  - CSV log + train_config.json + best checkpoint + test eval
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import WindowNPZDataset
from src.eval_baselines import run_all_baselines
from src.models_baseline import GRUBaseline, GRUBaselineConfig
from src.resample_config import freq_slug, get_resample_freq
from src.train_utils import set_seed
from src.window_pick import pick_window_dir


def inverse_target_from_json(z: np.ndarray, scaler_json: Path) -> np.ndarray:
    d = json.loads(Path(scaler_json).read_text(encoding="utf-8"))
    mu = float(d["target"]["mean"])
    std = float(d["target"]["std"])
    return z * std + mu


def _torch_load_trusted(path: Path, device: torch.device) -> dict:
    """
    PyTorch 2.6+ defaults `weights_only=True`, which can fail when our checkpoint
    includes non-tensor metadata (e.g. Paths / argparse namespaces). We only load checkpoints we wrote.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, scaler_json: Path) -> dict[str, float]:
    model.eval()
    se = []
    ae = []
    n_valid = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            x_mask = batch["x_mask"].to(device)
            y = batch.get("y_seq")
            y_mask = batch.get("y_seq_mask")
            if y is None or y_mask is None:
                y = batch["y"]
                y_mask = batch["y_mask"]
            y = y.to(device)
            y_mask = y_mask.to(device)

            valid = y_mask & torch.isfinite(y)
            pred = model(x, x_mask)
            valid = valid & torch.isfinite(pred)
            if valid.sum() == 0:
                continue

            y_np = y[valid].detach().cpu().numpy()
            p_np = pred[valid].detach().cpu().numpy()
            y_raw = inverse_target_from_json(y_np, scaler_json)
            p_raw = inverse_target_from_json(p_np, scaler_json)
            e = p_raw - y_raw
            ok = np.isfinite(e)
            if ok.sum() == 0:
                continue
            se.append((e[ok]) ** 2)
            ae.append(np.abs(e[ok]))
            n_valid += int(ok.sum())

    if not se:
        return {"rmse": float("nan"), "mae": float("nan"), "n_valid": 0}
    se_all = np.concatenate(se)
    ae_all = np.concatenate(ae)
    return {"rmse": float(np.sqrt(se_all.mean())), "mae": float(ae_all.mean()), "n_valid": int(n_valid)}


def weighted_mse(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/y: (B,) or (B,H); w: same shape or None; mask: bool same shape.
    """
    if w is None:
        err2 = (pred - y) ** 2
        return err2[mask].mean()
    ww = torch.where(torch.isfinite(w), w, torch.zeros_like(w)).clamp(min=0.0)
    mask2 = mask & (ww > 0)
    if mask2.sum() == 0:
        return torch.tensor(float("nan"), device=pred.device)
    num = ((pred - y) ** 2)[mask2] * ww[mask2]
    den = ww[mask2].sum().clamp_min(1e-12)
    return num.sum() / den


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Baseline GRU training on windowed NPZ data.")
    p.add_argument("--freq", default=None, help="resample slug (default: env or 30min)")
    p.add_argument("--rule-a", action="store_true", help="use resampled_<slug>_ruleA/normalized_split")
    p.add_argument("--window-dir", type=Path, default=None, help="folder with train/val/test.npz")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm; 0 disables")
    p.add_argument(
        "--scheduler",
        choices=("none", "plateau", "cosine"),
        default="plateau",
        help="LR schedule: ReduceLROnPlateau(val RMSE), CosineAnnealingLR, or off",
    )
    p.add_argument("--plateau-patience", type=int, default=2, help="epochs without LR reduce (plateau)")
    p.add_argument("--plateau-factor", type=float, default=0.5, help="LR *= factor on plateau")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="stop if val RMSE does not improve for N epochs; 0 = disabled",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="minimum val RMSE improvement to reset early-stopping counter",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", help="full cudnn deterministic (slower)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--pin-memory", action="store_true", help="DataLoader pin_memory (useful on GPU)")
    p.add_argument("--checkpoint-dir", type=Path, default=None, help="default: <window-dir>/checkpoints")
    p.add_argument("--skip-baselines", action="store_true", help="do not write baseline_metrics.json")
    p.add_argument("--log-csv", type=Path, default=None, help="default: <checkpoint-dir>/metrics.csv")
    args = p.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    freq = get_resample_freq(cli=args.freq)
    slug = freq_slug(freq)
    ra = "_ruleA" if args.rule_a else ""
    norm_dir = base / "processed" / "chl_shallow" / f"resampled_{slug}{ra}" / "normalized_split"
    if args.window_dir is None:
        try:
            window_dir = pick_window_dir(norm_dir)
        except FileNotFoundError:
            raise SystemExit(f"No windowed_* dir under {norm_dir}; run run_build_window_dataset.py first.") from None
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
    log_csv = args.log_csv or (ckpt_dir / "metrics.csv")

    pin = bool(args.pin_memory and device.type == "cuda")
    train_ds = WindowNPZDataset(train_npz)
    val_ds = WindowNPZDataset(val_npz)
    test_ds = WindowNPZDataset(test_npz) if test_npz.is_file() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        if test_ds is not None
        else None
    )

    # Infer pred_len from NPZ if available (multi-step), else legacy scalar.
    tmp = np.load(train_npz, allow_pickle=False)
    pred_len = int(tmp["Y_z"].shape[1]) if "Y_z" in tmp.files else 1
    cfg = GRUBaselineConfig(pred_len=pred_len)
    model = GRUBaseline(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)

    best_val_rmse = float("inf")
    best_epoch = 0
    best_val_mae = float("nan")
    epochs_no_improve = 0
    stopped_reason = "max_epochs"

    with open(log_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(
            fcsv,
            fieldnames=[
                "epoch",
                "train_mse_z",
                "val_rmse_chl",
                "val_mae_chl",
                "val_n_valid",
                "lr",
                "seconds",
            ],
        )
        w.writeheader()

        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            total_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                x = batch["x"].to(device)
                x_mask = batch["x_mask"].to(device)
                y = batch.get("y_seq")
                y_mask = batch.get("y_seq_mask")
                y_w = batch.get("y_seq_w")
                if y is None or y_mask is None:
                    y = batch["y"]
                    y_mask = batch["y_mask"]
                    y_w = batch.get("y_w")
                y = y.to(device)
                y_mask = y_mask.to(device)
                if y_w is not None:
                    y_w = y_w.to(device)

                valid = y_mask & torch.isfinite(y)
                if valid.sum() == 0:
                    continue

                opt.zero_grad()
                pred = model(x, x_mask)
                if not torch.isfinite(pred).any().item():
                    continue
                loss = weighted_mse(pred, y, y_w, valid)
                if not torch.isfinite(loss).item():
                    # Can happen if all weights are zero in this batch.
                    continue
                loss.backward()
                # Guard against NaN/Inf gradients
                bad_grad = False
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all().item():
                        bad_grad = True
                        break
                if bad_grad:
                    opt.zero_grad(set_to_none=True)
                    continue
                if args.grad_clip and args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            metrics = eval_epoch(model, val_loader, device, scaler_json)
            lr_now = float(opt.param_groups[0]["lr"])
            dt = time.perf_counter() - t0

            w.writerow(
                {
                    "epoch": epoch,
                    "train_mse_z": f"{avg_loss:.6f}",
                    "val_rmse_chl": f"{metrics['rmse']:.6f}",
                    "val_mae_chl": f"{metrics['mae']:.6f}",
                    "val_n_valid": int(metrics.get("n_valid", 0)),
                    "lr": f"{lr_now:.8f}",
                    "seconds": f"{dt:.2f}",
                }
            )
            fcsv.flush()

            print(
                f"Epoch {epoch:02d}: train_MSE_z={avg_loss:.4f}, "
                f"val_RMSE_Chl={metrics['rmse']:.4f}, val_MAE_Chl={metrics['mae']:.4f}, "
                f"val_n_valid={int(metrics.get('n_valid', 0))}, lr={lr_now:.2e}"
            )

            improved = np.isfinite(metrics["rmse"]) and (
                metrics["rmse"] < best_val_rmse - args.early_stopping_min_delta
            )
            if improved:
                best_val_rmse = metrics["rmse"]
                best_val_mae = metrics["mae"]
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                        "config": {"hidden_dim": cfg.hidden_dim, "num_layers": cfg.num_layers},
                        "val_rmse_chl_rf": best_val_rmse,
                        "val_mae_chl_rf": best_val_mae,
                        "window_dir": str(window_dir.resolve()),
                        "scaler_json": str(scaler_json.resolve()),
                        "train_args": vars(args),
                    },
                    best_path,
                )
            else:
                epochs_no_improve += 1

            if scheduler is not None:
                if args.scheduler == "plateau":
                    scheduler.step(metrics["rmse"])
                else:
                    scheduler.step()

            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                stopped_reason = f"early_stopping_{args.early_stopping_patience}"
                print(f"Early stopping (no val improvement for {args.early_stopping_patience} epochs).")
                break

    print(f"\nBest checkpoint: epoch {best_epoch}, val_RMSE_Chl={best_val_rmse:.4f} → {best_path}")

    train_config = {
        "stopped_reason": stopped_reason,
        "best_epoch": best_epoch,
        "best_val_rmse_chl": best_val_rmse,
        "best_val_mae_chl": best_val_mae,
        "window_dir": str(window_dir.resolve()),
        "scaler_json": str(scaler_json.resolve()),
        "device": str(device),
        "hyperparams": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
    }
    (ckpt_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, default=str), encoding="utf-8")

    if test_loader is not None and best_path.is_file():
        checkpoint = _torch_load_trusted(best_path, device)
        model.load_state_dict(checkpoint["model_state"])
        test_m = eval_epoch(model, test_loader, device, scaler_json)
        summary = {
            "best_epoch": best_epoch,
            "val_rmse_chl": best_val_rmse,
            "val_mae_chl": best_val_mae,
            "test_rmse_chl": test_m["rmse"],
            "test_mae_chl": test_m["mae"],
            "test_n_valid": int(test_m.get("n_valid", 0)),
            "checkpoint": str(best_path.resolve()),
            "stopped_reason": stopped_reason,
        }
        summ_path = ckpt_dir / "gru_eval_summary.json"
        summ_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"Test (best val): RMSE_Chl={test_m['rmse']:.4f}, MAE_Chl={test_m['mae']:.4f} → {summ_path}"
        )


if __name__ == "__main__":
    main()
