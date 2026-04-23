#!/usr/bin/env python3
"""
Train slide-style model: Patch encoders + cross-attention (see ``src/models_slide.py``).

Requires window NPZ with ``X6_z`` / ``X6_mask`` — rebuild with ``run_build_window_dataset.py`` first.

Uses the same training utilities as ``run_train_baseline.py`` (AdamW, scheduler, early stopping, CSV log).
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

from src.datasets import SlideWindowNPZDataset
from src.eval_baselines import run_all_baselines
from src.models_slide import SlidePatchCrossAttn, SlidePatchCrossAttnConfig
from src.resample_config import freq_slug, get_resample_freq
from src.tdalign_loss import tdalign_total_loss
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
    includes non-tensor metadata (e.g. Paths). We only load checkpoints we wrote.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older torch without weights_only kwarg.
        return torch.load(path, map_location=device)


def _json_safe(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def collect_valid_preds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Return (y_z_valid, pred_z_valid, context_end_time_valid, target_time_valid) concatenated over the loader.
    Only includes samples where y_mask is true and y is finite.
    """
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    cts: list[np.ndarray] = []
    tts: list[np.ndarray] = []
    have_times: bool | None = None
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            x_mask = batch["x_mask"].to(device)
            x6 = batch["x6"].to(device)
            x6_mask = batch["x6_mask"].to(device)
            y = batch.get("y_seq")
            y_mask = batch.get("y_seq_mask")
            if y is None or y_mask is None:
                y = batch["y"]
                y_mask = batch["y_mask"]
            y = y.to(device)
            y_mask = y_mask.to(device)

            valid = y_mask & torch.isfinite(y)
            if valid.sum() == 0:
                continue
            pred = model(x, x_mask, x6, x6_mask)
            ys.append(y[valid].detach().cpu().numpy())
            ps.append(pred[valid].detach().cpu().numpy())
            if have_times is None:
                have_times = ("context_end_time_ns" in batch) and ("target_time_ns" in batch)
            if have_times:
                # Times are numpy datetime64 scalars/arrays; keep them on CPU.
                v_np = valid.detach().cpu().numpy().astype(bool)
                cts.append(np.asarray(batch["context_end_time_ns"])[v_np].astype(np.int64))
                tts.append(np.asarray(batch["target_time_ns"])[v_np].astype(np.int64))

    if not ys:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), None, None

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    if have_times:
        return y_all, p_all, np.concatenate(cts), np.concatenate(tts)
    return y_all, p_all, None, None


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler_json: Path,
) -> dict[str, float]:
    model.eval()
    se = []
    ae = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            x_mask = batch["x_mask"].to(device)
            x6 = batch["x6"].to(device)
            x6_mask = batch["x6_mask"].to(device)
            y = batch.get("y_seq")
            y_mask = batch.get("y_seq_mask")
            if y is None or y_mask is None:
                y = batch["y"]
                y_mask = batch["y_mask"]
            y = y.to(device)
            y_mask = y_mask.to(device)
            chl_end = batch.get("chl_end")
            chl_end_mask = batch.get("chl_end_mask")

            valid = y_mask & torch.isfinite(y)
            if valid.sum() == 0:
                continue
            pred_z = model(x, x_mask, x6, x6_mask)
            valid = valid & torch.isfinite(pred_z)
            if valid.sum() == 0:
                continue
            if getattr(model, "_predicts_delta", False):
                if chl_end is None or chl_end_mask is None:
                    raise RuntimeError("Residual mode requires chl_end/chl_end_mask in batch.")
                ce = chl_end.to(device)
                cem = chl_end_mask.to(device)
                if valid.ndim == 2:
                    valid = valid & cem[:, None] & torch.isfinite(ce)[:, None]
                else:
                    valid = valid & cem & torch.isfinite(ce)
                if valid.sum() == 0:
                    continue
                if pred_z.ndim == 1:
                    pred = (pred_z + ce)[valid]
                else:
                    pred = (pred_z + ce[:, None])[valid]
                y = y[valid]
            else:
                y = y[valid]
                pred = pred_z[valid]

            y_np = y.detach().cpu().numpy()
            p_np = pred.detach().cpu().numpy()
            y_raw = inverse_target_from_json(y_np, scaler_json)
            p_raw = inverse_target_from_json(p_np, scaler_json)
            e = p_raw - y_raw
            ok = np.isfinite(e)
            if ok.sum() == 0:
                continue
            se.append((e[ok]) ** 2)
            ae.append(np.abs(e[ok]))

    if not se:
        return {"rmse": float("nan"), "mae": float("nan")}
    se_all = np.concatenate(se)
    ae_all = np.concatenate(ae)
    return {"rmse": float(np.sqrt(se_all.mean())), "mae": float(ae_all.mean())}


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Slide PatchTST + cross-attention training.")
    p.add_argument("--freq", default=None)
    p.add_argument("--rule-a", action="store_true")
    p.add_argument("--window-dir", type=Path, default=None)
    p.add_argument("--patch-len", type=int, default=16, help="must divide context length L (e.g. 96/16=6 patches)")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--scheduler", choices=("none", "plateau", "cosine"), default="plateau")
    p.add_argument("--plateau-patience", type=int, default=2)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--early-stopping-patience", type=int, default=8)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--log-csv", type=Path, default=None)
    p.add_argument(
        "--save-preds",
        action="store_true",
        help="save y_true/y_pred arrays (valid samples only) for best checkpoint to checkpoints_slide/preds_{val,test}.npz",
    )
    p.add_argument(
        "--residual",
        action="store_true",
        help="train to predict delta_z = y_z - chl_end_z; final y_hat_z = chl_end_z + delta_z (requires chl_z_at_window_end in NPZ)",
    )
    p.add_argument(
        "--tdalign",
        action="store_true",
        help="use TDAlign objective on multi-step horizons (requires Y_z in NPZ and chl_end in batch)",
    )
    p.add_argument(
        "--tdalign-loss",
        choices=("mse", "mae"),
        default="mse",
        help="loss type for L_Y and L_D in TDAlign",
    )
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
            raise SystemExit(f"No windowed_* under {norm_dir}") from None
    else:
        window_dir = args.window_dir

    train_npz = window_dir / "train.npz"
    val_npz = window_dir / "val.npz"
    test_npz = window_dir / "test.npz"
    scaler_json = norm_dir / "scaler_params.json"

    for f in (train_npz, val_npz, scaler_json):
        if not f.is_file():
            raise SystemExit(f"Missing {f}")

    tmp = np.load(train_npz, allow_pickle=False)
    if "X6_z" not in tmp.files:
        raise SystemExit(
            "train.npz has no X6_z. Re-run: python run_build_window_dataset.py (same --freq / --rule-a / L / H / S)"
        )
    L = int(tmp["X_z"].shape[1])
    if L % args.patch_len != 0:
        raise SystemExit(f"context_len L={L} not divisible by --patch-len={args.patch_len}")
    pred_len = int(tmp["Y_z"].shape[1]) if "Y_z" in tmp.files else 1

    if not args.skip_baselines:
        bl = run_all_baselines(train_npz, val_npz, test_npz if test_npz.is_file() else None, scaler_json)
        bl_path = window_dir / "baseline_metrics.json"
        bl_path.write_text(json.dumps(bl, indent=2, default=str), encoding="utf-8")
        print("=== Naive baselines (RFU) ===")
        print(json.dumps(bl, indent=2, default=str))
        print(f"(saved {bl_path})\n")

    ckpt_dir = args.checkpoint_dir or (window_dir / "checkpoints_slide")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = "best_slide_residual.pt" if args.residual else "best_slide.pt"
    best_path = ckpt_dir / ckpt_name
    log_name = "metrics_slide_residual.csv" if args.residual else "metrics_slide.csv"
    log_csv = args.log_csv or (ckpt_dir / log_name)

    pin = bool(args.pin_memory and device.type == "cuda")
    train_ds = SlideWindowNPZDataset(train_npz)
    val_ds = SlideWindowNPZDataset(val_npz)
    test_ds = SlideWindowNPZDataset(test_npz) if test_npz.is_file() else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin
    )
    test_loader = (
        DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin
        )
        if test_ds is not None
        else None
    )

    mcfg = SlidePatchCrossAttnConfig(
        patch_len=args.patch_len,
        d_model=args.d_model,
        nhead=args.nhead,
        encoder_layers=args.encoder_layers,
        pred_len=pred_len,
        dropout=0.1,
    )
    model = SlidePatchCrossAttn(mcfg).to(device)
    if args.residual:
        # marker used in eval/pred saving
        setattr(model, "_predicts_delta", True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr
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
                "train_loss_z",
                "train_loss_name",
                "train_LY",
                "train_LD",
                "train_rho",
                "val_rmse_chl",
                "val_mae_chl",
                "lr",
                "seconds",
            ],
        )
        w.writeheader()

        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            total_loss = 0.0
            total_ly = 0.0
            total_ld = 0.0
            total_rho = 0.0
            n_batches = 0
            n_tdalign_batches = 0
            for batch in train_loader:
                x = batch["x"].to(device)
                x_mask = batch["x_mask"].to(device)
                x6 = batch["x6"].to(device)
                x6_mask = batch["x6_mask"].to(device)
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
                chl_end = batch.get("chl_end")
                chl_end_mask = batch.get("chl_end_mask")

                valid = y_mask & torch.isfinite(y)
                if valid.sum() == 0:
                    continue

                opt.zero_grad()
                pred = model(x, x_mask, x6, x6_mask)
                if not torch.isfinite(pred).all().item():
                    # Numerical issue; skip this batch to avoid poisoning weights.
                    continue
                if args.residual:
                    if chl_end is None or chl_end_mask is None:
                        raise RuntimeError("Residual mode requires chl_end/chl_end_mask in batch.")
                    ce = chl_end.to(device)
                    cem = chl_end_mask.to(device)
                    # valid is (B,H) for multi-step; cem/ce are (B,)
                    if y.ndim == 2:
                        valid2 = valid & cem[:, None] & torch.isfinite(ce)[:, None]
                    else:
                        valid2 = valid & cem & torch.isfinite(ce)
                    if valid2.sum() == 0:
                        continue
                    delta = (y - (ce[:, None] if y.ndim == 2 else ce))
                    if y_w is None or y.ndim == 1:
                        loss = loss_fn(pred[valid2], delta[valid2])
                    else:
                        ww = torch.where(torch.isfinite(y_w), y_w, torch.zeros_like(y_w)).clamp(min=0.0)
                        m = valid2 & (ww > 0)
                        num = ((pred - delta) ** 2)[m] * ww[m]
                        den = ww[m].sum().clamp_min(1e-12)
                        loss = num.sum() / den
                elif args.tdalign and y.ndim == 2:
                    if chl_end is None or chl_end_mask is None:
                        raise RuntimeError("TDAlign requires chl_end/chl_end_mask in batch.")
                    ce = chl_end.to(device)
                    cem = chl_end_mask.to(device)
                    valid2 = valid & cem[:, None] & torch.isfinite(ce)[:, None]
                    if valid2.sum() == 0:
                        continue
                    l, ly, ld, rho = tdalign_total_loss(
                        y_true=y,
                        y_hat=pred,
                        y_end=ce,
                        mask=valid2,
                        weights=y_w if (y_w is not None and y_w.ndim == 2) else None,
                        loss=args.tdalign_loss,
                    )
                    loss = l
                    # Accumulate TDAlign diagnostics per-batch (meaned later).
                    if torch.isfinite(ly).item():
                        total_ly += float(ly.item())
                        total_ld += float(ld.item())
                        total_rho += float(rho.item())
                        n_tdalign_batches += 1
                else:
                    if y_w is None or y.ndim == 1:
                        loss = loss_fn(pred[valid], y[valid])
                    else:
                        ww = torch.where(torch.isfinite(y_w), y_w, torch.zeros_like(y_w)).clamp(min=0.0)
                        m = valid & (ww > 0)
                        num = ((pred - y) ** 2)[m] * ww[m]
                        den = ww[m].sum().clamp_min(1e-12)
                        loss = num.sum() / den
                if not torch.isfinite(loss).item() or (not loss.requires_grad):
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
                # Guard against NaN/Inf parameters
                bad_param = False
                for p in model.parameters():
                    if not torch.isfinite(p).all().item():
                        bad_param = True
                        break
                if bad_param:
                    raise RuntimeError("Model parameters became NaN/Inf during training (try lower LR / stronger clip).")

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            if n_tdalign_batches > 0:
                avg_ly = total_ly / n_tdalign_batches
                avg_ld = total_ld / n_tdalign_batches
                avg_rho = total_rho / n_tdalign_batches
            else:
                avg_ly = float("nan")
                avg_ld = float("nan")
                avg_rho = float("nan")
            metrics = eval_epoch(model, val_loader, device, scaler_json)
            lr_now = float(opt.param_groups[0]["lr"])
            dt = time.perf_counter() - t0
            loss_name = (
                f"tdalign_{args.tdalign_loss}" if (args.tdalign and not args.residual and pred_len > 1) else "mse"
            )

            w.writerow(
                {
                    "epoch": epoch,
                    "train_loss_z": f"{avg_loss:.6f}",
                    "train_loss_name": loss_name,
                    "train_LY": f"{avg_ly:.6f}" if np.isfinite(avg_ly) else "nan",
                    "train_LD": f"{avg_ld:.6f}" if np.isfinite(avg_ld) else "nan",
                    "train_rho": f"{avg_rho:.6f}" if np.isfinite(avg_rho) else "nan",
                    "val_rmse_chl": f"{metrics['rmse']:.6f}",
                    "val_mae_chl": f"{metrics['mae']:.6f}",
                    "lr": f"{lr_now:.8f}",
                    "seconds": f"{dt:.2f}",
                }
            )
            fcsv.flush()

            msg = (
                f"Epoch {epoch:02d}: train_{loss_name}_z={avg_loss:.4f}, "
                f"val_RMSE_Chl={metrics['rmse']:.4f}, val_MAE_Chl={metrics['mae']:.4f}, lr={lr_now:.2e}"
            )
            if args.tdalign and n_tdalign_batches > 0:
                msg += f" (LY={avg_ly:.4f}, LD={avg_ld:.4f}, rho={avg_rho:.3f})"
            print(msg)

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
                        "model_config": {
                            "patch_len": mcfg.patch_len,
                            "d_model": mcfg.d_model,
                            "nhead": mcfg.nhead,
                            "encoder_layers": mcfg.encoder_layers,
                            "pred_len": mcfg.pred_len,
                        },
                        "val_rmse_chl_rf": best_val_rmse,
                        "val_mae_chl_rf": best_val_mae,
                        "window_dir": str(window_dir.resolve()),
                        "scaler_json": str(scaler_json.resolve()),
                        "train_args": _json_safe(vars(args)),
                    },
                    best_path,
                )
            else:
                epochs_no_improve += 1
                # Ensure we have at least one compatible checkpoint for this mode.
                if best_epoch == 0:
                    best_epoch = epoch
                    best_val_rmse = float(metrics["rmse"])
                    best_val_mae = float(metrics["mae"])
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": opt.state_dict(),
                            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                            "model_config": {
                                "patch_len": mcfg.patch_len,
                                "d_model": mcfg.d_model,
                                "nhead": mcfg.nhead,
                                "encoder_layers": mcfg.encoder_layers,
                                "pred_len": mcfg.pred_len,
                            },
                            "val_rmse_chl_rf": best_val_rmse,
                            "val_mae_chl_rf": best_val_mae,
                            "window_dir": str(window_dir.resolve()),
                            "scaler_json": str(scaler_json.resolve()),
                            "train_args": _json_safe(vars(args)),
                        },
                        best_path,
                    )

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
        "model": "SlidePatchCrossAttn",
        "window_dir": str(window_dir.resolve()),
        "hyperparams": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (ckpt_dir / "train_config_slide.json").write_text(json.dumps(train_config, indent=2, default=str), encoding="utf-8")

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
            "checkpoint": str(best_path.resolve()),
            "stopped_reason": stopped_reason,
        }
        (ckpt_dir / "slide_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"Test (best val): RMSE_Chl={test_m['rmse']:.4f}, MAE_Chl={test_m['mae']:.4f} → {ckpt_dir / 'slide_eval_summary.json'}"
        )

        if args.save_preds:
            # Save valid-only arrays to keep file small & avoid ambiguity about masked targets.
            yv_z, pv_z, yv_ct, yv_tt = collect_valid_preds(model, val_loader, device)
            yt_z, pt_z, yt_ct, yt_tt = collect_valid_preds(model, test_loader, device)

            # If residual mode: pv_z / pt_z are delta_z; reconstruct y_hat_z with chl_end from NPZ.
            if args.residual:
                val_npz = np.load(window_dir / "val.npz", allow_pickle=False)
                test_npz = np.load(window_dir / "test.npz", allow_pickle=False)
                # valid-only arrays were collected in order of dataloader; easiest is to recompute preds with end added.
                # Re-collect with ends applied by calling eval loop style.
                # For saving, we only guarantee y_true/y_pred in RFU align; z values stored as future z.
                # We'll override pv_z/pt_z to be y_hat_z (future) for convenience.
                # NOTE: This assumes `collect_valid_preds` returns valid samples in the same order for y and pred.
                # We rebuild by iterating loader once more.
                def collect_future(model, loader):
                    model.eval()
                    ys=[]; ps=[]
                    with torch.no_grad():
                        for b in loader:
                            x=b["x"].to(device); xm=b["x_mask"].to(device)
                            x6=b["x6"].to(device); x6m=b["x6_mask"].to(device)
                            y=b["y"].to(device); ym=b["y_mask"].to(device)
                            ce=b["chl_end"].to(device); cem=b["chl_end_mask"].to(device)
                            v=ym & torch.isfinite(y) & cem & torch.isfinite(ce)
                            if v.sum()==0: continue
                            dz=model(x,xm,x6,x6m)
                            ys.append(y[v].cpu().numpy()); ps.append((dz+ce)[v].cpu().numpy())
                    return np.concatenate(ys) if ys else np.array([],dtype=np.float32), np.concatenate(ps) if ps else np.array([],dtype=np.float32)
                yv_z, pv_z = collect_future(model, val_loader)
                yt_z, pt_z = collect_future(model, test_loader)

            yv_rf = inverse_target_from_json(yv_z, scaler_json)
            pv_rf = inverse_target_from_json(pv_z, scaler_json)
            yt_rf = inverse_target_from_json(yt_z, scaler_json)
            pt_rf = inverse_target_from_json(pt_z, scaler_json)

            np.savez_compressed(
                ckpt_dir / "preds_val.npz",
                y_true_z=yv_z,
                y_pred_z=pv_z,
                y_true_rf=yv_rf,
                y_pred_rf=pv_rf,
                n_valid=int(yv_z.size),
                context_end_time=(
                    yv_ct.astype("datetime64[ns]") if yv_ct is not None else np.array([], dtype="datetime64[ns]")
                ),
                target_time=(
                    yv_tt.astype("datetime64[ns]") if yv_tt is not None else np.array([], dtype="datetime64[ns]")
                ),
            )
            np.savez_compressed(
                ckpt_dir / "preds_test.npz",
                y_true_z=yt_z,
                y_pred_z=pt_z,
                y_true_rf=yt_rf,
                y_pred_rf=pt_rf,
                n_valid=int(yt_z.size),
                context_end_time=(
                    yt_ct.astype("datetime64[ns]") if yt_ct is not None else np.array([], dtype="datetime64[ns]")
                ),
                target_time=(
                    yt_tt.astype("datetime64[ns]") if yt_tt is not None else np.array([], dtype="datetime64[ns]")
                ),
            )
            print(f"Saved preds: {ckpt_dir / 'preds_val.npz'} and {ckpt_dir / 'preds_test.npz'}")


if __name__ == "__main__":
    main()
