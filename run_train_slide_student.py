#!/usr/bin/env python3
"""
Train Setup-B student model (deployable): inference uses only 5 soft-sensor channels.

Student is trained with distillation from a teacher SlidePatchCrossAttn checkpoint:
  - supervised loss: MSE(student_pred_z, y_z)
  - distill loss:    MSE(student_pred_z, teacher_pred_z.detach())

Both losses are computed on valid targets only (y_mask & finite).
Metrics are reported in RFU via scaler_params.json (same as other scripts).
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
from src.models_slide import SlidePatchCrossAttn, SlidePatchCrossAttnConfig, SlideStudentCrossAttn, SlideStudentResidual
from src.models_student_simple import GRUStudent, GRUStudentConfig, MLPStudent, MLPStudentConfig
from src.resample_config import freq_slug, get_resample_freq
from src.tdalign_loss import tdalign_total_loss
from src.train_utils import set_seed
from src.window_pick import pick_window_dir


def inverse_target_from_json(z: np.ndarray, scaler_json: Path) -> np.ndarray:
    d = json.loads(Path(scaler_json).read_text(encoding="utf-8"))
    mu = float(d["target"]["mean"])
    std = float(d["target"]["std"])
    return z * std + mu


def masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Numerically safe masked (weighted) MSE.

    Key trick: avoid NaNs in the computation graph by filling target with pred.detach()
    where mask is false, so error becomes 0 there without evaluating NaN branches.
    """
    m = mask & torch.isfinite(pred)
    if weights is not None:
        w = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights)).clamp(min=0.0)
        m = m & (w > 0)
    else:
        w = None

    if m.sum() == 0:
        return torch.tensor(float("nan"), device=pred.device)

    tgt = torch.where(m, target, pred.detach())
    err2 = (pred - tgt) ** 2
    if w is None:
        return err2[m].mean()
    ww = w[m]
    return (err2[m] * ww).sum() / ww.sum().clamp_min(1e-12)


def _torch_load_trusted(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _json_safe(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@torch.no_grad()
def collect_valid_preds_student(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Return (y_z_valid, pred_z_valid, context_end_time_ns_valid, target_time_ns_valid) concatenated over loader.
    Only includes samples where y_mask is true and y is finite.
    """
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    cts: list[np.ndarray] = []
    tts: list[np.ndarray] = []
    have_times: bool | None = None

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
        if valid.sum() == 0:
            continue
        pred = model(x, x_mask)
        ys.append(y[valid].detach().cpu().numpy())
        ps.append(pred[valid].detach().cpu().numpy())

        if have_times is None:
            have_times = ("context_end_time_ns" in batch) and ("target_time_ns" in batch)
        if have_times:
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


@torch.no_grad()
def eval_epoch_student(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler_json: Path,
) -> dict[str, float]:
    model.eval()
    se = []
    ae = []
    n_valid = 0
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
        if valid.sum() == 0:
            continue
        pred = model(x, x_mask)
        # Residual student returns (y_future_hat, y_end_hat, delta_hat)
        if isinstance(pred, tuple):
            pred = pred[0]
        valid = valid & torch.isfinite(pred)
        if valid.sum() == 0:
            continue
        yv = y[valid]
        pv = pred[valid]

        y_np = yv.detach().cpu().numpy()
        p_np = pv.detach().cpu().numpy()
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


def load_teacher(ckpt_path: Path, device: torch.device) -> SlidePatchCrossAttn:
    ckpt = _torch_load_trusted(ckpt_path, device)
    mcfg_d = ckpt.get("model_config") or {}
    if "pred_len" not in mcfg_d:
        # Backward compatibility: older checkpoints didn't store pred_len.
        try:
            mcfg_d["pred_len"] = int(ckpt["model_state"]["head.2.bias"].shape[0])
        except Exception:
            mcfg_d["pred_len"] = 1
    cfg = SlidePatchCrossAttnConfig(
        patch_len=int(mcfg_d.get("patch_len", 16)),
        d_model=int(mcfg_d.get("d_model", 64)),
        nhead=int(mcfg_d.get("nhead", 4)),
        encoder_layers=int(mcfg_d.get("encoder_layers", 2)),
        pred_len=int(mcfg_d.get("pred_len", 1)),
        dropout=0.1,
    )
    model = SlidePatchCrossAttn(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main() -> None:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Train Setup-B student (X5 only) with teacher distillation.")
    p.add_argument("--freq", default=None)
    p.add_argument("--rule-a", action="store_true")
    p.add_argument("--window-dir", type=Path, default=None)
    p.add_argument("--teacher-ckpt", type=Path, required=True, help="Path to teacher best_slide.pt")
    p.add_argument(
        "--student-arch",
        choices=("transformer", "gru", "mlp"),
        default="transformer",
        help="student architecture (gru is recommended for very long horizons to avoid NaN grads)",
    )
    p.add_argument("--mlp-hidden-dim", type=int, default=256, help="MLP student hidden dim (when --student-arch mlp)")
    p.add_argument("--mlp-dropout", type=float, default=0.2, help="MLP student dropout (when --student-arch mlp)")
    p.add_argument("--patch-len", type=int, default=16)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--kv-encoder-layers", type=int, default=None)
    p.add_argument("--share-q-kv-encoder", action="store_true")
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
    p.add_argument(
        "--alpha-supervised",
        type=float,
        default=0.5,
        help="loss = alpha*MSE(y) + (1-alpha)*MSE(teacher); alpha in [0,1]",
    )
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
        help="save y_true/y_pred arrays (valid samples only) for best checkpoint to checkpoints_slide_student/preds_{val,test}.npz",
    )
    p.add_argument(
        "--residual",
        action="store_true",
        help="deployable residual: student predicts y_end_hat_z + delta_hat_z; supervised on y_end (from chl_z_at_window_end) and delta (y - chl_end)",
    )
    p.add_argument("--beta-end", type=float, default=0.2, help="weight for end-of-window supervised loss when --residual")
    p.add_argument(
        "--tdalign",
        action="store_true",
        help="use TDAlign objective for the supervised term on multi-step horizons (requires Y_z and chl_end in batch)",
    )
    p.add_argument(
        "--tdalign-loss",
        choices=("mse", "mae"),
        default="mse",
        help="loss type for L_Y and L_D in TDAlign",
    )
    args = p.parse_args()

    if not (0.0 <= args.alpha_supervised <= 1.0):
        raise SystemExit("--alpha-supervised must be in [0, 1]")

    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # oneDNN / MKLDNN can produce NaN gradients for some ops on some CPU builds.
    if device.type == "cpu":
        try:
            torch.backends.mkldnn.enabled = False  # type: ignore[attr-defined]
        except Exception:
            pass

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
    for f in (train_npz, val_npz, scaler_json, args.teacher_ckpt):
        if not f.is_file():
            raise SystemExit(f"Missing {f}")

    # Optional baseline metrics on same windows.
    if not args.skip_baselines:
        bl = run_all_baselines(train_npz, val_npz, test_npz if test_npz.is_file() else None, scaler_json)
        bl_path = window_dir / "baseline_metrics.json"
        if not bl_path.is_file():
            bl_path.write_text(json.dumps(bl, indent=2, default=str), encoding="utf-8")
        print("=== Naive baselines (RFU) ===")
        print(json.dumps(bl, indent=2, default=str))
        print(f"(saved {bl_path})\n")

    # Data
    pin = bool(args.pin_memory and device.type == "cuda")
    train_ds = SlideWindowNPZDataset(train_npz)
    val_ds = SlideWindowNPZDataset(val_npz)
    test_ds = SlideWindowNPZDataset(test_npz) if test_npz.is_file() else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
        if test_ds is not None
        else None
    )

    # Teacher (frozen)
    teacher = load_teacher(args.teacher_ckpt, device)

    # Student
    tmp = np.load(train_npz, allow_pickle=False)
    L = int(tmp["X_z"].shape[1])
    if L % args.patch_len != 0:
        raise SystemExit(f"context_len L={L} not divisible by --patch-len={args.patch_len}")
    pred_len = int(tmp["Y_z"].shape[1]) if "Y_z" in tmp.files else 1

    if args.student_arch == "mlp":
        if args.residual:
            raise SystemExit("--student-arch mlp currently supports non-residual only.")
        mcfg2 = MLPStudentConfig(
            context_len=L,
            pred_len=pred_len,
            hidden_dim=int(args.mlp_hidden_dim),
            dropout=float(args.mlp_dropout),
        )
        student: nn.Module = MLPStudent(mcfg2).to(device)
    elif args.student_arch == "gru":
        if args.residual:
            raise SystemExit("--student-arch gru currently supports non-residual only.")
        gcfg = GRUStudentConfig(pred_len=pred_len)
        student: nn.Module = GRUStudent(gcfg).to(device)
    else:
        scfg = SlidePatchCrossAttnConfig(
            patch_len=args.patch_len,
            d_model=args.d_model,
            nhead=args.nhead,
            encoder_layers=args.encoder_layers,
            kv_encoder_layers=args.kv_encoder_layers,
            share_q_kv_encoder=bool(args.share_q_kv_encoder),
            pred_len=pred_len,
            dropout=0.1,
        )
        if args.residual:
            student = SlideStudentResidual(scfg).to(device)
        else:
            student = SlideStudentCrossAttn(scfg).to(device)

    params = list(student.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)

    ckpt_dir = args.checkpoint_dir or (window_dir / "checkpoints_slide_student")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_student.pt"
    log_csv = args.log_csv or (ckpt_dir / "metrics_student.csv")

    best_val_rmse = float("inf")
    best_val_mae = float("nan")
    best_epoch = 0
    epochs_no_improve = 0
    stopped_reason = "max_epochs"

    with open(log_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(
            fcsv,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_loss_sup",
                "train_loss_distill",
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
            student.train()
            total = 0.0
            total_sup = 0.0
            total_dst = 0.0
            n_batches = 0

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
                    y_future_s, y_end_s, delta_s = student(x, x_mask)
                    # Broadcast end over horizon when y is vector.
                    if y.ndim == 1:
                        delta_true = (y - ce)
                    else:
                        delta_true = (y - ce[:, None])
                    # For residual mode, the student outputs y_future_s (not pred_s)
                    pred_s = y_future_s
                else:
                    pred_s = student(x, x_mask)
                with torch.no_grad():
                    pred_t = teacher(x, x_mask, x6, x6_mask)
                # ensure we never compute losses on NaN/inf predictions
                if y.ndim == 2:
                    valid = valid & torch.isfinite(pred_s) & torch.isfinite(pred_t)
                else:
                    valid = valid & torch.isfinite(pred_s) & torch.isfinite(pred_t)
                if valid.sum() == 0:
                    continue

                if args.residual:
                    if y_w is not None and y.ndim == 2:
                        ww = torch.where(torch.isfinite(y_w), y_w, torch.zeros_like(y_w)).clamp(min=0.0)
                        m_bool = valid2 & (ww > 0) & torch.isfinite(delta_s) & torch.isfinite(delta_true)
                        wm = ww * m_bool.float()
                        den = wm.sum().clamp_min(1e-12)
                        # If all weights are zero for this batch, skip it.
                        if den.item() <= 1e-12:
                            continue
                        err2 = (delta_s - delta_true) ** 2
                        err2 = torch.where(m_bool, err2, torch.zeros_like(err2))
                        sup = (err2 * ww).sum() / den
                    else:
                        sup = loss_fn(delta_s[valid2], delta_true[valid2])
                    # Distill target: if teacher was trained in residual mode, it outputs delta_z directly.
                    # We detect that by checkpoint filename convention (best_slide_residual.pt).
                    teacher_is_residual = "residual" in str(args.teacher_ckpt).lower()
                    if teacher_is_residual:
                        tgt = pred_t.detach()
                    else:
                        # teacher predicts y_future; convert to delta target
                        tgt = (pred_t - ce[:, None]).detach()
                    # masked distill MSE without boolean indexing (more stable for long horizons)
                    m_dst_bool = valid2 & torch.isfinite(delta_s) & torch.isfinite(tgt)
                    m_dst = m_dst_bool.float()
                    if m_dst.sum().item() <= 0:
                        continue
                    derr2 = (delta_s - tgt) ** 2
                    derr2 = torch.where(m_dst_bool, derr2, torch.zeros_like(derr2))
                    dst = derr2.sum() / m_dst.sum().clamp_min(1e-12)
                    # y_end_s and ce are (B,)
                    end_mask = cem & torch.isfinite(ce) & torch.isfinite(y_end_s)
                    if end_mask.sum() == 0:
                        continue
                    end_sup = loss_fn(y_end_s[end_mask], ce[end_mask])
                    loss = args.alpha_supervised * sup + (1.0 - args.alpha_supervised) * dst + args.beta_end * end_sup
                else:
                    if args.tdalign and y.ndim == 2:
                        if chl_end is None or chl_end_mask is None:
                            raise RuntimeError("TDAlign requires chl_end/chl_end_mask in batch.")
                        ce = chl_end.to(device)
                        cem = chl_end_mask.to(device)
                        valid2 = valid & cem[:, None] & torch.isfinite(ce)[:, None]
                        if valid2.sum() == 0:
                            continue
                        sup, _, _, _ = tdalign_total_loss(
                            y_true=y,
                            y_hat=pred_s,
                            y_end=ce,
                            mask=valid2,
                            weights=y_w if (y_w is not None and y_w.ndim == 2) else None,
                            loss=args.tdalign_loss,
                        )
                    else:
                        if y_w is not None and y.ndim == 2:
                            sup = masked_weighted_mse(pred_s, y, valid, weights=y_w)
                        else:
                            sup = masked_weighted_mse(pred_s, y, valid, weights=None)
                    # Distill target: if teacher was trained in residual mode, it outputs delta_z.
                    teacher_is_residual = "residual" in str(args.teacher_ckpt).lower()
                    if teacher_is_residual:
                        if chl_end is None or chl_end_mask is None:
                            raise RuntimeError("Residual teacher distill requires chl_end/chl_end_mask in batch.")
                        ce = chl_end.to(device)
                        # teacher_y_future = ce + delta_teacher
                        tgt = (pred_t + ce[:, None]).detach() if pred_t.ndim == 2 else (pred_t + ce).detach()
                    else:
                        tgt = pred_t.detach()
                    dst = masked_weighted_mse(pred_s, tgt, valid, weights=None)
                    loss = args.alpha_supervised * sup + (1.0 - args.alpha_supervised) * dst
                if not torch.isfinite(loss).item():
                    continue
                loss.backward()
                # Guard NaN/Inf gradients
                bad_grad = False
                for p in student.parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all().item():
                        bad_grad = True
                        break
                if bad_grad:
                    opt.zero_grad(set_to_none=True)
                    continue
                if args.grad_clip and args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                opt.step()

                total += float(loss.item())
                total_sup += float(sup.item())
                total_dst += float(dst.item())
                n_batches += 1

            avg = total / max(1, n_batches)
            avg_sup = total_sup / max(1, n_batches)
            avg_dst = total_dst / max(1, n_batches)

            # Eval: `eval_epoch_student` handles both normal and residual student outputs (tuple).
            metrics = eval_epoch_student(student, val_loader, device, scaler_json)
            lr_now = float(opt.param_groups[0]["lr"])
            dt = time.perf_counter() - t0

            w.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{avg:.6f}",
                    "train_loss_sup": f"{avg_sup:.6f}",
                    "train_loss_distill": f"{avg_dst:.6f}",
                    "val_rmse_chl": f"{metrics['rmse']:.6f}",
                    "val_mae_chl": f"{metrics['mae']:.6f}",
                    "val_n_valid": int(metrics.get("n_valid", 0)),
                    "lr": f"{lr_now:.8f}",
                    "seconds": f"{dt:.2f}",
                }
            )
            fcsv.flush()

            print(
                f"Epoch {epoch:02d}: train_loss={avg:.4f} (sup={avg_sup:.4f}, distill={avg_dst:.4f}), "
                f"val_RMSE_Chl={metrics['rmse']:.4f}, val_MAE_Chl={metrics['mae']:.4f}, "
                f"val_n_valid={int(metrics.get('n_valid', 0))}, lr={lr_now:.2e}"
            )

            improved = np.isfinite(metrics["rmse"]) and (metrics["rmse"] < best_val_rmse - args.early_stopping_min_delta)
            if improved:
                best_val_rmse = metrics["rmse"]
                best_val_mae = metrics["mae"]
                best_epoch = epoch
                epochs_no_improve = 0
                if args.student_arch in ("gru", "mlp"):
                    student_cfg_payload = {
                        "arch": args.student_arch,
                        "pred_len": pred_len,
                    }
                else:
                    student_cfg_payload = {
                        "arch": "transformer",
                        "patch_len": scfg.patch_len,
                        "d_model": scfg.d_model,
                        "nhead": scfg.nhead,
                        "encoder_layers": scfg.encoder_layers,
                        "kv_encoder_layers": scfg.kv_encoder_layers,
                        "share_q_kv_encoder": scfg.share_q_kv_encoder,
                        "pred_len": pred_len,
                    }
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": student.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                        "student_config": student_cfg_payload,
                        "teacher_ckpt": str(args.teacher_ckpt.resolve()),
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

            if scheduler is not None:
                if args.scheduler == "plateau":
                    scheduler.step(metrics["rmse"])
                else:
                    scheduler.step()

            if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
                stopped_reason = f"early_stopping_{args.early_stopping_patience}"
                print(f"Early stopping (no val improvement for {args.early_stopping_patience} epochs).")
                break

    print(f"\nBest student checkpoint: epoch {best_epoch}, val_RMSE_Chl={best_val_rmse:.4f} → {best_path}")

    train_config = {
        "stopped_reason": stopped_reason,
        "best_epoch": best_epoch,
        "best_val_rmse_chl": best_val_rmse,
        "model": "SlideStudentCrossAttn",
        "window_dir": str(window_dir.resolve()),
        "teacher_ckpt": str(args.teacher_ckpt.resolve()),
        "hyperparams": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (ckpt_dir / "train_config_student.json").write_text(json.dumps(train_config, indent=2, default=str), encoding="utf-8")

    if test_loader is not None and best_path.is_file():
        ckpt = _torch_load_trusted(best_path, device)
        student.load_state_dict(ckpt["model_state"])
        test_m = eval_epoch_student(student, test_loader, device, scaler_json)
        summary = {
            "best_epoch": best_epoch,
            "val_rmse_chl": best_val_rmse,
            "val_mae_chl": best_val_mae,
            "test_rmse_chl": test_m["rmse"],
            "test_mae_chl": test_m["mae"],
            "test_n_valid": int(test_m.get("n_valid", 0)),
            "checkpoint": str(best_path.resolve()),
            "teacher_ckpt": str(args.teacher_ckpt.resolve()),
            "stopped_reason": stopped_reason,
        }
        (ckpt_dir / "student_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"Test (best val): RMSE_Chl={test_m['rmse']:.4f}, MAE_Chl={test_m['mae']:.4f} → {ckpt_dir / 'student_eval_summary.json'}"
        )

        if args.save_preds:
            if args.residual:
                # collect future preds from y_end_hat + delta_hat
                @torch.no_grad()
                def collect_future(loader):
                    student.eval()
                    ys=[]; ps=[]; cts=[]; tts=[]; have=None
                    for b in loader:
                        xx=b["x"].to(device); xm=b["x_mask"].to(device)
                        yy=b["y"].to(device); ym=b["y_mask"].to(device)
                        v=ym & torch.isfinite(yy)
                        if v.sum()==0: continue
                        yhat,_,_ = student(xx,xm)
                        ys.append(yy[v].cpu().numpy()); ps.append(yhat[v].cpu().numpy())
                        if have is None:
                            have=("context_end_time_ns" in b) and ("target_time_ns" in b)
                        if have:
                            v_np=v.cpu().numpy().astype(bool)
                            cts.append(np.asarray(b["context_end_time_ns"])[v_np].astype(np.int64))
                            tts.append(np.asarray(b["target_time_ns"])[v_np].astype(np.int64))
                    if not ys:
                        return np.array([],dtype=np.float32), np.array([],dtype=np.float32), None, None
                    y_all=np.concatenate(ys); p_all=np.concatenate(ps)
                    if have: return y_all,p_all,np.concatenate(cts),np.concatenate(tts)
                    return y_all,p_all,None,None
                yv_z, pv_z, yv_ct, yv_tt = collect_future(val_loader)
                yt_z, pt_z, yt_ct, yt_tt = collect_future(test_loader)
            else:
                yv_z, pv_z, yv_ct, yv_tt = collect_valid_preds_student(student, val_loader, device)
                yt_z, pt_z, yt_ct, yt_tt = collect_valid_preds_student(student, test_loader, device)

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

