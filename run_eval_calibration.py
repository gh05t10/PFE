#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.calibration_protocol import CalibrationConfig, simulate_daily_calibration
from src.datasets import SlideWindowNPZDataset
from src.models_slide import SlidePatchCrossAttn, SlidePatchCrossAttnConfig, SlideStudentCrossAttn


def _torch_load_trusted(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(
    *,
    ckpt_path: Path,
    device: torch.device,
    kind: Literal["teacher", "student"],
    pred_len: int,
) -> torch.nn.Module:
    ckpt = _torch_load_trusted(ckpt_path, device)
    mcfg_d = ckpt.get("model_config") or ckpt.get("student_config") or {}
    # teacher ckpt may not have pred_len stored (older); infer from head bias if possible
    if "pred_len" not in mcfg_d:
        try:
            mcfg_d["pred_len"] = int(ckpt["model_state"]["head.2.bias"].shape[0])
        except Exception:
            mcfg_d["pred_len"] = pred_len
    cfg = SlidePatchCrossAttnConfig(
        patch_len=int(mcfg_d.get("patch_len", 16)),
        d_model=int(mcfg_d.get("d_model", 64)),
        nhead=int(mcfg_d.get("nhead", 4)),
        encoder_layers=int(mcfg_d.get("encoder_layers", 2)),
        kv_encoder_layers=mcfg_d.get("kv_encoder_layers", None),
        share_q_kv_encoder=bool(mcfg_d.get("share_q_kv_encoder", False)),
        pred_len=int(mcfg_d.get("pred_len", pred_len)),
        dropout=0.1,
    )
    if kind == "teacher":
        m = SlidePatchCrossAttn(cfg).to(device)
    else:
        m = SlideStudentCrossAttn(cfg).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


@torch.no_grad()
def predict_npz(
    *,
    npz_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    kind: Literal["teacher", "student"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    ds = SlideWindowNPZDataset(npz_path)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    have_w = True

    for b in loader:
        y = b.get("y_seq")
        ym = b.get("y_seq_mask")
        if y is None or ym is None:
            raise RuntimeError("NPZ must contain Y_z/Y_mask for calibration eval.")
        y = y.to(device)
        ym = ym.to(device)

        if kind == "teacher":
            x = b["x"].to(device)
            xm = b["x_mask"].to(device)
            x6 = b["x6"].to(device)
            x6m = b["x6_mask"].to(device)
            pred = model(x, xm, x6, x6m)
        else:
            x = b["x"].to(device)
            xm = b["x_mask"].to(device)
            pred = model(x, xm)

        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

        w = b.get("y_seq_w")
        if w is None:
            have_w = False
        else:
            ws.append(w.detach().cpu().numpy())

    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(ps, axis=0)
    if have_w:
        w_all = np.concatenate(ws, axis=0)
    else:
        w_all = None

    m_all = np.load(npz_path, allow_pickle=False)["Y_mask"].astype(bool)
    return y_all, p_all, (w_all if w_all is not None else None), m_all


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate hard-calibration protocol (predict 14d, calib 1d).")
    p.add_argument("--window-dir", type=Path, required=True, help="windowed_L*_H*_P*_S* folder")
    p.add_argument("--ckpt", type=Path, required=True, help="teacher best_slide*.pt or student best_student.pt")
    p.add_argument("--kind", choices=("teacher", "student"), default="teacher")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--pred-len", type=int, default=672, help="14 days @30min = 672")
    p.add_argument("--calib-len", type=int, default=48, help="1 day @30min = 48")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz = args.window_dir / f"{args.split}.npz"
    if not npz.is_file():
        raise SystemExit(f"Missing {npz}")

    model = load_model(ckpt_path=args.ckpt, device=device, kind=args.kind, pred_len=args.pred_len)
    y_true, y_pred, w, m = predict_npz(npz_path=npz, model=model, device=device, kind=args.kind)

    cfg = CalibrationConfig(pred_len=args.pred_len, calib_len=args.calib_len)
    rep = simulate_daily_calibration(y_true=y_true, y_pred=y_pred, mask=m, weights=w, cfg=cfg)

    out: dict[str, Any] = {
        "window_dir": str(args.window_dir.resolve()),
        "split": args.split,
        "kind": args.kind,
        "ckpt": str(args.ckpt.resolve()),
        "cfg": {"pred_len": cfg.pred_len, "calib_len": cfg.calib_len, "update": cfg.update},
        "baseline": rep.baseline,
        "calibrated": rep.calibrated,
        "per_block": rep.per_block[: 14 * 5],  # keep JSON small; still enough for inspection
        "per_block_note": "truncated; recompute or modify script for full list",
    }

    out_path = args.out or (args.window_dir / f"calibration_eval_{args.kind}_{args.split}.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("baseline:", rep.baseline)
    print("calibrated:", rep.calibrated)


if __name__ == "__main__":
    main()

