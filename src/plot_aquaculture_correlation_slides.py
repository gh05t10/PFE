"""
Ma trận tương quan Pearson / Spearman theo slide 6–7 (Aquaculture report),
tính **trực tiếp** từ buoy FRDR (BPBuoyData_*_Cleaned.csv), không copy số từ slide.

QC: bỏ quan giá trị khi có cờ B7 / C / M (giống ``chl_shallow_pipeline``).

Chạy:  python -m src.plot_aquaculture_correlation_slides

Ảnh PNG: ``refs/correlation_slide_plots/``
Tùy chọn: lưu CSV ma trận (``--save-csv``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .chl_shallow_pipeline import EXCLUDE_FLAGS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "FRDR_dataset_1095"
OUTPUT_DIR = BASE_DIR / "refs" / "correlation_slide_plots"

# Cột trong CSV (slide 6: nước — tên độ trong là TurbShallow_NTU+ trong FRDR)
WATER_COLS: tuple[str, ...] = (
    "TempShallow_C",
    "pHShallow",
    "ODOShallow_mg/L",
    "SpCondShallow_uS/cm",
    "TurbShallow_NTU+",
    "ChlRFUShallow_RFU",
)

# Slide 7: khí tượng + Chl nông
MET_COLS: tuple[str, ...] = (
    "BarometricPress_kPa",
    "RelativeHum_%",
    "WindSp_km/h",
    "DailyRain_mm",
    "AirTemp_C",
    "ChlRFUShallow_RFU",
)


def _display_label(col: str) -> str:
    """Nhãn hiển thị giống slide (TurbShallow_NTU thay vì TurbShallow_NTU+)."""
    if col == "TurbShallow_NTU+":
        return "TurbShallow_NTU"
    return col


def load_frdr_masked(data_dir: Path, cols: tuple[str, ...]) -> pd.DataFrame:
    """Nối mọi năm Cleaned; mỗi cột chỉ giữ số khi không bị cờ B7/C/M."""
    paths = sorted(data_dir.glob("BPBuoyData_*_Cleaned.csv"))
    if not paths:
        raise FileNotFoundError(f"No BPBuoyData_*_Cleaned.csv under {data_dir}")

    frames: list[pd.DataFrame] = []
    for p in paths:
        header = pd.read_csv(p, nrows=0).columns.tolist()
        flag_cols = [f"{c}_Flag" for c in cols]
        usecols = ["DateTime"] + [c for c in cols if c in header]
        usecols += [f for f in flag_cols if f in header]
        missing = [c for c in cols if c not in header]
        if missing:
            raise KeyError(f"{p.name} missing columns: {missing}")

        df = pd.read_csv(p, usecols=usecols, low_memory=False)
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df = df.dropna(subset=["DateTime"])

        block = pd.DataFrame({"DateTime": df["DateTime"]})
        for col in cols:
            v = pd.to_numeric(df[col], errors="coerce")
            fc = f"{col}_Flag"
            if fc in df.columns:
                bad = (
                    df[fc]
                    .map(lambda x: str(x).strip() if pd.notna(x) else None)
                    .isin(EXCLUDE_FLAGS)
                )
            else:
                bad = pd.Series(False, index=df.index)
            block[col] = np.where(~bad.to_numpy() & v.notna().to_numpy(), v, np.nan)

        block["source_file"] = p.name
        frames.append(block)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("DateTime").drop_duplicates(subset=["DateTime"], keep="first")
    return out


def correlation_matrices(df: pd.DataFrame, cols: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson và Spearman (pandas: pairwise bỏ NaN theo từng cặp)."""
    sub = df[list(cols)]
    pearson = sub.corr(method="pearson", min_periods=1)
    spearman = sub.corr(method="spearman", min_periods=1)
    return pearson, spearman


def _annot_labels(mat: np.ndarray) -> np.ndarray:
    out = np.empty(mat.shape, dtype=object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = float(mat[i, j])
            if np.isnan(v):
                out[i, j] = ""
            elif abs(v) < 0.01 and v != 0.0:
                out[i, j] = f"{v:.1e}"
            else:
                out[i, j] = f"{v:.2f}"
    return out


def _plot_pair(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    labels: tuple[str, ...],
    title_bar: str,
    outfile: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), constrained_layout=True)
    fig.suptitle(title_bar, fontsize=14, fontweight="bold", y=1.02)

    cmap = "Blues"
    vmin, vmax = -1.0, 1.0
    cbar_ticks = np.arange(-1.0, 1.01, 0.25)

    for ax, mat_df, subt in zip(
        axes,
        (pearson, spearman),
        ("Pearson Correlation (Linear)", "Spearman Correlation (Nonlinear)"),
    ):
        mat = mat_df.to_numpy(dtype=float)
        plot_df = mat_df.copy()
        plot_df.index = labels
        plot_df.columns = labels

        sns.heatmap(
            plot_df,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            annot=_annot_labels(mat),
            fmt="",
            annot_kws={"size": 8},
            cbar_kws={"ticks": cbar_ticks, "label": ""},
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(subt, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.savefig(outfile, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _print_summary(name: str, df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    sub = df[list(cols)]
    n_rows = len(sub)
    n_complete = sub.dropna().shape[0]
    print(f"[{name}] timestamps: {n_rows:,} | rows with all 6 vars non-NaN: {n_complete:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation heatmaps from FRDR buoy data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Folder with BPBuoyData_*_Cleaned.csv (default: FRDR_dataset_1095)",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help=f"Also write pearson/spearman CSVs next to PNGs under {OUTPUT_DIR}",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Water + met share Chl; load union of columns once for efficiency
    all_cols = tuple(dict.fromkeys(WATER_COLS + MET_COLS))
    raw = load_frdr_masked(args.data_dir, all_cols)

    water_labels = tuple(_display_label(c) for c in WATER_COLS)
    met_labels = tuple(_display_label(c) for c in MET_COLS)

    _print_summary("water", raw, WATER_COLS)
    p_w, s_w = correlation_matrices(raw, WATER_COLS)
    p_w.index = p_w.columns = water_labels
    s_w.index = s_w.columns = water_labels

    _print_summary("met", raw, MET_COLS)
    p_m, s_m = correlation_matrices(raw, MET_COLS)
    p_m.index = p_m.columns = met_labels
    s_m.index = s_m.columns = met_labels

    _plot_pair(
        p_w,
        s_w,
        water_labels,
        "Recaps (Correlation – Water parameters) [FRDR data]",
        OUTPUT_DIR / "page06_water_correlation.png",
    )
    _plot_pair(
        p_m,
        s_m,
        met_labels,
        "Recaps (Correlation – Meteorological) [FRDR data]",
        OUTPUT_DIR / "page07_meteorological_correlation.png",
    )

    if args.save_csv:
        p_w.to_csv(OUTPUT_DIR / "page06_pearson.csv")
        s_w.to_csv(OUTPUT_DIR / "page06_spearman.csv")
        p_m.to_csv(OUTPUT_DIR / "page07_pearson.csv")
        s_m.to_csv(OUTPUT_DIR / "page07_spearman.csv")
        print(f"CSV matrices saved under {OUTPUT_DIR}")

    print(
        "Saved:\n"
        f"  {OUTPUT_DIR / 'page06_water_correlation.png'}\n"
        f"  {OUTPUT_DIR / 'page07_meteorological_correlation.png'}"
    )


if __name__ == "__main__":
    main()
