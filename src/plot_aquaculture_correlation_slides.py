from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "refs" / "Aquaculture-report17_1.pdf"
OUTPUT_DIR = BASE_DIR / "refs" / "correlation_slide_plots"

# --- Slide 6: Water parameters (Recaps Correlation – Water parameters)
WATER_LABELS: tuple[str, ...] = (
    "TempShallow_C",
    "pHShallow",
    "ODOShallow_mg/L",
    "SpCondShallow_uS/cm",
    "TurbShallow_NTU",
    "ChlRFUShallow_RFU",
)

PEARSON_WATER = np.array(
    [
        [1.0, 0.39, 0.23, 0.38, -0.094, 0.45],
        [0.39, 1.0, 0.81, -0.45, -0.14, 0.21],
        [0.23, 0.81, 1.0, -0.34, -0.15, 0.11],
        [0.38, -0.45, -0.34, 1.0, 0.24, 0.24],
        [-0.094, -0.14, -0.15, 0.24, 1.0, 0.55],
        [0.45, 0.21, 0.11, 0.24, 0.55, 1.0],
    ]
)

SPEARMAN_WATER = np.array(
    [
        [1.0, 0.34, 0.15, 0.43, 0.19, 0.62],
        [0.34, 1.0, 0.81, -0.36, 0.45, 0.35],
        [0.15, 0.81, 1.0, -0.4, 0.21, 0.13],
        [0.43, -0.36, -0.4, 1.0, 0.085, 0.25],
        [0.19, 0.45, 0.21, 0.085, 1.0, 0.36],
        [0.62, 0.35, 0.13, 0.25, 0.36, 1.0],
    ]
)

# --- Slide 7: Meteorological (Recaps Correlation – Meteorological)
MET_LABELS: tuple[str, ...] = (
    "BarometricPress_kPa",
    "RelativeHum_%",
    "WindSp_km/h",
    "DailyRain_mm",
    "AirTemp_C",
    "ChlRFUShallow_RFU",
)

PEARSON_MET = np.array(
    [
        [1.0, -0.14, -0.28, -0.21, 0.028, 0.34],
        [-0.14, 1.0, 0.026, 0.19, -0.61, 0.057],
        [-0.28, 0.026, 1.0, 0.22, -0.076, -0.098],
        [-0.21, 0.19, 0.22, 1.0, -0.14, 9.2e-5],
        [0.028, -0.61, -0.076, -0.14, 1.0, 0.13],
        [0.34, 0.057, -0.098, 9.2e-5, 0.13, 1.0],
    ]
)

SPEARMAN_MET = np.array(
    [
        [1.0, -0.14, -0.24, -0.35, 0.051, 0.38],
        [-0.14, 1.0, 0.0012, 0.22, -0.61, 0.011],
        [-0.24, 0.0012, 1.0, 0.31, -0.048, -0.17],
        [-0.35, 0.22, 0.31, 1.0, -0.16, -0.12],
        [0.051, -0.61, -0.048, -0.16, 1.0, 0.33],
        [0.38, 0.011, -0.17, -0.12, 0.33, 1.0],
    ]
)


def _annot_labels(mat: np.ndarray) -> np.ndarray:
    """Chuỗi trong ô: 2 chữ số thập phân; giá trị rất nhỏ dùng ký hiệu e."""
    out = np.empty(mat.shape, dtype=object)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = float(mat[i, j])
            if abs(v) < 0.01 and v != 0.0:
                out[i, j] = f"{v:.1e}"
            else:
                out[i, j] = f"{v:.2f}"
    return out


def _plot_pair(
    pearson: np.ndarray,
    spearman: np.ndarray,
    labels: tuple[str, ...],
    title_bar: str,
    outfile: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2), constrained_layout=True)
    fig.suptitle(title_bar, fontsize=14, fontweight="bold", y=1.02)

    cmap = "Blues"
    vmin, vmax = -1.0, 1.0
    cbar_ticks = np.arange(-1.0, 1.01, 0.25)

    for ax, mat, subt in zip(
        axes,
        (pearson, spearman),
        ("Pearson Correlation (Linear)", "Spearman Correlation (Nonlinear)"),
    ):
        df = pd.DataFrame(mat, index=labels, columns=labels)
        sns.heatmap(
            df,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=None,
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


def main() -> None:
    if not PDF_PATH.is_file():
        print(f"Warning: PDF not found at {PDF_PATH} (matrices are still plotted from embedded constants).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _plot_pair(
        PEARSON_WATER,
        SPEARMAN_WATER,
        WATER_LABELS,
        "Recaps (Correlation – Water parameters)",
        OUTPUT_DIR / "page06_water_correlation.png",
    )
    _plot_pair(
        PEARSON_MET,
        SPEARMAN_MET,
        MET_LABELS,
        "Recaps (Correlation – Meteorological)",
        OUTPUT_DIR / "page07_meteorological_correlation.png",
    )

    print(f"Saved:\n  {OUTPUT_DIR / 'page06_water_correlation.png'}\n  {OUTPUT_DIR / 'page07_meteorological_correlation.png'}")


if __name__ == "__main__":
    main()
