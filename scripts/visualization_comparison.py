"""
visualization_comparison.py
-----------------------------
Generate 6 comparison visualizations for all 7 imputation methods.

Plots produced
--------------
1. 01_overlay_plot_all_methods.png  – All imputed series vs original data
2. 02_zoom_b7_region.png            – Zoom into the B7 gap region
3. 03_error_boxplot.png             – Distribution of absolute errors per method
4. 04_performance_metrics_bar.png   – MAE / RMSE / MAPE grouped bar chart
5. 05_speed_vs_accuracy_tradeoff.png– Scatter: Execution Time vs RMSE
6. 06_cv_heatmap.png                – Cross-validation RMSE heatmap (fold × method)

Prerequisites
-------------
Run these scripts first (or set regenerate=True to run them internally):
  python scripts/remove_biofouling_b7.py
  python scripts/performance_comparison.py
  python scripts/cross_validation.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "FRDR_dataset_1095"
REPORTS_DIR = "reports"
VIZ_DIR = "visualizations"
INPUT_ORIGINAL = os.path.join(DATA_DIR, "BPBuoyData_2014_Cleaned.csv")
INPUT_B7REMOVED = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")

CHLRFU_COL = "ChlRFUShallow_RFU"

# Colour palette (consistent across all plots)
METHOD_COLORS = {
    "Linear":          "#2196F3",   # blue
    "Spline":          "#4CAF50",   # green
    "Polynomial":      "#FF9800",   # orange
    "kNN":             "#9C27B0",   # purple
    "LightGBM":        "#00BCD4",   # cyan
    "MissForest":      "#F44336",   # red
    "GaussianProcess": "#795548",   # brown
}
FAST_METHODS = ["Linear", "Spline", "Polynomial", "kNN", "LightGBM"]
ADV_METHODS  = ["MissForest", "GaussianProcess"]
ALL_METHODS  = FAST_METHODS + ADV_METHODS


def _load_imputed(name: str) -> pd.DataFrame | None:
    group = "Fast" if name in FAST_METHODS else "Advanced"
    path = os.path.join(DATA_DIR, f"BPBuoyData_2014_{group}_{name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["DateTime"])


def _b7_period(df_orig: pd.DataFrame):
    """Return (start, end) timestamps of the B7 period."""
    b7 = df_orig[df_orig["ChlRFUShallow_RFU_Flag"] == "B7"]["DateTime"]
    if b7.empty:
        return None, None
    return b7.min(), b7.max()


# ============================================================
# Plot 1: Overlay – all methods on full time series
# ============================================================

def plot_overlay(df_orig: pd.DataFrame, imputed: dict[str, pd.DataFrame], out_dir: str):
    fig, ax = plt.subplots(figsize=(16, 5))

    b7_start, b7_end = _b7_period(df_orig)

    # Original (non-B7) data
    non_b7 = df_orig[df_orig["ChlRFUShallow_RFU_Flag"] != "B7"].copy()
    ax.plot(
        df_orig["DateTime"], df_orig[CHLRFU_COL],
        color="black", linewidth=0.6, alpha=0.4, label="Original (all)", zorder=1,
    )

    # Imputed series
    for name, df_imp in imputed.items():
        group = "FAST" if name in FAST_METHODS else "ADV"
        label = f"{name} [{group}]"
        ax.plot(
            df_imp["DateTime"], df_imp[CHLRFU_COL],
            color=METHOD_COLORS[name], linewidth=1.0, alpha=0.7, label=label, zorder=2,
        )

    # B7 region shading
    if b7_start and b7_end:
        ax.axvspan(b7_start, b7_end, color="red", alpha=0.12, label="B7 gap region")

    ax.set_title("Chlorophyll RFU 2014 – All Imputation Methods Overlay", fontsize=13)
    ax.set_xlabel("DateTime")
    ax.set_ylabel("ChlRFUShallow_RFU")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    plt.tight_layout()
    path = os.path.join(out_dir, "01_overlay_plot_all_methods.png")
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Plot 2: Zoom into B7 region
# ============================================================

def plot_zoom_b7(df_orig: pd.DataFrame, imputed: dict[str, pd.DataFrame], out_dir: str):
    b7_start, b7_end = _b7_period(df_orig)
    if b7_start is None:
        print("No B7 period detected, skipping zoom plot.")
        return

    margin = pd.Timedelta(days=3)
    zoom_start = b7_start - margin
    zoom_end = b7_end + margin

    fig, ax = plt.subplots(figsize=(14, 5))

    # Original
    orig_zoom = df_orig[(df_orig["DateTime"] >= zoom_start) & (df_orig["DateTime"] <= zoom_end)]
    ax.plot(orig_zoom["DateTime"], orig_zoom[CHLRFU_COL],
            color="black", linewidth=1.2, alpha=0.5, label="Original", zorder=1)

    for name, df_imp in imputed.items():
        zoom = df_imp[(df_imp["DateTime"] >= zoom_start) & (df_imp["DateTime"] <= zoom_end)]
        ax.plot(zoom["DateTime"], zoom[CHLRFU_COL],
                color=METHOD_COLORS[name], linewidth=1.4, alpha=0.85, label=name, zorder=2)

    ax.axvspan(b7_start, b7_end, color="red", alpha=0.12, label="B7 gap")
    ax.set_title(f"Zoom: B7 Gap Region  ({b7_start.date()} – {b7_end.date()})", fontsize=12)
    ax.set_xlabel("DateTime")
    ax.set_ylabel("ChlRFUShallow_RFU")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(out_dir, "02_zoom_b7_region.png")
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Plot 3: Error box plot (requires performance_comparison data)
# ============================================================

def plot_error_boxplot(df_orig: pd.DataFrame, imputed: dict[str, pd.DataFrame], out_dir: str):
    """Compute absolute errors in the B7 region using linear as proxy ground-truth
    (since we don't have true values there); or use masked clean-data approach."""
    # Use the performance_comparison CSV if available
    perf_csv = os.path.join(REPORTS_DIR, "performance_comparison.csv")
    if not os.path.exists(perf_csv):
        print("performance_comparison.csv not found, skipping boxplot.")
        return

    perf_df = pd.read_csv(perf_csv)

    # Build error arrays by masking a sample of valid data
    valid_mask = df_orig[CHLRFU_COL].notna() & (df_orig["ChlRFUShallow_RFU_Flag"] != "B7")
    valid_idx = df_orig.index[valid_mask].tolist()

    rng = np.random.default_rng(42)
    n_sample = min(500, len(valid_idx))
    sample_idx = rng.choice(valid_idx, size=n_sample, replace=False)
    y_true = df_orig.loc[sample_idx, CHLRFU_COL].values

    errors_data = []
    labels = []
    for name, df_imp in imputed.items():
        y_pred = np.clip(df_imp.loc[sample_idx, CHLRFU_COL].values, 0, None)
        abs_err = np.abs(y_true - y_pred)
        errors_data.append(abs_err)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(12, 5))
    bplot = ax.boxplot(errors_data, labels=labels, patch_artist=True, showfliers=False)
    colors = [METHOD_COLORS[n] for n in labels]
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Absolute Error Distribution per Imputation Method", fontsize=12)
    ax.set_xlabel("Method")
    ax.set_ylabel("Absolute Error (RFU)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = os.path.join(out_dir, "03_error_boxplot.png")
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Plot 4: Performance metrics bar chart
# ============================================================

def plot_performance_bar(out_dir: str):
    csv_path = os.path.join(REPORTS_DIR, "performance_comparison.csv")
    if not os.path.exists(csv_path):
        print("performance_comparison.csv not found, skipping bar chart.")
        return

    df = pd.read_csv(csv_path)
    metrics = ["MAE", "RMSE", "MAPE"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, metrics):
        colors = [
            "#F44336" if g == "ADVANCED" else "#2196F3"
            for g in df["Group"]
        ]
        bars = ax.bar(df["Method"], df[metric], color=colors, alpha=0.8, edgecolor="white")
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("Method")
        ax.set_ylabel(metric)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # Legend for group colours
    fast_patch = mpatches.Patch(color="#2196F3", alpha=0.8, label="FAST")
    adv_patch = mpatches.Patch(color="#F44336", alpha=0.8, label="ADVANCED")
    fig.legend(handles=[fast_patch, adv_patch], loc="upper right", fontsize=9)

    fig.suptitle("Performance Metrics – FAST vs ADVANCED Methods", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "04_performance_metrics_bar.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Plot 5: Speed vs Accuracy trade-off scatter
# ============================================================

def plot_speed_vs_accuracy(out_dir: str):
    csv_path = os.path.join(REPORTS_DIR, "performance_comparison.csv")
    if not os.path.exists(csv_path):
        print("performance_comparison.csv not found, skipping speed-accuracy plot.")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in df.iterrows():
        color = "#F44336" if row["Group"] == "ADVANCED" else "#2196F3"
        marker = "^" if row["Group"] == "ADVANCED" else "o"
        ax.scatter(row["Time_s"], row["RMSE"], color=color, marker=marker, s=120, zorder=3,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(row["Method"], (row["Time_s"], row["RMSE"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Pareto frontier (lower-left is better)
    sorted_df = df.sort_values("Time_s")
    pareto_rmse = float("inf")
    pareto_x, pareto_y = [], []
    for _, row in sorted_df.iterrows():
        if row["RMSE"] < pareto_rmse:
            pareto_rmse = row["RMSE"]
            pareto_x.append(row["Time_s"])
            pareto_y.append(row["RMSE"])
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, "k--", linewidth=1.2, alpha=0.5, label="Pareto frontier")

    fast_patch = mpatches.Patch(color="#2196F3", label="FAST methods")
    adv_patch  = mpatches.Patch(color="#F44336", label="ADVANCED methods")
    ax.legend(handles=[fast_patch, adv_patch, plt.Line2D([], [], color="k", linestyle="--",
              label="Pareto frontier")], fontsize=9)

    ax.set_xscale("log")
    ax.set_title("Speed vs Accuracy Trade-off (lower-left = better)", fontsize=12)
    ax.set_xlabel("Execution Time (seconds, log scale)")
    ax.set_ylabel("RMSE (RFU)")
    plt.tight_layout()
    path = os.path.join(out_dir, "05_speed_vs_accuracy_tradeoff.png")
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Plot 6: Cross-validation heatmap
# ============================================================

def plot_cv_heatmap(out_dir: str):
    cv_path = os.path.join(REPORTS_DIR, "cross_validation_results.csv")
    if not os.path.exists(cv_path):
        print("cross_validation_results.csv not found, skipping CV heatmap.")
        return

    cv_df = pd.read_csv(cv_path)
    pivot = cv_df.pivot_table(index="Method", columns="Fold", values="RMSE", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot,
        annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "RMSE (RFU)"},
    )
    ax.set_title("Cross-Validation RMSE per Method × Fold", fontsize=12)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Method")
    plt.tight_layout()
    path = os.path.join(out_dir, "06_cv_heatmap.png")
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load original and imputed dataframes
    if not os.path.exists(INPUT_ORIGINAL):
        print(f"Missing: {INPUT_ORIGINAL}")
        sys.exit(1)

    df_orig = pd.read_csv(INPUT_ORIGINAL, parse_dates=["DateTime"])

    imputed = {}
    for name in ALL_METHODS:
        df_imp = _load_imputed(name)
        if df_imp is not None:
            imputed[name] = df_imp
        else:
            print(f"Warning: imputed CSV not found for {name}, skipping.")

    if not imputed:
        print("No imputed CSVs found. Run the interpolation scripts first.")
        sys.exit(1)

    print(f"Loaded {len(imputed)} imputed datasets: {list(imputed.keys())}\n")

    plot_overlay(df_orig, imputed, VIZ_DIR)
    plot_zoom_b7(df_orig, imputed, VIZ_DIR)
    plot_error_boxplot(df_orig, imputed, VIZ_DIR)
    plot_performance_bar(VIZ_DIR)
    plot_speed_vs_accuracy(VIZ_DIR)
    plot_cv_heatmap(VIZ_DIR)

    print(f"\nAll visualizations saved to '{VIZ_DIR}/'")


if __name__ == "__main__":
    main()
