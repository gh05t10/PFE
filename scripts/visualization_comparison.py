"""
visualization_comparison.py
-----------------------------
Generates six comparison plots and saves them in reports/visualizations/.

5.1  01_overlay_plot.png        – 6 interpolation curves overlaid on raw data
5.2  02_zoom_b7_region.png      – Zoom-in on the B7 gap
5.3  03_error_boxplot.png       – Error distribution box plot
5.4  04_performance_comparison.png – MAE / RMSE / MAPE bar chart
5.5  05_execution_time.png      – Execution time comparison
5.6  06_cv_heatmap.png          – Cross-validation MAE heat map (fold × method)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from chlorophyll_interpolation import ChlorophyllInterpolator, CHLOROPHYLL_COL, get_method_functions

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "FRDR_dataset_1095")
VIZ_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "visualizations")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

B7_REMOVED_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")

METHODS = ["Linear", "Spline", "Polynomial", "kNN", "MissForest", "MICE"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
PALETTE = dict(zip(METHODS, COLORS))


def _apply_all_methods(masked_df: pd.DataFrame) -> dict:
    """Run all six methods and return a dict {name: filled_df}."""
    interp = ChlorophyllInterpolator(masked_df)
    fns = get_method_functions()
    results = {}
    for name, fn in fns.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results[name] = fn(interp)
    return results


def _time_methods(masked_df: pd.DataFrame) -> dict:
    """Return execution time (seconds) for each method."""
    fns = get_method_functions()
    times = {}
    for name, fn in fns.items():
        interp = ChlorophyllInterpolator(masked_df)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(interp)
        times[name] = time.time() - t0
    return times


def generate_all_plots(input_path: str = B7_REMOVED_FILE, viz_dir: str = VIZ_DIR):
    os.makedirs(viz_dir, exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=["DateTime"])

    b7_start_idx = df[df[CHLOROPHYLL_COL].isna()].index.min()
    b7_end_idx = df[df[CHLOROPHYLL_COL].isna()].index.max()

    b7_start_dt = df.loc[b7_start_idx, "DateTime"]
    b7_end_dt = df.loc[b7_end_idx, "DateTime"]

    print(f"B7 gap: {b7_start_dt} → {b7_end_dt}")

    # Apply all methods to the full dataset (fill only the B7 gap)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filled = _apply_all_methods(df)

    # ------------------------------------------------------------------
    # 5.1 Overlay Plot
    # ------------------------------------------------------------------
    print("Generating 01_overlay_plot.png …")
    fig, ax = plt.subplots(figsize=(16, 6))

    # Raw data (non-B7)
    raw = df.dropna(subset=[CHLOROPHYLL_COL])
    ax.plot(raw["DateTime"], raw[CHLOROPHYLL_COL],
            color="black", linewidth=1.2, label="Raw data", zorder=5)

    # B7 shading
    ax.axvspan(b7_start_dt, b7_end_dt, alpha=0.15, color="red", label="B7 gap (biofouling)")

    # Interpolated curves (show only the B7 region)
    b7_mask = df[CHLOROPHYLL_COL].isna()
    for name, fdf in filled.items():
        fdf = fdf.set_index("DateTime")
        b7_values = fdf.loc[b7_mask.values, CHLOROPHYLL_COL]
        ax.plot(b7_values.index, b7_values,
                color=PALETTE[name], linewidth=1.8,
                label=name, alpha=0.85)

    ax.set_title("Chlorophyll Interpolation Comparison – 2014", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("ChlRFUShallow (RFU)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "01_overlay_plot.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5.2 Zoom-in on B7 region
    # ------------------------------------------------------------------
    print("Generating 02_zoom_b7_region.png …")
    margin = pd.Timedelta(days=7)
    fig, ax = plt.subplots(figsize=(14, 5))

    window_mask = (df["DateTime"] >= b7_start_dt - margin) & \
                  (df["DateTime"] <= b7_end_dt + margin)
    raw_zoom = df[window_mask].dropna(subset=[CHLOROPHYLL_COL])
    ax.plot(raw_zoom["DateTime"], raw_zoom[CHLOROPHYLL_COL],
            color="black", linewidth=1.5, label="Raw data", zorder=5)

    ax.axvspan(b7_start_dt, b7_end_dt, alpha=0.15, color="red", label="B7 gap")

    for name, fdf in filled.items():
        fdf_w = fdf[window_mask].copy()
        ax.plot(fdf_w["DateTime"], fdf_w[CHLOROPHYLL_COL],
                color=PALETTE[name], linewidth=2, label=name, alpha=0.85)

    ax.set_title("Zoom-in: B7 Biofouling Region with Interpolations", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("ChlRFUShallow (RFU)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "02_zoom_b7_region.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5.3 Error Box Plot  (based on 80/20 split on pre-B7 data)
    # ------------------------------------------------------------------
    print("Generating 03_error_boxplot.png …")
    pre_b7 = df.loc[:b7_start_idx - 1, :].dropna(subset=[CHLOROPHYLL_COL]).copy()
    pre_b7 = pre_b7.reset_index(drop=True)
    split = int(len(pre_b7) * 0.8)

    masked_pre = pre_b7.copy()
    masked_pre.loc[masked_pre.index[split:], CHLOROPHYLL_COL] = np.nan
    y_true = pre_b7.iloc[split:][CHLOROPHYLL_COL].values

    pre_filled = _apply_all_methods(masked_pre)

    error_data = []
    for name, fdf in pre_filled.items():
        y_pred = fdf.iloc[split:][CHLOROPHYLL_COL].values[: len(y_true)]
        errors = np.abs(y_true - y_pred)
        for e in errors:
            error_data.append({"Method": name, "Absolute Error (RFU)": e})

    error_df = pd.DataFrame(error_data)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=error_df, x="Method", y="Absolute Error (RFU)",
                hue="Method", palette=PALETTE, ax=ax, order=METHODS,
                legend=False)
    ax.set_title("Error Distribution per Interpolation Method (20% hold-out)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "03_error_boxplot.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5.4 Performance Metrics Bar Chart
    # ------------------------------------------------------------------
    print("Generating 04_performance_comparison.png …")
    metrics = []
    for name, fdf in pre_filled.items():
        y_pred = fdf.iloc[split:][CHLOROPHYLL_COL].values[: len(y_true)]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mask_nonzero = y_true != 0
        mape_val = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero])
                                   / y_true[mask_nonzero])) * 100
        metrics.append({"Method": name, "MAE": mae, "RMSE": rmse, "MAPE(%)": mape_val})

    metrics_df = pd.DataFrame(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col in zip(axes, ["MAE", "RMSE", "MAPE(%)"]):
        sorted_df = metrics_df.sort_values(col)
        colors_ordered = [PALETTE[m] for m in sorted_df["Method"]]
        ax.bar(sorted_df["Method"], sorted_df[col], color=colors_ordered)
        ax.set_title(col, fontsize=13)
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Performance Metrics Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "04_performance_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5.5 Execution Time
    # ------------------------------------------------------------------
    print("Generating 05_execution_time.png …")
    times = _time_methods(df)
    time_df = pd.DataFrame(list(times.items()), columns=["Method", "Time(s)"])
    time_df = time_df.sort_values("Time(s)")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_t = [PALETTE[m] for m in time_df["Method"]]
    ax.barh(time_df["Method"], time_df["Time(s)"], color=colors_t)
    ax.set_xlabel("Execution Time (seconds)")
    ax.set_title("Execution Time per Method", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "05_execution_time.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5.6 Cross-Validation Heatmap
    # ------------------------------------------------------------------
    print("Generating 06_cv_heatmap.png …")
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)
    fold_indices = list(kf.split(pre_b7))

    fns = get_method_functions()


    heatmap_data = {name: [] for name in METHODS}
    for fold_num, (_, test_idx) in enumerate(fold_indices):
        masked_cv = pre_b7.copy()
        masked_cv.loc[test_idx, CHLOROPHYLL_COL] = np.nan
        y_true_cv = pre_b7.loc[test_idx, CHLOROPHYLL_COL].values

        for name, fn in fns.items():
            interp = ChlorophyllInterpolator(masked_cv)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fdf = fn(interp)
            y_pred_cv = fdf.loc[test_idx, CHLOROPHYLL_COL].values
            heatmap_data[name].append(mean_absolute_error(y_true_cv, y_pred_cv))

    heatmap_df = pd.DataFrame(heatmap_data,
                               index=[f"Fold {i+1}" for i in range(n_splits)])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax)
    ax.set_title(f"Cross-Validation MAE Heatmap ({n_splits} Folds)", fontsize=13)
    ax.set_xlabel("Method")
    ax.set_ylabel("Fold")
    plt.tight_layout()
    fig.savefig(os.path.join(viz_dir, "06_cv_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"\nAll plots saved to: {viz_dir}")


if __name__ == "__main__":
    generate_all_plots()
