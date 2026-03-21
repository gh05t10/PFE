"""
demo_interpolation.py
---------------------
Demonstrates all six interpolation strategies provided by
chlorophyll_interpolation.py on the 2014 chlorophyll dataset after
B7 biofouling values have been replaced with NaN.

Prerequisites
-------------
Run remove_biofouling_b7.py first to generate BPBuoyData_2014_B7Removed.csv.

Usage
-----
    python FRDR_dataset_1095/scripts/demo_interpolation.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Allow importing the sibling module regardless of working directory
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from chlorophyll_interpolation import ChlorophyllInterpolator  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(SCRIPTS_DIR, "..")
B7_REMOVED_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")
ORIGINAL_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_Cleaned.csv")
OUTPUT_PLOT = os.path.join(DATA_DIR, "reports", "interpolation_comparison.png")

# Columns
VALUE_COL = "ChlRFUShallow_RFU"
FLAG_COL = "ChlRFUShallow_RFU_Flag"


def load_data():
    """Load the B7-removed dataset (NaN in B7 period) and the original."""
    if not os.path.exists(B7_REMOVED_FILE):
        raise FileNotFoundError(
            f"Cannot find {B7_REMOVED_FILE}.\n"
            "Please run remove_biofouling_b7.py first."
        )
    df_nan = pd.read_csv(B7_REMOVED_FILE, parse_dates=["DateTime"])
    df_orig = pd.read_csv(ORIGINAL_FILE, parse_dates=["DateTime"])
    return df_nan, df_orig


def zoom_window(df: pd.DataFrame, start: str = "2014-08-01", end: str = "2014-09-30"):
    """Return a date-range slice for plotting."""
    mask = (df["DateTime"] >= start) & (df["DateTime"] <= end)
    return df[mask]


def run_demo():
    print("Loading data …")
    df_nan, df_orig = load_data()

    interp = ChlorophyllInterpolator(df_nan, value_col=VALUE_COL)

    print("Running interpolation methods …")
    results = {
        "1. Linear": interp.linear_interpolate(),
        "2. Spline (order=3)": interp.spline_interpolate(order=3),
        "3. Polynomial (deg=3)": interp.polynomial_interpolate(degree=3),
        "4. Seasonal Decomp. (REC.)": interp.seasonal_decompose_interpolate(period=96),
        "5. Forward Fill": interp.fill_interpolate(method="ffill"),
        "6. Backward Fill": interp.fill_interpolate(method="bfill"),
    }

    # Console summary
    nan_count = int(df_nan[VALUE_COL].isna().sum())
    print(f"\n{VALUE_COL} — NaN count: {nan_count}")
    print(f"{'Method':<35} {'Remaining NaN':>13} {'Mean (gap period)':>18}")
    print("-" * 70)

    # Identify gap period
    b7_mask_orig = df_orig[FLAG_COL].astype(str).str.strip().str.upper() == "B7"
    gap_times = df_orig.loc[b7_mask_orig, "DateTime"]
    gap_start, gap_end = gap_times.iloc[0], gap_times.iloc[-1]

    for name, df_res in results.items():
        rem_nan = int(df_res[VALUE_COL].isna().sum())
        gap_mask = (df_res["DateTime"] >= gap_start) & (df_res["DateTime"] <= gap_end)
        gap_mean = df_res.loc[gap_mask, VALUE_COL].mean()
        print(f"  {name:<33} {rem_nan:>13} {gap_mean:>18.4f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    print("\nGenerating comparison plot …")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = ["steelblue", "darkorange", "green", "purple", "brown", "teal"]

    for ax, (name, df_res), color in zip(axes, results.items(), colors):
        z_res = zoom_window(df_res)
        z_orig = zoom_window(df_orig)
        z_nan = zoom_window(df_nan)

        # Original (biofouling visible) in light red
        ax.plot(
            z_orig["DateTime"], z_orig[VALUE_COL],
            color="lightcoral", linewidth=0.8, alpha=0.6, label="Original (with B7)"
        )
        # Interpolated result
        ax.plot(
            z_res["DateTime"], z_res[VALUE_COL],
            color=color, linewidth=1.2, label=name
        )
        # Mark the gap period
        ax.axvspan(gap_start, gap_end, color="sandybrown", alpha=0.2, label="B7 gap")

        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylabel("Chl-a (RFU)", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Chlorophyll Interpolation Methods — 2014 B7 Gap (Aug 20 – Aug 26)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {OUTPUT_PLOT}")
    plt.close()


if __name__ == "__main__":
    run_demo()
