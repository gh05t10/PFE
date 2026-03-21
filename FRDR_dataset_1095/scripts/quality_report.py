"""
quality_report.py
-----------------
Generates a detailed quality report comparing the 2014 chlorophyll dataset
before and after B7 biofouling removal.

Output
------
Prints a summary to the console and saves a text report to
FRDR_dataset_1095/reports/B7_removal_report.txt (appends quality stats).

Also optionally produces a before/after plot if matplotlib is available.

Usage
-----
    python FRDR_dataset_1095/scripts/quality_report.py
"""

import os
import sys
import textwrap
import pandas as pd
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPTS_DIR, "..")
REPORTS_DIR = os.path.normpath(os.path.join(DATA_DIR, "reports"))

ORIGINAL_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_Cleaned.csv")
CLEANED_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")
REPORT_FILE = os.path.join(REPORTS_DIR, "B7_removal_report.txt")

VALUE_COL = "ChlRFUShallow_RFU"
FLAG_COL = "ChlRFUShallow_RFU_Flag"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def basic_stats(series: pd.Series, label: str) -> dict:
    clean = series.dropna()
    return {
        "label": label,
        "count": int(len(clean)),
        "nan_count": int(series.isna().sum()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "p5": float(np.percentile(clean, 5)),
        "p95": float(np.percentile(clean, 95)),
    }


def format_stats(s: dict) -> str:
    return (
        f"  {s['label']}\n"
        f"    Valid observations : {s['count']:,}\n"
        f"    NaN / missing      : {s['nan_count']:,}\n"
        f"    Mean               : {s['mean']:.4f} RFU\n"
        f"    Median             : {s['median']:.4f} RFU\n"
        f"    Std dev            : {s['std']:.4f} RFU\n"
        f"    Min                : {s['min']:.4f} RFU\n"
        f"    Max                : {s['max']:.4f} RFU\n"
        f"    5th percentile     : {s['p5']:.4f} RFU\n"
        f"    95th percentile    : {s['p95']:.4f} RFU\n"
    )


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def generate_report() -> None:
    if not os.path.exists(CLEANED_FILE):
        print(
            f"ERROR: {CLEANED_FILE} not found.\n"
            "Run remove_biofouling_b7.py first."
        )
        sys.exit(1)

    df_orig = pd.read_csv(ORIGINAL_FILE, parse_dates=["DateTime"])
    df_clean = pd.read_csv(CLEANED_FILE, parse_dates=["DateTime"])

    b7_mask = df_orig[FLAG_COL].astype(str).str.strip().str.upper() == "B7"
    b7_rows = df_orig[b7_mask]
    b7_values = b7_rows[VALUE_COL]

    stats_orig = basic_stats(df_orig[VALUE_COL], "BEFORE removal (all data, incl. B7)")
    stats_b7 = basic_stats(b7_values, "B7 segment only (biofouling period)")
    stats_clean = basic_stats(df_clean[VALUE_COL], "AFTER  removal (B7 → NaN)")

    # Non-B7 clean statistics for comparison
    non_b7_orig = df_orig.loc[~b7_mask, VALUE_COL]
    stats_non_b7 = basic_stats(non_b7_orig, "Valid data (excluding B7)")

    sep = "=" * 70
    thin = "-" * 70

    report_lines = [
        sep,
        "  QUALITY REPORT — 2014 Chlorophyll Data (ChlRFUShallow_RFU)",
        sep,
        "",
        f"  Original file : {ORIGINAL_FILE}",
        f"  Cleaned file  : {CLEANED_FILE}",
        f"  Total rows    : {len(df_orig):,}",
        "",
        thin,
        "  B7 Biofouling Period",
        thin,
        f"  Number of B7-flagged rows : {b7_mask.sum():,}",
        f"  Start of B7 period        : {b7_rows['DateTime'].iloc[0]}",
        f"  End   of B7 period        : {b7_rows['DateTime'].iloc[-1]}",
        f"  Duration                  : {b7_rows['DateTime'].iloc[-1] - b7_rows['DateTime'].iloc[0]}",
        "",
        thin,
        "  Descriptive Statistics",
        thin,
        "",
        format_stats(stats_orig),
        format_stats(stats_b7),
        format_stats(stats_non_b7),
        format_stats(stats_clean),
        thin,
        "",
        "  Data quality summary",
        thin,
        f"  B7 anomaly: mean {stats_b7['mean']:.2f} RFU vs normal {stats_non_b7['mean']:.2f} RFU "
        f"({stats_b7['mean'] / stats_non_b7['mean']:.1f}× higher)",
        f"  Data coverage after removal: "
        f"{stats_clean['count'] / len(df_orig) * 100:.1f}% valid readings",
        "",
        textwrap.fill(
            "RECOMMENDATION: The B7 gap spans approximately 6 days (Aug 20-26). "
            "Seasonal decomposition interpolation is recommended for recovery, as it "
            "preserves the daily cycle and long-term trend while producing smooth, "
            "physically plausible chlorophyll estimates.",
            width=68,
            initial_indent="  ",
            subsequent_indent="  ",
        ),
        "",
        sep,
    ]

    report_text = "\n".join(report_lines)

    # Print to console
    print(report_text)

    # Save to file
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        fh.write(report_text + "\n")
    print(f"\nReport saved → {REPORT_FILE}")

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        plot_path = os.path.join(REPORTS_DIR, "before_after_b7_removal.png")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        start_zoom, end_zoom = "2014-07-01", "2014-10-01"
        z_orig = df_orig[
            (df_orig["DateTime"] >= start_zoom) & (df_orig["DateTime"] <= end_zoom)
        ]
        z_clean = df_clean[
            (df_clean["DateTime"] >= start_zoom) & (df_clean["DateTime"] <= end_zoom)
        ]
        b7_start = b7_rows["DateTime"].iloc[0]
        b7_end = b7_rows["DateTime"].iloc[-1]

        # --- Before ---
        ax1.plot(z_orig["DateTime"], z_orig[VALUE_COL],
                 color="steelblue", linewidth=0.9, label="Chlorophyll (original)")
        ax1.plot(
            df_orig.loc[b7_mask, "DateTime"], df_orig.loc[b7_mask, VALUE_COL],
            color="red", linewidth=1.2, label="B7 biofouling segment"
        )
        ax1.axvspan(b7_start, b7_end, color="sandybrown", alpha=0.25, label="B7 period")
        ax1.set_title("Before B7 Removal", fontweight="bold")
        ax1.set_ylabel("Chl-a (RFU)")
        ax1.legend(fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.set_ylim(bottom=0)

        # --- After ---
        ax2.plot(z_clean["DateTime"], z_clean[VALUE_COL],
                 color="steelblue", linewidth=0.9, label="Chlorophyll (B7 → NaN)")
        ax2.axvspan(b7_start, b7_end, color="lightgrey", alpha=0.5, label="Gap (NaN)")
        ax2.set_title("After B7 Removal (NaN gap)", fontweight="bold")
        ax2.set_ylabel("Chl-a (RFU)")
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.set_ylim(bottom=0)

        for ax in (ax1, ax2):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

        fig.suptitle(
            "2014 Chlorophyll — Before vs After B7 Biofouling Removal",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved  → {plot_path}")

    except ImportError:
        print("matplotlib not available — skipping plot generation.")


if __name__ == "__main__":
    generate_report()
