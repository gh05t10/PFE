"""
remove_biofouling_b7.py
-----------------------
Reads the 2014 buoy dataset, locates every row where the chlorophyll
quality flag equals 'B7' (biofouling), replaces the corresponding
measurement values with NaN and saves the result as
BPBuoyData_2014_B7Removed.csv.

A plain-text summary report is written to
FRDR_dataset_1095/reports/B7_removal_report.txt.

Usage
-----
Run from the repository root:
    python FRDR_dataset_1095/scripts/remove_biofouling_b7.py
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to the repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_DIR = os.path.join(REPO_ROOT, "FRDR_dataset_1095")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

INPUT_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_Cleaned.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")
REPORT_FILE = os.path.join(REPORTS_DIR, "B7_removal_report.txt")

# Chlorophyll columns that carry B7 flags (value column + flag column pairs)
CHLOROPHYLL_PAIRS = [
    ("ChlRFUShallow_RFU", "ChlRFUShallow_RFU_Flag"),
    ("ChlorophyllRFUDeep_RFU", "ChlorophyllRFUDeep_RFU_Flag"),
]


def remove_b7_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    For every (value_col, flag_col) pair, set value_col = NaN on rows where
    flag_col == 'B7'.

    Returns
    -------
    df_out : pd.DataFrame
        Modified copy of *df*.
    stats  : dict
        Per-column statistics for report generation.
    """
    df_out = df.copy()
    stats = {}

    for value_col, flag_col in CHLOROPHYLL_PAIRS:
        if flag_col not in df_out.columns:
            continue

        mask = df_out[flag_col].astype(str).str.strip().str.upper() == "B7"
        count = int(mask.sum())

        if count == 0:
            stats[value_col] = {"count": 0, "start": None, "end": None}
            continue

        # Record the affected time window before modifying the data
        b7_times = df_out.loc[mask, "DateTime"]
        stats[value_col] = {
            "count": count,
            "start": b7_times.iloc[0],
            "end": b7_times.iloc[-1],
            "pre_mean": float(df_out.loc[mask, value_col].mean()),
            "pre_max": float(df_out.loc[mask, value_col].max()),
        }

        # Replace values with NaN; leave the flag column as-is for traceability
        df_out.loc[mask, value_col] = np.nan

    return df_out, stats


def write_report(stats: dict, df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> None:
    """Write a human-readable text report summarising the B7 removal."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    lines = [
        "=" * 70,
        "  B7 BIOFOULING REMOVAL REPORT — 2014 Chlorophyll Data",
        "=" * 70,
        "",
        f"Input file : {INPUT_FILE}",
        f"Output file: {OUTPUT_FILE}",
        f"Total rows : {len(df_original)}",
        "",
        "-" * 70,
        "  Affected columns",
        "-" * 70,
    ]

    for value_col, info in stats.items():
        lines.append(f"\n  {value_col}")
        if info["count"] == 0:
            lines.append("    No B7 flags found — no changes made.")
        else:
            lines += [
                f"    B7 rows replaced with NaN : {info['count']}",
                f"    First affected timestamp  : {info['start']}",
                f"    Last  affected timestamp  : {info['end']}",
                f"    Mean value (before)       : {info['pre_mean']:.4f} RFU",
                f"    Max  value (before)       : {info['pre_max']:.4f} RFU",
            ]

    total_b7 = sum(v["count"] for v in stats.values())
    lines += [
        "",
        "-" * 70,
        f"  Total B7 rows processed : {total_b7}",
        f"  Remaining valid rows    : {int(df_cleaned['ChlRFUShallow_RFU'].notna().sum())}",
        "-" * 70,
        "",
        "NOTE: B7 flag values were replaced with NaN.",
        "      The original flag columns are preserved for traceability.",
        "      Use chlorophyll_interpolation.py to optionally recover the gap.",
        "",
        "=" * 70,
    ]

    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Report saved → {REPORT_FILE}")


def main() -> None:
    print(f"Reading  → {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, parse_dates=["DateTime"])

    df_cleaned, stats = remove_b7_flags(df)

    os.makedirs(DATA_DIR, exist_ok=True)
    df_cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved    → {OUTPUT_FILE}")

    write_report(stats, df, df_cleaned)

    # Console summary
    for col, info in stats.items():
        if info["count"] > 0:
            print(
                f"[{col}] {info['count']} B7 rows removed "
                f"({info['start']} → {info['end']})"
            )
        else:
            print(f"[{col}] No B7 rows found.")


if __name__ == "__main__":
    main()
