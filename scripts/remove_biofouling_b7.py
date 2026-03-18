"""
remove_biofouling_b7.py
-----------------------
Read the 2014 buoy CSV, replace every ChlRFUShallow_RFU value whose
quality flag is 'B7' (biofouling) with NaN, and save the result.

Output: FRDR_dataset_1095/BPBuoyData_2014_B7Removed.csv
Report: reports/B7_removal_report.txt
"""

import os
import pandas as pd

DATA_DIR = "FRDR_dataset_1095"
INPUT_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_Cleaned.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")
REPORT_FILE = os.path.join("reports", "B7_removal_report.txt")

CHLRFU_COL = "ChlRFUShallow_RFU"
CHLRFU_FLAG_COL = "ChlRFUShallow_RFU_Flag"


def remove_b7_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Replace ChlRFUShallow_RFU values flagged as 'B7' with NaN.

    Returns the modified DataFrame and a summary statistics dict.
    """
    df = df.copy()

    b7_mask = df[CHLRFU_FLAG_COL] == "B7"
    b7_rows = int(b7_mask.sum())

    b7_subset = df.loc[b7_mask, CHLRFU_COL]
    stats = {
        "total_rows": len(df),
        "b7_rows": b7_rows,
        "b7_pct": round(b7_rows / len(df) * 100, 2),
        "b7_start": str(df.loc[b7_mask, "DateTime"].min()) if b7_rows else "N/A",
        "b7_end": str(df.loc[b7_mask, "DateTime"].max()) if b7_rows else "N/A",
        "chl_mean_before": round(df[CHLRFU_COL].mean(), 4),
        "chl_min_before": round(df[CHLRFU_COL].min(), 4),
        "chl_max_before": round(df[CHLRFU_COL].max(), 4),
        "b7_chl_mean": round(b7_subset.mean(), 4) if b7_rows else "N/A",
        "b7_chl_min": round(b7_subset.min(), 4) if b7_rows else "N/A",
        "b7_chl_max": round(b7_subset.max(), 4) if b7_rows else "N/A",
    }

    # Replace B7 chlorophyll values with NaN
    df.loc[b7_mask, CHLRFU_COL] = float("nan")

    stats["nan_after"] = int(df[CHLRFU_COL].isna().sum())
    stats["chl_mean_after"] = round(df[CHLRFU_COL].mean(), 4)

    return df, stats


def write_report(stats: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("B7 Biofouling Removal Report\n")
        fh.write("=" * 40 + "\n\n")
        fh.write(f"Input file      : {INPUT_FILE}\n")
        fh.write(f"Output file     : {OUTPUT_FILE}\n\n")
        fh.write("Dataset overview\n")
        fh.write("-" * 40 + "\n")
        fh.write(f"  Total rows              : {stats['total_rows']}\n")
        fh.write(f"  B7-flagged rows         : {stats['b7_rows']} ({stats['b7_pct']}%)\n")
        fh.write(f"  B7 period (start)       : {stats['b7_start']}\n")
        fh.write(f"  B7 period (end)         : {stats['b7_end']}\n\n")
        fh.write("ChlRFUShallow_RFU statistics\n")
        fh.write("-" * 40 + "\n")
        fh.write(f"  Mean  (before removal)  : {stats['chl_mean_before']}\n")
        fh.write(f"  Min   (before removal)  : {stats['chl_min_before']}\n")
        fh.write(f"  Max   (before removal)  : {stats['chl_max_before']}\n")
        fh.write(f"  Mean  (B7 values only)  : {stats['b7_chl_mean']}\n")
        fh.write(f"  Min   (B7 values only)  : {stats['b7_chl_min']}\n")
        fh.write(f"  Max   (B7 values only)  : {stats['b7_chl_max']}\n")
        fh.write(f"  NaN count (after)       : {stats['nan_after']}\n")
        fh.write(f"  Mean  (after removal)   : {stats['chl_mean_after']}\n\n")
        fh.write("Result: B7 values replaced with NaN.\n")
        fh.write("        Flag column preserved for reference.\n")


def main():
    print(f"Reading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["DateTime"])

    df_clean, stats = remove_b7_flags(df)

    os.makedirs(DATA_DIR, exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned data  -> {OUTPUT_FILE}")

    write_report(stats, REPORT_FILE)
    print(f"Saved removal report -> {REPORT_FILE}")

    print(f"\nSummary: {stats['b7_rows']} B7-flagged rows ({stats['b7_pct']}%) "
          f"from {stats['b7_start']} to {stats['b7_end']}")
    print(f"ChlRFUShallow_RFU mean: {stats['chl_mean_before']} -> {stats['chl_mean_after']}")


if __name__ == "__main__":
    main()
