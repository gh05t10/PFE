"""
remove_biofouling_b7.py
-----------------------
Reads BPBuoyData_2014_Cleaned.csv, finds all rows where the chlorophyll
quality flag is 'B7' (biofouling), replaces the corresponding chlorophyll
values with NaN, and saves the result as BPBuoyData_2014_B7Removed.csv.

Also writes a plain-text summary to reports/B7_removal_report.txt.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "FRDR_dataset_1095")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

INPUT_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_Cleaned.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")
REPORT_FILE = os.path.join(REPORT_DIR, "B7_removal_report.txt")

CHLOROPHYLL_COL = "ChlRFUShallow_RFU"
FLAG_COL = "ChlRFUShallow_RFU_Flag"
B7_FLAG = "B7"


def remove_b7_biofouling(input_path: str = INPUT_FILE,
                          output_path: str = OUTPUT_FILE,
                          report_path: str = REPORT_FILE) -> pd.DataFrame:
    """
    Load the 2014 CSV, replace B7-flagged chlorophyll values with NaN,
    and save the cleaned dataset.

    Returns the cleaned DataFrame.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=["DateTime"])

    b7_mask = df[FLAG_COL] == B7_FLAG
    n_b7 = b7_mask.sum()
    b7_dates = df.loc[b7_mask, "DateTime"]

    original_values = df.loc[b7_mask, CHLOROPHYLL_COL].copy()

    df.loc[b7_mask, CHLOROPHYLL_COL] = float("nan")

    df.to_csv(output_path, index=False)

    report_lines = [
        "=" * 60,
        "B7 Biofouling Removal Report",
        "=" * 60,
        f"Input file  : {input_path}",
        f"Output file : {output_path}",
        "",
        f"Total rows in dataset        : {len(df)}",
        f"Rows with B7 flag            : {n_b7}",
        f"B7 period start              : {b7_dates.min()}",
        f"B7 period end                : {b7_dates.max()}",
        "",
        "Statistics of removed values:",
        f"  Mean   : {original_values.mean():.4f} RFU",
        f"  Std    : {original_values.std():.4f} RFU",
        f"  Min    : {original_values.min():.4f} RFU",
        f"  Max    : {original_values.max():.4f} RFU",
        "",
        "Normal chlorophyll range: 2-3 RFU",
        "Action: All B7-flagged chlorophyll values replaced with NaN.",
        "=" * 60,
    ]
    report_text = "\n".join(report_lines)
    with open(report_path, "w") as fh:
        fh.write(report_text)

    print(report_text)
    print(f"\nCleaned file saved to: {output_path}")
    print(f"Report saved to: {report_path}")

    return df


if __name__ == "__main__":
    remove_b7_biofouling()
