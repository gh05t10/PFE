"""
performance_comparison.py
--------------------------
Evaluates six chlorophyll interpolation methods on the pre-B7 segment of
the 2014 dataset using an 80/20 train-test split.

Strategy
--------
1. Load the B7-removed CSV (NaN in the biofouling gap).
2. Extract the portion BEFORE the B7 gap where data are clean.
3. Mask the last 20 % of that segment as pseudo-missing data.
4. Apply each method to fill the masked values.
5. Compare predictions against the true values.

Metrics: MAE, RMSE, R², MAPE, execution time.
Output:  reports/performance_summary.txt
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make the scripts directory importable when run directly
import sys
sys.path.insert(0, os.path.dirname(__file__))
from chlorophyll_interpolation import ChlorophyllInterpolator, CHLOROPHYLL_COL, get_method_functions

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "FRDR_dataset_1095")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

B7_REMOVED_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")
REPORT_FILE = os.path.join(REPORT_DIR, "performance_summary.txt")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def run_comparison(
    input_path: str = B7_REMOVED_FILE,
    report_path: str = REPORT_FILE,
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=["DateTime"])

    # Use only data BEFORE the B7 gap (no NaN in chlorophyll column)
    pre_b7 = df[df[CHLOROPHYLL_COL].notna()].copy().reset_index(drop=True)

    # Keep only the first continuous clean block (before the B7 gap starts)
    b7_start_idx = df[df[CHLOROPHYLL_COL].isna()].index.min()
    pre_b7 = df.loc[:b7_start_idx - 1, :].dropna(subset=[CHLOROPHYLL_COL]).copy()
    pre_b7 = pre_b7.reset_index(drop=True)

    split = int(len(pre_b7) * 0.8)
    train_df = pre_b7.iloc[:split].copy()
    test_df = pre_b7.iloc[split:].copy()

    # Mask the test portion as NaN to simulate missing data
    masked_df = pre_b7.copy()
    masked_df.loc[masked_df.index[split:], CHLOROPHYLL_COL] = np.nan
    y_true = test_df[CHLOROPHYLL_COL].values

    methods = get_method_functions()

    results = []
    for name, fn in methods.items():
        interp = ChlorophyllInterpolator(masked_df)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filled = fn(interp)
        elapsed = time.time() - t0

        y_pred = filled.loc[split:, CHLOROPHYLL_COL].values[: len(y_true)]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape_val = mape(y_true, y_pred)

        results.append(
            {
                "Method": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE(%)": mape_val,
                "Time(s)": elapsed,
            }
        )
        print(f"  {name:<12} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  "
              f"MAPE={mape_val:.2f}%  t={elapsed:.2f}s")

    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    results_df["Rank"] = range(1, len(results_df) + 1)

    header = f"{'Rank':<5} {'Method':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE(%)':>9} {'Time(s)':>8}"
    separator = "-" * len(header)
    rows = []
    for _, row in results_df.iterrows():
        star = " ⭐" if row["Rank"] == 1 else ""
        rows.append(
            f"{int(row['Rank']):<5} {row['Method']:<12} "
            f"{row['MAE']:>8.4f} {row['RMSE']:>8.4f} {row['R2']:>8.4f} "
            f"{row['MAPE(%)']:>9.2f} {row['Time(s)']:>8.2f}{star}"
        )

    report = "\n".join([
        "=" * 65,
        "Performance Comparison – Chlorophyll Interpolation Methods",
        "=" * 65,
        f"Dataset  : {input_path}",
        f"Strategy : 80/20 train-test split on pre-B7 segment",
        f"Test rows: {len(y_true)}",
        "",
        header,
        separator,
        *rows,
        separator,
        "",
        "Best method (lowest MAE): " + results_df.iloc[0]["Method"],
        "=" * 65,
    ])

    print("\n" + report)
    with open(report_path, "w") as fh:
        fh.write(report)
    print(f"\nReport saved to: {report_path}")

    return results_df


if __name__ == "__main__":
    run_comparison()
