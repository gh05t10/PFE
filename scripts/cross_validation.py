"""
cross_validation.py
--------------------
K-Fold cross-validation (default k=5) for all six chlorophyll interpolation
methods on the pre-B7 clean segment of the 2014 dataset.

For each fold:
  - mask one contiguous block as pseudo-missing data
  - apply each interpolation method
  - compute MAE, RMSE, R² against the true values

Output
------
- Console table: Method  MAE±σ  RMSE±σ  R²±σ  Time(s)
- reports/cross_validation_results.csv
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

import sys
sys.path.insert(0, os.path.dirname(__file__))
from chlorophyll_interpolation import ChlorophyllInterpolator, CHLOROPHYLL_COL, get_method_functions

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "FRDR_dataset_1095")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

B7_REMOVED_FILE = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")
CV_RESULTS_FILE = os.path.join(REPORT_DIR, "cross_validation_results.csv")


def run_cross_validation(
    input_path: str = B7_REMOVED_FILE,
    output_csv: str = CV_RESULTS_FILE,
    n_splits: int = 5,
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=["DateTime"])

    # Only the clean pre-B7 segment
    b7_start_idx = df[df[CHLOROPHYLL_COL].isna()].index.min()
    pre_b7 = df.loc[:b7_start_idx - 1, :].dropna(subset=[CHLOROPHYLL_COL]).copy()
    pre_b7 = pre_b7.reset_index(drop=True)

    print(f"Clean pre-B7 segment: {len(pre_b7)} rows")
    print(f"Running {n_splits}-Fold Cross-Validation ...\n")

    methods = get_method_functions()


    kf = KFold(n_splits=n_splits, shuffle=False)
    fold_indices = list(kf.split(pre_b7))

    all_rows = []

    for name, fn in methods.items():
        fold_mae, fold_rmse, fold_r2, fold_times = [], [], [], []

        for fold_num, (train_idx, test_idx) in enumerate(fold_indices):
            masked_df = pre_b7.copy()
            masked_df.loc[test_idx, CHLOROPHYLL_COL] = np.nan
            y_true = pre_b7.loc[test_idx, CHLOROPHYLL_COL].values

            interp = ChlorophyllInterpolator(masked_df)
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filled = fn(interp)
            elapsed = time.time() - t0

            y_pred = filled.loc[test_idx, CHLOROPHYLL_COL].values

            fold_mae.append(mean_absolute_error(y_true, y_pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y_true, y_pred)))
            fold_r2.append(r2_score(y_true, y_pred))
            fold_times.append(elapsed)

            all_rows.append({
                "Method": name,
                "Fold": fold_num + 1,
                "MAE": fold_mae[-1],
                "RMSE": fold_rmse[-1],
                "R2": fold_r2[-1],
                "Time(s)": elapsed,
            })

        mae_mean, mae_std = np.mean(fold_mae), np.std(fold_mae)
        rmse_mean, rmse_std = np.mean(fold_rmse), np.std(fold_rmse)
        r2_mean, r2_std = np.mean(fold_r2), np.std(fold_r2)
        time_mean = np.mean(fold_times)

        print(
            f"  {name:<12} "
            f"MAE={mae_mean:.4f}±{mae_std:.4f}  "
            f"RMSE={rmse_mean:.4f}±{rmse_std:.4f}  "
            f"R²={r2_mean:.4f}±{r2_std:.4f}  "
            f"t={time_mean:.2f}s"
        )

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to: {output_csv}")

    # Summary table
    summary = (
        results_df.groupby("Method")
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            Time_mean=("Time(s)", "mean"),
        )
        .sort_values("MAE_mean")
        .reset_index()
    )

    header = (
        f"{'Method':<12} {'MAE±σ':>14} {'RMSE±σ':>14} {'R²±σ':>14} {'Time(s)':>8}"
    )
    sep = "─" * len(header)
    print(f"\nCross-Validation Results (k-fold={n_splits}):")
    print(sep)
    print(header)
    print(sep)
    for _, row in summary.iterrows():
        star = " ⭐" if row.name == 0 else ""
        print(
            f"{row['Method']:<12} "
            f"{row['MAE_mean']:.2f}±{row['MAE_std']:.2f}  "
            f"{row['RMSE_mean']:.2f}±{row['RMSE_std']:.2f}  "
            f"{row['R2_mean']:.2f}±{row['R2_std']:.2f}  "
            f"{row['Time_mean']:>8.2f}{star}"
        )
    print(sep)

    return results_df


if __name__ == "__main__":
    run_cross_validation()
