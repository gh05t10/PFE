"""
performance_comparison.py
--------------------------
Compare all 7 imputation methods using a held-out test approach:

1. Take the clean (non-B7) chlorophyll data.
2. Randomly mask 20 % of it as "artificial missing".
3. Apply every method to recover these values.
4. Compute MAE, RMSE, MAPE, R² and execution time.
5. Rank methods and save a summary report + CSV.

Output
------
reports/performance_summary.txt
reports/performance_comparison.csv
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.chlorophyll_interpolation import ChlorophyllInterpolator
from scripts.chlorophyll_advanced_imputation import AdvancedChlorophyllImputation

DATA_DIR = "FRDR_dataset_1095"
REPORTS_DIR = "reports"
INPUT_B7REMOVED = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")

CHLRFU_COL = "ChlRFUShallow_RFU"
TEST_FRACTION = 0.20
RANDOM_STATE = 42


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAPE": round(mape(y_true, y_pred), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def run_comparison(
    input_csv: str = INPUT_B7REMOVED,
    test_fraction: float = TEST_FRACTION,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run all 7 methods and return a DataFrame with performance metrics."""

    # ------------------------------------------------------------------ #
    # 1. Prepare evaluation dataset                                        #
    # ------------------------------------------------------------------ #
    df_base = pd.read_csv(input_csv, parse_dates=["DateTime"])

    # Work only with rows where chlorophyll is NOT already NaN (i.e. non-B7)
    valid_mask = df_base[CHLRFU_COL].notna()
    valid_idx = df_base.index[valid_mask].tolist()

    rng = np.random.default_rng(random_state)
    n_test = max(1, int(len(valid_idx) * test_fraction))
    test_idx = rng.choice(valid_idx, size=n_test, replace=False)

    # Ground truth values at test positions
    y_true = df_base.loc[test_idx, CHLRFU_COL].values.copy()

    # Create experiment dataframe: mask test positions as NaN
    df_exp = df_base.copy()
    df_exp.loc[test_idx, CHLRFU_COL] = float("nan")

    print(f"Evaluation setup: {len(valid_idx)} valid rows, {n_test} test positions masked.\n")

    # ------------------------------------------------------------------ #
    # 2. Define methods                                                   #
    # ------------------------------------------------------------------ #
    fast_interp = ChlorophyllInterpolator(df_exp)
    adv_interp = AdvancedChlorophyllImputation(df_exp)

    method_funcs = {
        "Linear": fast_interp.linear_interpolate,
        "Spline": lambda: fast_interp.spline_interpolate(order=3),
        "Polynomial": lambda: fast_interp.polynomial_interpolate(degree=3),
        "kNN": lambda: fast_interp.knn_imputation(k=5),
        "LightGBM": lambda: fast_interp.lightgbm_imputation(n_estimators=50, max_iter=5),
        "MissForest": lambda: adv_interp.missforest_imputation(n_estimators=50, max_iter=5),
        "GaussianProcess": lambda: adv_interp.gaussian_process_imputation(n_restarts_optimizer=3),
    }

    # ------------------------------------------------------------------ #
    # 3. Run each method and collect metrics                              #
    # ------------------------------------------------------------------ #
    records = []
    for name, func in method_funcs.items():
        print(f"Running {name} ...")
        t0 = time.time()
        result_df = func()
        elapsed = round(time.time() - t0, 3)

        y_pred = result_df.loc[test_idx, CHLRFU_COL].values
        # Clamp negative predictions (chlorophyll cannot be negative)
        y_pred = np.clip(y_pred, 0, None)

        metrics = evaluate(y_true, y_pred)
        metrics["Method"] = name
        metrics["Group"] = "FAST" if name in {"Linear", "Spline", "Polynomial", "kNN", "LightGBM"} else "ADVANCED"
        metrics["Time_s"] = elapsed
        records.append(metrics)

    # ------------------------------------------------------------------ #
    # 4. Build summary DataFrame                                          #
    # ------------------------------------------------------------------ #
    cols_order = ["Method", "Group", "MAE", "RMSE", "MAPE", "R2", "Time_s"]
    summary_df = pd.DataFrame(records)[cols_order].sort_values("RMSE")
    return summary_df


def write_summary(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Imputation Method Performance Summary\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("Ranked by RMSE (lower = better)\n\n")

        for rank, (_, row) in enumerate(df.iterrows(), 1):
            star = "⭐⭐⭐" if rank == 1 else ("⭐⭐" if rank == 2 else "")
            fh.write(
                f"#{rank:2d} {row['Method']:<20} [{row['Group']}]  "
                f"MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}  "
                f"MAPE={row['MAPE']:.2f}%  R²={row['R2']:.4f}  "
                f"Time={row['Time_s']:.3f}s  {star}\n"
            )

        fh.write("\n\nColumn definitions\n")
        fh.write("-" * 40 + "\n")
        fh.write("MAE  = Mean Absolute Error (RFU)\n")
        fh.write("RMSE = Root Mean Squared Error (RFU)\n")
        fh.write("MAPE = Mean Absolute Percentage Error (%)\n")
        fh.write("R²   = Coefficient of determination\n")
        fh.write("Time = Wall-clock execution time (seconds)\n")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not os.path.exists(INPUT_B7REMOVED):
        print(f"Input file not found: {INPUT_B7REMOVED}")
        print("Run scripts/remove_biofouling_b7.py first.")
        sys.exit(1)

    summary_df = run_comparison()

    csv_path = os.path.join(REPORTS_DIR, "performance_comparison.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved performance CSV -> {csv_path}")

    txt_path = os.path.join(REPORTS_DIR, "performance_summary.txt")
    write_summary(summary_df, txt_path)
    print(f"Saved performance summary -> {txt_path}")

    print("\n" + "=" * 60)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
