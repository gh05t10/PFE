"""
cross_validation.py
--------------------
K-Fold cross-validation of all 7 imputation methods on the
non-B7 portion of the 2014 chlorophyll data.

For each fold:
  - Mask ~1/k of the valid chlorophyll values as "missing"
  - Apply every method to recover them
  - Compute MAE, RMSE, MAPE, R² per fold

Final output: mean ± std across k folds (confidence intervals).

Output
------
reports/cross_validation_results.csv
visualizations/06_cv_heatmap.png  (produced by visualization_comparison.py)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.chlorophyll_interpolation import ChlorophyllInterpolator
from scripts.chlorophyll_advanced_imputation import AdvancedChlorophyllImputation

DATA_DIR = "FRDR_dataset_1095"
REPORTS_DIR = "reports"
INPUT_B7REMOVED = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")

CHLRFU_COL = "ChlRFUShallow_RFU"
N_SPLITS = 5
RANDOM_STATE = 42


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Replace any remaining NaN predictions with the mean of y_true
    if np.isnan(y_pred).any():
        fallback = float(np.nanmean(y_pred)) if not np.isnan(y_pred).all() else float(np.mean(y_true))
        y_pred = np.where(np.isnan(y_pred), fallback, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


METHOD_NAMES = ["Linear", "Spline", "Polynomial", "kNN", "LightGBM", "MissForest", "GaussianProcess"]
METHOD_GROUPS = {
    "Linear": "FAST", "Spline": "FAST", "Polynomial": "FAST",
    "kNN": "FAST", "LightGBM": "FAST",
    "MissForest": "ADVANCED", "GaussianProcess": "ADVANCED",
}


def _run_methods_on(df_exp: pd.DataFrame, test_idx) -> dict[str, np.ndarray]:
    """Apply all 7 methods to df_exp and return predicted values at test_idx."""
    fast = ChlorophyllInterpolator(df_exp)
    adv = AdvancedChlorophyllImputation(df_exp)

    funcs = {
        "Linear": fast.linear_interpolate,
        "Spline": lambda: fast.spline_interpolate(order=3),
        "Polynomial": lambda: fast.polynomial_interpolate(degree=3),
        "kNN": lambda: fast.knn_imputation(k=5),
        "LightGBM": lambda: fast.lightgbm_imputation(n_estimators=30, max_iter=3),
        "MissForest": lambda: adv.missforest_imputation(n_estimators=30, max_iter=3),
        "GaussianProcess": lambda: adv.gaussian_process_imputation(n_restarts_optimizer=1),
    }

    preds = {}
    for name, func in funcs.items():
        result_df = func()
        y_pred = result_df.loc[test_idx, CHLRFU_COL].values
        y_pred = np.clip(y_pred, 0, None)
        # Fill any remaining NaN with forward/backward fill fallback
        if np.isnan(y_pred).any():
            series = pd.Series(y_pred)
            series = series.ffill().bfill().fillna(series.mean())
            y_pred = series.values
        preds[name] = y_pred
    return preds


def run_cross_validation(
    input_csv: str = INPUT_B7REMOVED,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run K-Fold cross-validation for all methods.

    Returns a DataFrame with columns:
    Method, Group, Fold, MAE, RMSE, MAPE, R2, Time_s
    """
    df_base = pd.read_csv(input_csv, parse_dates=["DateTime"])

    valid_mask = df_base[CHLRFU_COL].notna()
    valid_idx = np.array(df_base.index[valid_mask].tolist())

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []

    for fold_num, (_, test_fold) in enumerate(kf.split(valid_idx), 1):
        test_idx = valid_idx[test_fold]
        y_true = df_base.loc[test_idx, CHLRFU_COL].values.copy()

        df_exp = df_base.copy()
        df_exp.loc[test_idx, CHLRFU_COL] = float("nan")

        print(f"\n--- Fold {fold_num}/{n_splits}  ({len(test_idx)} test samples) ---")

        t_fold = time.time()
        preds = _run_methods_on(df_exp, test_idx)
        fold_time = time.time() - t_fold

        for name, y_pred in preds.items():
            m = evaluate(y_true, y_pred)
            records.append({
                "Method": name,
                "Group": METHOD_GROUPS[name],
                "Fold": fold_num,
                "MAE": round(m["MAE"], 4),
                "RMSE": round(m["RMSE"], 4),
                "MAPE": round(m["MAPE"], 4),
                "R2": round(m["R2"], 4),
            })
            print(f"  {name:<20} MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R2']:.4f}")

    return pd.DataFrame(records)


def aggregate_cv_results(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std of metrics per method."""
    agg = (
        cv_df.groupby(["Method", "Group"])[["MAE", "RMSE", "MAPE", "R2"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values("RMSE_mean")
    return agg


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not os.path.exists(INPUT_B7REMOVED):
        print(f"Input not found: {INPUT_B7REMOVED}")
        print("Run scripts/remove_biofouling_b7.py first.")
        sys.exit(1)

    print(f"Running {N_SPLITS}-fold cross-validation for all 7 methods ...\n")
    cv_df = run_cross_validation(n_splits=N_SPLITS)

    raw_path = os.path.join(REPORTS_DIR, "cross_validation_results.csv")
    cv_df.to_csv(raw_path, index=False)
    print(f"\nSaved per-fold results -> {raw_path}")

    agg_df = aggregate_cv_results(cv_df)
    agg_path = os.path.join(REPORTS_DIR, "cross_validation_summary.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved aggregated summary -> {agg_path}")

    print("\n" + "=" * 70)
    print("Cross-Validation Summary (mean ± std)")
    print("=" * 70)
    for _, row in agg_df.iterrows():
        print(
            f"  {row['Method']:<20} [{row['Group']}]  "
            f"RMSE={row['RMSE_mean']:.4f}±{row['RMSE_std']:.4f}  "
            f"R²={row['R2_mean']:.4f}±{row['R2_std']:.4f}"
        )


if __name__ == "__main__":
    main()
