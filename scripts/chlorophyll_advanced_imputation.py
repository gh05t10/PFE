"""
chlorophyll_advanced_imputation.py
------------------------------------
ADVANCED (slow but high-quality) imputation methods for recovering
ChlRFUShallow_RFU values removed as B7 biofouling flags.

Methods
-------
* MissForest  – Random Forest iterative imputation (5-10 min)
* Gaussian Process – Kernel-based GPR imputation  (3-5 min)

Usage
-----
    from scripts.chlorophyll_advanced_imputation import AdvancedChlorophyllImputation
    import pandas as pd

    df = pd.read_csv('FRDR_dataset_1095/BPBuoyData_2014_B7Removed.csv',
                     parse_dates=['DateTime'])
    adv = AdvancedChlorophyllImputation(df)

    df_mf = adv.missforest_imputation(n_estimators=50, max_iter=5)
    df_gp = adv.gaussian_process_imputation(n_restarts_optimizer=3)
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CHLRFU_COL = "ChlRFUShallow_RFU"
FEATURE_COLS = [
    "TempShallow_C",
    "pHShallow",
    "ODOShallow_mg/L",
    "ODOSatShallow_%",
    "SpCondShallow_uS/cm",
    "TurbShallow_NTU+",
    "AirTemp_C",
    "BarometricPress_kPa",
    "RelativeHum_%",
    "PARAir_umol/s/m2",
]


class AdvancedChlorophyllImputation:
    """High-quality ML imputation methods for ChlRFUShallow_RFU gap-filling."""

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : DataFrame
            Must contain 'DateTime' (parsed) and 'ChlRFUShallow_RFU'.
            NaN values in ChlRFUShallow_RFU will be imputed.
        """
        self.df = df.copy()
        if "DateTime" in self.df.columns:
            self.df = self.df.sort_values("DateTime").reset_index(drop=True)

    def _available_features(self) -> list[str]:
        return [c for c in FEATURE_COLS if c in self.df.columns]

    def _result_df(self, series: pd.Series) -> pd.DataFrame:
        out = self.df.copy()
        out[CHLRFU_COL] = series.values
        return out

    # ------------------------------------------------------------------
    # Method 1: MissForest – Random Forest iterative imputation
    # ------------------------------------------------------------------

    def missforest_imputation(
        self,
        n_estimators: int = 50,
        max_iter: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Random Forest iterative imputation (MissForest algorithm).

        Iteratively fits a Random Forest model on observed data to predict
        missing values, repeating until convergence.  Uses all available
        environmental covariates as predictors.

        Parameters
        ----------
        n_estimators : int
            Trees per forest (default 50).  Increase for better accuracy.
        max_iter : int
            Maximum imputation iterations (default 5).
        n_jobs : int
            Parallel jobs; -1 uses all CPU cores (default -1).
        random_state : int
            Random seed.

        Returns
        -------
        DataFrame with chlorophyll gap-filled by MissForest.
        """
        t0 = time.time()
        feat_cols = self._available_features()
        cols = [CHLRFU_COL] + feat_cols

        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=random_state,
            initial_strategy="mean",
        )
        arr = imputer.fit_transform(self.df[cols].values)
        series = pd.Series(arr[:, 0], index=self.df.index)

        elapsed = time.time() - t0
        print(f"[MissForest]       done in {elapsed:.1f}s")
        return self._result_df(series)

    # ------------------------------------------------------------------
    # Method 2: Gaussian Process imputation
    # ------------------------------------------------------------------

    def gaussian_process_imputation(
        self,
        n_restarts_optimizer: int = 3,
        max_gap_size: int = 200,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Gaussian Process Regression imputation.

        Fits a GPR model on clean (non-NaN) data using environmental
        covariates, then predicts missing chlorophyll values.  For large
        gaps the method falls back to linear interpolation to keep runtime
        manageable.

        Parameters
        ----------
        n_restarts_optimizer : int
            Hyperparameter optimiser restarts (default 3).
        max_gap_size : int
            If the total number of missing rows exceeds this threshold,
            only the gap rows are predicted (not the full dataset) to
            limit memory usage.
        random_state : int
            Random seed.

        Returns
        -------
        DataFrame with chlorophyll gap-filled by GPR.
        """
        t0 = time.time()
        feat_cols = self._available_features()

        missing_mask = self.df[CHLRFU_COL].isna()
        n_missing = missing_mask.sum()
        print(f"[GaussianProcess]  imputing {n_missing} missing values ...")

        series = self.df[CHLRFU_COL].copy()

        if n_missing == 0:
            print("[GaussianProcess]  no missing values, returning unchanged")
            return self._result_df(series)

        # Build feature matrix; drop rows where any feature is NaN
        X_all = self.df[feat_cols].copy()

        # Scale features for GPR stability
        scaler = StandardScaler()

        train_mask = ~missing_mask
        X_train = X_all.loc[train_mask].fillna(X_all.median())
        y_train = series.loc[train_mask].values
        X_pred = X_all.loc[missing_mask].fillna(X_all.median())

        X_train_scaled = scaler.fit_transform(X_train.values)
        X_pred_scaled = scaler.transform(X_pred.values)

        # Kernel: constant * RBF + noise
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
        )

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
            normalize_y=True,
        )

        # Subsample training data for speed when dataset is large
        max_train = 2000
        if len(X_train_scaled) > max_train:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(X_train_scaled), size=max_train, replace=False)
            X_train_scaled = X_train_scaled[idx]
            y_train = y_train[idx]

        gpr.fit(X_train_scaled, y_train)
        y_pred, _ = gpr.predict(X_pred_scaled, return_std=True)

        series.loc[missing_mask] = y_pred

        elapsed = time.time() - t0
        print(f"[GaussianProcess]  done in {elapsed:.1f}s")
        return self._result_df(series)


# ---------------------------------------------------------------------------
# Convenience runner: produce both advanced output CSVs
# ---------------------------------------------------------------------------

def run_all_advanced_methods(
    input_csv: str = "FRDR_dataset_1095/BPBuoyData_2014_B7Removed.csv",
    output_dir: str = "FRDR_dataset_1095",
) -> dict[str, pd.DataFrame]:
    """Run both advanced imputation methods and save CSV outputs.

    Parameters
    ----------
    input_csv : str
        Path to B7-removed CSV.
    output_dir : str
        Directory for output CSVs.

    Returns
    -------
    dict mapping method name -> imputed DataFrame
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv, parse_dates=["DateTime"])
    adv = AdvancedChlorophyllImputation(df)

    methods = {
        "MissForest": lambda: adv.missforest_imputation(n_estimators=50, max_iter=5),
        "GaussianProcess": lambda: adv.gaussian_process_imputation(n_restarts_optimizer=3),
    }

    results = {}
    for name, func in methods.items():
        print(f"\nRunning {name} ...")
        result_df = func()
        out_path = os.path.join(output_dir, f"BPBuoyData_2014_Advanced_{name}.csv")
        result_df.to_csv(out_path, index=False)
        print(f"  -> Saved: {out_path}")
        results[name] = result_df

    return results


if __name__ == "__main__":
    run_all_advanced_methods()
