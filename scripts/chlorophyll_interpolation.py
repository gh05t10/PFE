"""
chlorophyll_interpolation.py
-----------------------------
FAST interpolation methods (< 1 minute each) for recovering
ChlRFUShallow_RFU values removed as B7 biofouling flags.

Usage
-----
    from scripts.chlorophyll_interpolation import ChlorophyllInterpolator
    import pandas as pd

    df = pd.read_csv('FRDR_dataset_1095/BPBuoyData_2014_B7Removed.csv',
                     parse_dates=['DateTime'])
    interp = ChlorophyllInterpolator(df)

    df_linear    = interp.linear_interpolate()
    df_spline    = interp.spline_interpolate(order=3)
    df_poly      = interp.polynomial_interpolate(degree=3)
    df_knn       = interp.knn_imputation(k=5)
    df_lightgbm  = interp.lightgbm_imputation(n_estimators=50, max_iter=5)
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from lightgbm import LGBMRegressor

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


class ChlorophyllInterpolator:
    """Fast interpolation methods for ChlRFUShallow_RFU gap-filling."""

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : DataFrame
            Must contain 'DateTime' (parsed) and 'ChlRFUShallow_RFU' columns.
            NaN values in ChlRFUShallow_RFU will be imputed.
        """
        self.df = df.copy()
        if "DateTime" in self.df.columns:
            self.df = self.df.sort_values("DateTime").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _result_df(self, series: pd.Series) -> pd.DataFrame:
        """Return a copy of self.df with the chlorophyll column replaced."""
        out = self.df.copy()
        out[CHLRFU_COL] = series.values
        return out

    def _available_features(self) -> list[str]:
        """Return feature columns that are present in the dataframe."""
        return [c for c in FEATURE_COLS if c in self.df.columns]

    # ------------------------------------------------------------------
    # Method 1: Linear interpolation (~0.01 s)
    # ------------------------------------------------------------------

    def linear_interpolate(self) -> pd.DataFrame:
        """Linear interpolation: connect adjacent valid values with a straight line.

        Returns
        -------
        DataFrame with NaN gaps filled using pandas linear interpolation.
        """
        t0 = time.time()
        series = self.df[CHLRFU_COL].interpolate(method="linear", limit_direction="both")
        elapsed = time.time() - t0
        print(f"[Linear]     done in {elapsed:.3f}s")
        return self._result_df(series)

    # ------------------------------------------------------------------
    # Method 2: Spline interpolation (~0.02 s)
    # ------------------------------------------------------------------

    def spline_interpolate(self, order: int = 3) -> pd.DataFrame:
        """Cubic (or arbitrary-order) spline interpolation.

        Parameters
        ----------
        order : int
            Spline order (default 3 = cubic).

        Returns
        -------
        DataFrame with gaps filled by a smooth spline curve.
        """
        t0 = time.time()
        series = self.df[CHLRFU_COL].interpolate(
            method="spline", order=order, limit_direction="both"
        )
        elapsed = time.time() - t0
        print(f"[Spline-{order}]   done in {elapsed:.3f}s")
        return self._result_df(series)

    # ------------------------------------------------------------------
    # Method 3: Polynomial interpolation (~0.03 s)
    # ------------------------------------------------------------------

    def polynomial_interpolate(self, degree: int = 3) -> pd.DataFrame:
        """Polynomial interpolation using pandas (degree-n polynomial fit).

        Parameters
        ----------
        degree : int
            Polynomial degree (default 3).

        Returns
        -------
        DataFrame with gaps filled by polynomial fit.
        """
        t0 = time.time()
        series = self.df[CHLRFU_COL].interpolate(
            method="polynomial", order=degree, limit_direction="both"
        )
        elapsed = time.time() - t0
        print(f"[Poly-{degree}]     done in {elapsed:.3f}s")
        return self._result_df(series)

    # ------------------------------------------------------------------
    # Method 4: k-Nearest Neighbors imputation (~0.5 s)
    # ------------------------------------------------------------------

    def knn_imputation(self, k: int = 5) -> pd.DataFrame:
        """k-Nearest Neighbors multivariate imputation.

        Uses environmental covariates (temperature, pH, DO, etc.) to find
        the k most similar time-steps and averages their chlorophyll values.

        Parameters
        ----------
        k : int
            Number of neighbours (default 5).

        Returns
        -------
        DataFrame with chlorophyll gap-filled using kNN.
        """
        t0 = time.time()
        feat_cols = self._available_features()
        cols = [CHLRFU_COL] + feat_cols

        imputer = KNNImputer(n_neighbors=k)
        arr = imputer.fit_transform(self.df[cols].values)
        series = pd.Series(arr[:, 0], index=self.df.index)

        elapsed = time.time() - t0
        print(f"[kNN-{k}]       done in {elapsed:.3f}s")
        return self._result_df(series)

    # ------------------------------------------------------------------
    # Method 5: LightGBM iterative imputation (~2-3 min)
    # ------------------------------------------------------------------

    def lightgbm_imputation(
        self,
        n_estimators: int = 50,
        max_iter: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """LightGBM iterative (MICE-style) imputation.

        Uses gradient boosting trees as the estimator inside
        sklearn's IterativeImputer for fast, high-quality multivariate
        gap-filling.

        Parameters
        ----------
        n_estimators : int
            Number of trees in the LightGBM estimator (default 50).
        max_iter : int
            Maximum imputation iterations (default 5).
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        DataFrame with chlorophyll gap-filled using LightGBM iterative imputation.
        """
        t0 = time.time()
        feat_cols = self._available_features()
        cols = [CHLRFU_COL] + feat_cols

        estimator = LGBMRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=random_state,
            initial_strategy="mean",
            imputation_order="ascending",
        )
        arr = imputer.fit_transform(self.df[cols].values)
        series = pd.Series(arr[:, 0], index=self.df.index)

        elapsed = time.time() - t0
        print(f"[LightGBM]   done in {elapsed:.1f}s")
        return self._result_df(series)


# ---------------------------------------------------------------------------
# Convenience runner: produce all 5 output CSVs
# ---------------------------------------------------------------------------

def run_all_fast_methods(
    input_csv: str = "FRDR_dataset_1095/BPBuoyData_2014_B7Removed.csv",
    output_dir: str = "FRDR_dataset_1095",
) -> dict[str, pd.DataFrame]:
    """Run all five fast interpolation methods and save CSV outputs.

    Parameters
    ----------
    input_csv : str
        Path to the B7-removed CSV (NaN in place of B7 values).
    output_dir : str
        Directory where output CSVs will be written.

    Returns
    -------
    dict mapping method name -> imputed DataFrame
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv, parse_dates=["DateTime"])
    interp = ChlorophyllInterpolator(df)

    methods = {
        "Linear": interp.linear_interpolate,
        "Spline": lambda: interp.spline_interpolate(order=3),
        "Polynomial": lambda: interp.polynomial_interpolate(degree=3),
        "kNN": lambda: interp.knn_imputation(k=5),
        "LightGBM": lambda: interp.lightgbm_imputation(n_estimators=50, max_iter=5),
    }

    results = {}
    for name, func in methods.items():
        print(f"\nRunning {name} ...")
        result_df = func()
        out_path = os.path.join(output_dir, f"BPBuoyData_2014_Fast_{name}.csv")
        result_df.to_csv(out_path, index=False)
        print(f"  -> Saved: {out_path}")
        results[name] = result_df

    return results


if __name__ == "__main__":
    run_all_fast_methods()
