"""
chlorophyll_interpolation.py
-----------------------------
Provides the ChlorophyllInterpolator class with six interpolation / imputation
methods for reconstructing missing (NaN) chlorophyll values:

  1. linear_interpolate        – time-based linear interpolation
  2. spline_interpolate        – cubic spline interpolation
  3. polynomial_interpolate    – polynomial interpolation (order 3)
  4. knn_imputation            – k-Nearest Neighbours imputation (sklearn)
  5. missforest_imputation     – Random-Forest iterative imputation (sklearn)
  6. mice_imputation           – MICE via sklearn IterativeImputer (BayesianRidge)

All methods operate on a copy of the input DataFrame and return the completed
DataFrame without modifying the original.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge

CHLOROPHYLL_COL = "ChlRFUShallow_RFU"

# Margin (in standard deviations) used to clip polynomial interpolation
# to a physically plausible range.
STD_MARGIN = 1.0

NUMERIC_FEATURES = [
    "BarometricPress_kPa",
    "RelativeHum_%",
    "WindSp_km/h",
    "AirTemp_C",
    "TempShallow_C",
    "ODOSatShallow_%",
    "ODOShallow_mg/L",
    "SpCondShallow_uS/cm",
    "TurbShallow_NTU+",
    "BGPCShallowRFU_RFU",
    "TempDeep_C",
]


class ChlorophyllInterpolator:
    """
    Encapsulates six methods for imputing NaN chlorophyll values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that must contain at least the 'ChlRFUShallow_RFU' column
        and a 'DateTime' column (or be indexed by datetime).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if "DateTime" in self.df.columns:
            self.df = self.df.set_index("DateTime")

    def _result(self, series: pd.Series) -> pd.DataFrame:
        """Return a DataFrame copy with the chlorophyll column replaced."""
        out = self.df.copy()
        out[CHLOROPHYLL_COL] = series
        return out.reset_index()

    # ------------------------------------------------------------------
    # 1. Linear Interpolation
    # ------------------------------------------------------------------
    def linear_interpolate(self) -> pd.DataFrame:
        """Baseline: time-based linear interpolation.

        Falls back to forward/backward fill for boundary NaN values.
        """
        series = self.df[CHLOROPHYLL_COL].interpolate(method="time")
        series = series.ffill().bfill()
        return self._result(series)

    # ------------------------------------------------------------------
    # 2. Spline Interpolation
    # ------------------------------------------------------------------
    def spline_interpolate(self, order: int = 3) -> pd.DataFrame:
        """Smooth cubic-spline interpolation.

        Falls back to linear fill for any boundary NaN values that the
        spline cannot handle (e.g. gaps at the very start/end of the series).
        """
        series = self.df[CHLOROPHYLL_COL].interpolate(
            method="spline", order=order
        )
        # Fill remaining boundary NaN with linear interpolation
        series = series.interpolate(method="linear").ffill().bfill()
        return self._result(series)

    # ------------------------------------------------------------------
    # 3. Polynomial Interpolation
    # ------------------------------------------------------------------
    def polynomial_interpolate(self, order: int = 3) -> pd.DataFrame:
        """Polynomial interpolation of the given order.

        Falls back to linear fill for any boundary NaN values and clips
        results to a physically plausible range to prevent Runge oscillations.
        """
        series = self.df[CHLOROPHYLL_COL].interpolate(
            method="polynomial", order=order
        )
        # Fill remaining boundary NaN with linear interpolation
        series = series.interpolate(method="linear").ffill().bfill()
        # Clip to valid range determined from observed (non-NaN) values
        valid = self.df[CHLOROPHYLL_COL].dropna()
        if len(valid) > 0:
            lo = max(0.0, valid.min() - STD_MARGIN * valid.std())
            hi = valid.max() + STD_MARGIN * valid.std()
            series = series.clip(lower=lo, upper=hi)
        return self._result(series)

    # ------------------------------------------------------------------
    # 4. kNN Imputation
    # ------------------------------------------------------------------
    def knn_imputation(self, k: int = 5) -> pd.DataFrame:
        """
        k-Nearest Neighbours imputation using chlorophyll plus auxiliary
        numeric features to find the k closest rows.
        """
        cols = self._feature_cols()
        subset = self.df[cols].copy()

        imputer = KNNImputer(n_neighbors=k)
        imputed = imputer.fit_transform(subset)
        imputed_df = pd.DataFrame(imputed, columns=cols, index=self.df.index)

        return self._result(imputed_df[CHLOROPHYLL_COL])

    # ------------------------------------------------------------------
    # 5. MissForest (Random Forest iterative imputation)
    # ------------------------------------------------------------------
    def missforest_imputation(
        self,
        n_estimators: int = 100,
        max_iter: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Random-Forest iterative imputation (equivalent to MissForest).
        Uses sklearn's IterativeImputer with a RandomForestRegressor estimator.
        """
        cols = self._feature_cols()
        subset = self.df[cols].copy()

        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=random_state,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputed = imputer.fit_transform(subset)

        imputed_df = pd.DataFrame(imputed, columns=cols, index=self.df.index)
        return self._result(imputed_df[CHLOROPHYLL_COL])

    # ------------------------------------------------------------------
    # 6. MICE (Multiple Imputation by Chained Equations)
    # ------------------------------------------------------------------
    def mice_imputation(
        self,
        n_imputations: int = 5,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        MICE via sklearn's IterativeImputer (BayesianRidge estimator).
        Runs n_imputations independent draws and averages the results.
        """
        cols = self._feature_cols()
        subset = self.df[cols].values

        results = []
        for i in range(n_imputations):
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=10,
                random_state=random_state + i,
                sample_posterior=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imputed = imputer.fit_transform(subset)
            chl_idx = cols.index(CHLOROPHYLL_COL)
            results.append(imputed[:, chl_idx])

        averaged = np.mean(results, axis=0)
        series = pd.Series(averaged, index=self.df.index, name=CHLOROPHYLL_COL)
        return self._result(series)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _feature_cols(self) -> list:
        """Return chlorophyll column + available numeric feature columns."""
        available = [
            c for c in NUMERIC_FEATURES if c in self.df.columns
        ]
        cols = [CHLOROPHYLL_COL] + available
        return cols


def get_method_functions(
    missforest_n_estimators: int = 50,
    missforest_max_iter: int = 5,
    knn_k: int = 5,
    mice_n_imputations: int = 3,
) -> dict:
    """
    Return a dict mapping method names to callables ``fn(interpolator)``.

    All hyperparameters are centralised here so that performance_comparison,
    cross_validation, and visualization_comparison stay consistent.
    """
    return {
        "Linear":     lambda i: i.linear_interpolate(),
        "Spline":     lambda i: i.spline_interpolate(),
        "Polynomial": lambda i: i.polynomial_interpolate(),
        "kNN":        lambda i: i.knn_imputation(k=knn_k),
        "MissForest": lambda i: i.missforest_imputation(
            n_estimators=missforest_n_estimators,
            max_iter=missforest_max_iter,
        ),
        "MICE":       lambda i: i.mice_imputation(
            n_imputations=mice_n_imputations,
        ),
    }
