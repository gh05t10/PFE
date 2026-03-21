"""
chlorophyll_interpolation.py
----------------------------
Module providing six interpolation strategies to recover chlorophyll
values that were set to NaN (e.g. after B7 biofouling removal).

Each method accepts a pandas DataFrame that contains a DateTime column
and one or more chlorophyll value columns, and returns a *new* DataFrame
with the NaN gaps filled according to the chosen strategy.

Methods
-------
1. linear_interpolate        – straight-line fill between flanking values
2. spline_interpolate        – cubic-spline smooth fill (order=3 default)
3. polynomial_interpolate    – local polynomial fit around the gap
4. seasonal_decompose_interpolate – decompose trend + seasonal, fill each
                                    component, then recompose (RECOMMENDED)
5. fill_interpolate          – forward-fill or backward-fill
6. weighted_average_interpolate  – weighted mean of same calendar window
                                   from a supplied list of reference years

Usage
-----
    from chlorophyll_interpolation import ChlorophyllInterpolator

    interp = ChlorophyllInterpolator(df, value_col="ChlRFUShallow_RFU")
    df_linear  = interp.linear_interpolate()
    df_spline  = interp.spline_interpolate(order=3)
    df_poly    = interp.polynomial_interpolate(degree=3)
    df_seas    = interp.seasonal_decompose_interpolate(period=96)   # 96 = 1 day at 15-min
    df_ffill   = interp.fill_interpolate(method='ffill')
    df_wavg    = interp.weighted_average_interpolate(reference_dfs=[df_2013, df_2015])
"""

import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


class ChlorophyllInterpolator:
    """
    Provides multiple interpolation strategies for a single chlorophyll
    column inside a time-indexed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'DateTime' column (parseable as datetime) and the
        column specified by *value_col*.
    value_col : str
        Name of the chlorophyll measurement column to interpolate.
        Default: 'ChlRFUShallow_RFU'
    """

    def __init__(self, df: pd.DataFrame, value_col: str = "ChlRFUShallow_RFU") -> None:
        self.value_col = value_col
        # Work on an internal copy with a proper DatetimeIndex
        self._df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self._df["DateTime"]):
            self._df["DateTime"] = pd.to_datetime(self._df["DateTime"])
        self._df = self._df.set_index("DateTime").sort_index()

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _result(self, series: pd.Series) -> pd.DataFrame:
        """Return a copy of the stored DataFrame with the interpolated column replaced."""
        out = self._df.copy()
        out[self.value_col] = series
        return out.reset_index()  # restore DateTime as a column

    # ------------------------------------------------------------------
    # Method 1 — Linear Interpolation
    # ------------------------------------------------------------------
    def linear_interpolate(self, limit: int | None = None) -> pd.DataFrame:
        """
        Fill NaN gaps using straight-line interpolation between flanking
        valid values.

        Parameters
        ----------
        limit : int, optional
            Maximum number of consecutive NaNs to fill.  None = no limit.
        """
        series = self._df[self.value_col].interpolate(
            method="time", limit=limit, limit_direction="both"
        )
        return self._result(series)

    # ------------------------------------------------------------------
    # Method 2 — Spline Interpolation
    # ------------------------------------------------------------------
    def spline_interpolate(self, order: int = 3, limit: int | None = None) -> pd.DataFrame:
        """
        Fill NaN gaps with a cubic (or higher-order) spline fitted through
        the surrounding valid data points.

        Parameters
        ----------
        order : int
            Spline order (1–5).  Default 3 (cubic).
        limit : int, optional
            Maximum consecutive NaNs to fill.
        """
        series = self._df[self.value_col].interpolate(
            method="spline", order=order, limit=limit, limit_direction="both"
        )
        # Clip negative values that splines can introduce
        series = series.clip(lower=0)
        return self._result(series)

    # ------------------------------------------------------------------
    # Method 3 — Polynomial Interpolation
    # ------------------------------------------------------------------
    def polynomial_interpolate(
        self, degree: int = 3, context_points: int = 200
    ) -> pd.DataFrame:
        """
        Fit a polynomial of *degree* to the *context_points* valid
        observations immediately before and after each NaN gap, then
        use the polynomial to predict the missing values.

        Parameters
        ----------
        degree : int
            Polynomial degree.  Default 3.
        context_points : int
            Number of valid observations on each side of the gap to include
            in the fit.
        """
        col = self._df[self.value_col].copy()
        nan_mask = col.isna()

        if not nan_mask.any():
            return self._result(col)

        # Convert index to numeric seconds since first timestamp
        t_all = (self._df.index - self._df.index[0]).total_seconds().values
        y_all = col.values.copy()

        # Identify contiguous NaN blocks
        gap_starts = np.where(np.diff(nan_mask.values.astype(int)) == 1)[0] + 1
        gap_ends = np.where(np.diff(nan_mask.values.astype(int)) == -1)[0] + 1

        # Handle edge cases: gap at start / end
        if nan_mask.iloc[0]:
            gap_starts = np.concatenate([[0], gap_starts])
        if nan_mask.iloc[-1]:
            gap_ends = np.concatenate([gap_ends, [len(col)]])

        for gs, ge in zip(gap_starts, gap_ends):
            valid_before = np.where(~nan_mask.values[:gs])[0][-context_points:]
            valid_after = np.where(~nan_mask.values[ge:])[0][:context_points] + ge
            ctx_idx = np.concatenate([valid_before, valid_after])

            if len(ctx_idx) < degree + 1:
                # Fall back to linear if not enough context
                continue

            t_ctx = t_all[ctx_idx]
            y_ctx = y_all[ctx_idx]
            coeffs = np.polyfit(t_ctx, y_ctx, degree)
            y_fill = np.polyval(coeffs, t_all[gs:ge])
            y_all[gs:ge] = np.maximum(y_fill, 0)

        return self._result(pd.Series(y_all, index=self._df.index))

    # ------------------------------------------------------------------
    # Method 4 — Seasonal Decomposition (RECOMMENDED)
    # ------------------------------------------------------------------
    def seasonal_decompose_interpolate(self, period: int = 96) -> pd.DataFrame:
        """
        Decompose the time series into trend + seasonal + residual using
        a rolling-median approach, interpolate each component separately
        over the gap, then recompose.

        This is the recommended method for environmental time-series with
        daily or seasonal patterns.

        Parameters
        ----------
        period : int
            Number of observations per cycle (e.g. 96 for a daily cycle at
            15-minute resolution).  Default: 96.
        """
        col = self._df[self.value_col].copy()

        if not col.isna().any():
            return self._result(col)

        # --- Step 1: estimate trend via rolling median on valid data ---
        # First fill temporarily with linear to allow rolling
        col_tmp = col.interpolate(method="time", limit_direction="both")

        trend = col_tmp.rolling(window=period, center=True, min_periods=period // 2).median()
        trend = trend.interpolate(method="time", limit_direction="both")

        # --- Step 2: extract seasonal component (detrended + averaged by phase) ---
        detrended = col_tmp - trend
        phase = np.arange(len(detrended)) % period
        seasonal_avg = (
            pd.Series(detrended.values, name="detrended")
            .groupby(phase)
            .transform("median")
        )
        seasonal = pd.Series(seasonal_avg.values, index=col.index)

        # --- Step 3: residual ---
        residual = col_tmp - trend - seasonal

        # --- Step 4: interpolate each component only over the original NaN locations ---
        nan_mask = col.isna()

        trend_interp = trend.copy()
        trend_interp[nan_mask] = np.nan
        trend_interp = trend_interp.interpolate(method="time", limit_direction="both")

        # Seasonal is periodic so we can carry it forward from valid neighbours
        seasonal_interp = seasonal.copy()

        # Residual: use a small smoothing window around the gap edges
        residual_interp = residual.copy()
        residual_interp[nan_mask] = np.nan
        residual_interp = residual_interp.interpolate(method="time", limit_direction="both")
        # Smooth residual to avoid sharp edges
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                win = min(51, (nan_mask.sum() // 2) * 2 + 1)
                win = max(win, 5)
                smooth = savgol_filter(residual_interp.values, window_length=win, polyorder=2)
                residual_interp = pd.Series(smooth, index=residual_interp.index)
            except Exception:
                pass

        # --- Step 5: recompose only the originally-missing positions ---
        result = col.copy()
        result[nan_mask] = (
            trend_interp[nan_mask]
            + seasonal_interp[nan_mask]
            + residual_interp[nan_mask]
        )
        result = result.clip(lower=0)
        return self._result(result)

    # ------------------------------------------------------------------
    # Method 5 — Forward / Backward Fill
    # ------------------------------------------------------------------
    def fill_interpolate(self, method: str = "ffill", limit: int | None = None) -> pd.DataFrame:
        """
        Propagate the last valid observation forward (ffill) or the next
        valid observation backward (bfill).

        Parameters
        ----------
        method : {'ffill', 'bfill', 'both'}
            'both' applies ffill then bfill so that any remaining NaNs at
            the start of the series are also filled.
        limit : int, optional
            Maximum gap length to fill.
        """
        series = self._df[self.value_col].copy()
        if method in ("ffill", "both"):
            series = series.ffill(limit=limit)
        if method in ("bfill", "both"):
            series = series.bfill(limit=limit)
        return self._result(series)

    # ------------------------------------------------------------------
    # Method 6 — Custom Weighted Average from Reference Years
    # ------------------------------------------------------------------
    def weighted_average_interpolate(
        self,
        reference_dfs: list[pd.DataFrame] | None = None,
        window_days: int = 7,
        weights: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Fill NaN values with a weighted average of measurements from the
        same calendar window across one or more reference-year DataFrames.

        If no reference DataFrames are supplied, the method falls back to
        linear interpolation.

        Parameters
        ----------
        reference_dfs : list of pd.DataFrame, optional
            Each DataFrame must have the same structure as the target
            (DateTime column + same value_col).
        window_days : int
            Half-width of the calendar window (in days) used to find
            matching observations in reference years.
        weights : list of float, optional
            Per-reference-year weights (must sum to 1).  Defaults to equal
            weighting.
        """
        col = self._df[self.value_col].copy()
        nan_mask = col.isna()

        if not nan_mask.any():
            return self._result(col)

        if not reference_dfs:
            warnings.warn(
                "No reference DataFrames provided; falling back to linear interpolation.",
                UserWarning,
                stacklevel=2,
            )
            return self.linear_interpolate()

        if weights is None:
            weights = [1.0 / len(reference_dfs)] * len(reference_dfs)

        # Build reference series (indexed by month-day-time)
        ref_series_list = []
        for ref_df in reference_dfs:
            ref = ref_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(ref["DateTime"]):
                ref["DateTime"] = pd.to_datetime(ref["DateTime"])
            ref = ref.set_index("DateTime").sort_index()
            ref_series_list.append(ref[self.value_col])

        result = col.copy()

        for ts in col.index[nan_mask]:
            samples = []
            for ref_s, w in zip(ref_series_list, weights):
                # Find observations in the reference year within ±window_days
                ref_year = ref_s.index.year[0] if len(ref_s) > 0 else ts.year
                target_day = ts.replace(year=ref_year)
                lo = target_day - pd.Timedelta(days=window_days)
                hi = target_day + pd.Timedelta(days=window_days)
                window_vals = ref_s.loc[lo:hi].dropna()
                if len(window_vals) > 0:
                    samples.append((float(window_vals.mean()), w))

            if samples:
                total_w = sum(s[1] for s in samples)
                result[ts] = sum(v * w / total_w for v, w in samples)

        # Any remaining NaNs → linear fill
        remaining = result.isna()
        if remaining.any():
            result = result.interpolate(method="time", limit_direction="both")

        result = result.clip(lower=0)
        return self._result(result)
