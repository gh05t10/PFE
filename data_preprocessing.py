"""data_preprocessing.py
~~~~~~~~~~~~~~~~~~~~~~~
Full preprocessing pipeline for the Buffalo Pound Lake buoy dataset.

The pipeline focuses on ``ChlRFUShallow_RFU`` (shallow chlorophyll at 0.8 m depth)
which serves as ground-truth for downstream forecasting models.

Pipeline steps
--------------
1. Load each annual CSV (2014-2021) and concatenate into one DataFrame.
2. Apply dataset quality flags:
   - B7 (biofouling spike / decrease after site visit) → ``ChlRFUShallow_RFU`` set to NaN
   - C  (faulty / unrealistic data)                    → ``ChlRFUShallow_RFU`` set to NaN
   - M  (missing)                                      → value is already NaN in source
3. Physical-range validation: values < 0 → NaN (fluorescence cannot be negative).
4. Export per-year preprocessed CSVs preserving all original columns.
5. Statistical outlier removal using a rolling 24-hour IQR upper fence (k = 4.5),
   targeting isolated spikes that were not caught by the quality flags.
6. Resample the combined series to a regular 10-minute grid (mean aggregation).
7. Linear interpolation of short gaps (≤ 2 h = 12 consecutive missing 10-min bins).
   Longer gaps remain NaN to preserve temporal structure for the forecasting model.
8. Export ``ChlRFUShallow_RFU_GroundTruth.csv`` — a single-column, regularly-sampled,
   gap-filled time series ready for use as model ground truth.
9. Print a gap and data-coverage summary.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ─── constants ────────────────────────────────────────────────────────────────

DATA_DIR = "FRDR_dataset_1095"
YEARS = list(range(2014, 2022))
TARGET_COL = "ChlRFUShallow_RFU"
FLAG_COL = "ChlRFUShallow_RFU_Flag"
# Quality flags that render the chlorophyll measurement invalid
INVALID_FLAGS = {"B7", "C"}
# Regular temporal resolution for the ground-truth grid
RESAMPLE_FREQ = "10min"
# Maximum consecutive gap duration to fill by linear interpolation
INTERP_MAX_GAP = pd.Timedelta("2h")
# IQR fence multiplier for upper-tail outlier detection (conservative)
OUTLIER_IQR_K = 4.5
# Rolling time window used for outlier detection
ROLLING_WINDOW = "24h"


# ─── step 1: load ─────────────────────────────────────────────────────────────

def load_year(year: int, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load one annual CSV, parse DateTime, and tag with a Year column."""
    path = os.path.join(data_dir, f"BPBuoyData_{year}_Cleaned.csv")
    df = pd.read_csv(path, parse_dates=["DateTime"], low_memory=False)
    df["Year"] = year
    return df


def load_all_years(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load and concatenate all annual CSVs into a single DataFrame."""
    frames = [load_year(y, data_dir) for y in YEARS]
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("DateTime", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


# ─── step 2: quality flags ────────────────────────────────────────────────────

def apply_quality_flags(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    flag_col: str = FLAG_COL,
    invalid_flags: set = INVALID_FLAGS,
) -> tuple:
    """Set *target* to NaN wherever *flag_col* is in *invalid_flags*.

    Parameters
    ----------
    df : DataFrame
        Must contain *target* and *flag_col* columns.
    target : str
        Measurement column to invalidate.
    flag_col : str
        Quality-flag column.
    invalid_flags : set
        Flag codes that indicate unusable data.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Modified DataFrame and count of invalidated rows.
    """
    df = df.copy()
    mask = df[flag_col].isin(invalid_flags)
    df.loc[mask, target] = np.nan
    return df, int(mask.sum())


# ─── step 3: physical bounds ──────────────────────────────────────────────────

def apply_physical_bounds(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    min_val: float = 0.0,
) -> tuple:
    """Replace values below *min_val* with NaN (fluorescence cannot be negative).

    Parameters
    ----------
    df : DataFrame
    target : str
        Measurement column to validate.
    min_val : float
        Lower physical bound (default 0.0 RFU).

    Returns
    -------
    tuple[pd.DataFrame, int]
        Modified DataFrame and count of clamped rows.
    """
    df = df.copy()
    mask = df[target] < min_val
    df.loc[mask, target] = np.nan
    return df, int(mask.sum())


# ─── step 5: statistical outlier removal ─────────────────────────────────────

def remove_statistical_outliers(
    series: pd.Series,
    window: str = ROLLING_WINDOW,
    k: float = OUTLIER_IQR_K,
) -> tuple:
    """Remove isolated upper-tail spikes using a rolling IQR fence.

    A rolling Q75 and IQR are computed over a *window*-wide time window.
    Any value above Q75 + k × IQR is set to NaN.  Only the upper fence is
    applied because chlorophyll cannot be negative and low values carry
    physical meaning (e.g. night-time minima).

    Parameters
    ----------
    series : pd.Series
        Time-indexed ChlRFU series (must have a DatetimeIndex).
    window : str
        Rolling time window string passed to ``pd.Series.rolling`` (default '24h').
    k : float
        IQR fence multiplier (default 4.5 — very conservative).

    Returns
    -------
    tuple[pd.Series, int]
        Cleaned series and count of outliers removed.
    """
    series = series.copy()
    rolling = series.rolling(window, center=True, min_periods=3)
    q75 = rolling.quantile(0.75)
    q25 = rolling.quantile(0.25)
    iqr = q75 - q25
    upper_fence = q75 + k * iqr
    outlier_mask = series > upper_fence
    series.loc[outlier_mask] = np.nan
    return series, int(outlier_mask.sum())


# ─── step 6: resample to regular 10-minute grid ───────────────────────────────

def resample_to_regular_grid(
    series: pd.Series,
    freq: str = RESAMPLE_FREQ,
) -> pd.Series:
    """Resample *series* (time-indexed) to a regular *freq* grid via mean aggregation.

    Bins with no original data will be NaN.

    Parameters
    ----------
    series : pd.Series
        Time-indexed series.
    freq : str
        Target frequency string (default '10min').

    Returns
    -------
    pd.Series
        Regularly-sampled series with DatetimeIndex at *freq* resolution.
    """
    return series.resample(freq).mean()


# ─── step 7: gap interpolation ────────────────────────────────────────────────

def interpolate_short_gaps(
    series: pd.Series,
    max_gap: pd.Timedelta = INTERP_MAX_GAP,
) -> tuple:
    """Linearly interpolate NaN runs whose duration is ≤ *max_gap*.

    Longer gaps are left as NaN so the forecasting model can handle them
    explicitly.

    Parameters
    ----------
    series : pd.Series
        Time-indexed series (must have a uniform DatetimeIndex).
    max_gap : pd.Timedelta
        Maximum consecutive gap duration to fill (default 2 h).

    Returns
    -------
    tuple[pd.Series, int]
        Gap-filled series and number of previously-NaN points filled.
    """
    series = series.copy()
    na_before = int(series.isna().sum())

    diffs = series.index.to_series().diff().dropna()
    if diffs.empty:
        return series, 0
    mode_vals = diffs.mode()
    freq = mode_vals.iloc[0] if not mode_vals.empty else diffs.median()

    is_nan = series.isna()
    nan_groups = (is_nan != is_nan.shift()).cumsum()
    run_lengths = is_nan.groupby(nan_groups).transform("sum")
    run_durations = run_lengths * freq
    short_gap_mask = is_nan & (run_durations <= max_gap)

    interpolated = series.interpolate(method="time")
    series.loc[short_gap_mask] = interpolated.loc[short_gap_mask]

    filled = na_before - int(series.isna().sum())
    return series, filled


# ─── gap analysis ─────────────────────────────────────────────────────────────

def gap_summary(series: pd.Series) -> pd.DataFrame:
    """Return a DataFrame describing every NaN gap in *series*.

    Columns: ``start``, ``end``, ``duration``, ``n_missing``.
    """
    is_nan = series.isna()
    transitions = is_nan.astype(int).diff().fillna(0)
    gap_starts = series.index[transitions == 1]
    gap_ends = series.index[transitions == -1]

    if is_nan.iloc[0]:
        gap_starts = pd.Index([series.index[0]]).append(gap_starts)
    if is_nan.iloc[-1]:
        gap_ends = gap_ends.append(pd.Index([series.index[-1]]))

    records = []
    for s, e in zip(gap_starts, gap_ends):
        n = int(is_nan.loc[s:e].sum())
        records.append({"start": s, "end": e, "duration": e - s, "n_missing": n})
    return pd.DataFrame(records)


# ─── diagnostic plot ──────────────────────────────────────────────────────────

def plot_ground_truth(
    series: pd.Series,
    output_path: str = "ChlRFUShallow_RFU_GroundTruth.png",
) -> None:
    """Save a multi-panel overview of the ground-truth series (one panel per year)."""
    years = sorted(series.index.year.unique())
    n_years = len(years)
    fig, axes = plt.subplots(n_years, 1, figsize=(14, 2.5 * n_years), sharex=False)
    if n_years == 1:
        axes = [axes]

    for ax, yr in zip(axes, years):
        yr_data = series[series.index.year == yr]
        ax.plot(yr_data.index, yr_data.values, color="steelblue", linewidth=0.7)
        ax.set_ylabel("RFU", fontsize=8)
        ax.set_title(str(yr), fontsize=9, loc="left")
        ax.tick_params(axis="x", labelsize=7)
        valid_pct = 100 * yr_data.notna().sum() / len(yr_data)
        ax.text(
            0.99, 0.88,
            f"valid: {valid_pct:.1f} %",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            color="dimgray",
        )

    fig.suptitle("ChlRFUShallow_RFU — Ground Truth (2014-2021)", fontsize=11, y=1.005)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Ground-truth overview plot saved → {output_path}")


# ─── backward-compatible single-year helper ───────────────────────────────────

def preprocess_2014_data(
    input_path: str = "FRDR_dataset_1095/BPBuoyData_2014_Cleaned.csv",
    output_path: str = "FRDR_dataset_1095/BPBuoyData_2014_Preprocessed.csv",
    flag_value: str = "B7",
) -> int:
    """Remove biofouling-affected chlorophyll measurements from the 2014 dataset.

    Rows whose ``ChlRFUShallow_RFU_Flag`` equals *flag_value* (default ``'B7'``)
    have their ``ChlRFUShallow_RFU`` value replaced with ``NaN``.  All other
    columns and rows are preserved unchanged.

    .. note::
        This function is retained for backward compatibility.
        Use :func:`run_full_pipeline` for the complete multi-year pipeline.

    Parameters
    ----------
    input_path : str
        Path to the raw/cleaned 2014 CSV file.
    output_path : str
        Destination path for the preprocessed CSV file.
    flag_value : str
        Quality-flag code that identifies biofouling periods (default ``'B7'``).

    Returns
    -------
    int
        Number of rows whose chlorophyll value was set to ``NaN``.
    """
    df = pd.read_csv(input_path, parse_dates=["DateTime"])

    b7_mask = df[FLAG_COL] == flag_value
    affected_count = int(b7_mask.sum())

    if affected_count > 0:
        df.loc[b7_mask, TARGET_COL] = np.nan
        affected_rows = df.loc[b7_mask, "DateTime"]
        print(f"Flag '{flag_value}' detected in {affected_count} rows.")
        print(f"  Affected period: {affected_rows.min()} → {affected_rows.max()}")
    else:
        print(f"No rows with flag '{flag_value}' found.")

    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    print(f"Total rows: {len(df)}, rows with chlorophyll set to NaN: {affected_count}")
    return affected_count


def plot_preprocessed_2014() -> None:
    """Plot the preprocessed 2014 chlorophyll series."""
    data_path = "FRDR_dataset_1095/BPBuoyData_2014_Preprocessed.csv"
    df = pd.read_csv(data_path, parse_dates=["DateTime"])

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="DateTime", y=TARGET_COL, color="blue")
    plt.title("ChlRFUShallow_RFU in 2014 (Preprocessed)")
    plt.xlabel("DateTime")
    plt.ylabel(TARGET_COL)
    plt.tight_layout()
    plt.savefig("ChlRFUShallow_RFU_2014_Preprocessed.png")
    plt.close()
    print("Plot saved as ChlRFUShallow_RFU_2014_Preprocessed.png")


# ─── full pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(
    data_dir: str = DATA_DIR,
    output_dir: str = DATA_DIR,
    ground_truth_path: str = "ChlRFUShallow_RFU_GroundTruth.csv",
) -> pd.Series:
    """Run the complete preprocessing pipeline for ChlRFUShallow_RFU.

    Executes all steps described in the module docstring.

    Parameters
    ----------
    data_dir : str
        Directory containing the raw ``BPBuoyData_*_Cleaned.csv`` files.
    output_dir : str
        Directory for the per-year ``*_Preprocessed.csv`` outputs.
    ground_truth_path : str
        Destination path for the combined ground-truth CSV.

    Returns
    -------
    pd.Series
        The final cleaned, regularly-sampled, gap-filled ground-truth series
        with a DatetimeIndex at 10-minute resolution.
    """
    print("=" * 60)
    print("ChlRFUShallow_RFU Ground-Truth Preprocessing Pipeline")
    print("=" * 60)

    # ── 1. Load all years ────────────────────────────────────────
    print("\n[1] Loading data …")
    df_all = load_all_years(data_dir)
    print(
        f"    Loaded {len(df_all):,} rows  |  "
        f"{df_all['DateTime'].min().date()} → {df_all['DateTime'].max().date()}"
    )

    # ── 2. Apply quality flags ───────────────────────────────────
    print(f"\n[2] Applying quality flags  (invalidating: {', '.join(sorted(INVALID_FLAGS))}) …")
    df_all, n_flagged = apply_quality_flags(df_all)
    print(f"    {n_flagged:,} rows invalidated by quality flags.")

    # ── 3. Physical bounds ───────────────────────────────────────
    print("\n[3] Physical-range validation  (ChlRFU ≥ 0) …")
    df_all, n_neg = apply_physical_bounds(df_all)
    print(f"    {n_neg:,} sub-zero values removed.")

    # ── 4. Per-year preprocessed CSV exports ─────────────────────
    print("\n[4] Exporting per-year preprocessed CSVs …")
    for year in YEARS:
        yr_df = df_all[df_all["Year"] == year].copy()
        out_path = os.path.join(output_dir, f"BPBuoyData_{year}_Preprocessed.csv")
        yr_df.to_csv(out_path, index=False)
        valid = int(yr_df[TARGET_COL].notna().sum())
        total = len(yr_df)
        print(f"    {year}: {total:,} rows, {valid:,} valid ({100 * valid / total:.1f} %)  → {out_path}")

    # ── Prepare time-indexed series for remaining steps ──────────
    chl = df_all.set_index("DateTime")[TARGET_COL].sort_index()

    # ── 5. Statistical outlier removal ───────────────────────────
    print(f"\n[5] Statistical outlier removal  (rolling {ROLLING_WINDOW} IQR, k={OUTLIER_IQR_K}) …")
    chl, n_outliers = remove_statistical_outliers(chl)
    print(f"    {n_outliers:,} statistical outliers removed.")

    # ── 6. Resample to regular 10-minute grid ────────────────────
    print(f"\n[6] Resampling to {RESAMPLE_FREQ} regular grid …")
    chl = resample_to_regular_grid(chl)
    n_missing = int(chl.isna().sum())
    total_bins = len(chl)
    print(f"    {total_bins:,} bins  |  {n_missing:,} missing  ({100 * n_missing / total_bins:.1f} %)")

    # ── 7. Interpolate short gaps ────────────────────────────────
    print(f"\n[7] Interpolating gaps ≤ {INTERP_MAX_GAP} …")
    chl, n_filled = interpolate_short_gaps(chl)
    n_remaining = int(chl.isna().sum())
    print(
        f"    {n_filled:,} points filled  |  "
        f"{n_remaining:,} missing points remain  ({100 * n_remaining / total_bins:.1f} %)"
    )

    # ── Gap summary ──────────────────────────────────────────────
    gaps = gap_summary(chl)
    if not gaps.empty:
        print(f"\n    Remaining gaps: {len(gaps)}")
        top5 = gaps.nlargest(5, "n_missing")[["start", "end", "duration", "n_missing"]]
        print("    Longest gaps:")
        print(top5.to_string(index=False))

    # ── 8. Export ground truth ───────────────────────────────────
    print(f"\n[8] Saving ground-truth series → {ground_truth_path} …")
    out = chl.rename(TARGET_COL + "_GroundTruth").to_frame()
    out.to_csv(ground_truth_path)
    print(f"    {len(out):,} rows saved.")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    return chl


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ground_truth = run_full_pipeline()
    plot_ground_truth(ground_truth)
    plot_preprocessed_2014()
