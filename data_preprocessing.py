"""Complete time-series preprocessing pipeline for Buffalo Pound Lake chlorophyll data.

This module provides a full pipeline to load, quality-control, resample, interpolate,
feature-engineer, normalise, and split the 2014–2021 high-frequency buoy dataset into
training, validation, and test sets ready for forecasting.
"""

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = "FRDR_dataset_1095"
TARGET_COL = "ChlRFUShallow_RFU"
FLAG_COL = "ChlRFUShallow_RFU_Flag"
BAD_FLAGS = {"B7", "C", "M"}
YEARS = list(range(2014, 2022))


# ---------------------------------------------------------------------------
# A. Load all years
# ---------------------------------------------------------------------------

def load_all_years(
    data_dir: str = DATA_DIR,
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """Read all eight annual CSV files and return them as a dict and a combined DataFrame.

    Parameters
    ----------
    data_dir : str
        Directory that contains the ``BPBuoyData_{YEAR}_Cleaned.csv`` files.

    Returns
    -------
    year_dict : dict[int, pd.DataFrame]
        Mapping from year (int) to the corresponding raw DataFrame.
    combined : pd.DataFrame
        All years concatenated into a single DataFrame with an added ``year`` column.

    Raises
    ------
    FileNotFoundError
        If no annual CSV files are found in *data_dir*.
    """
    year_dict: Dict[int, pd.DataFrame] = {}
    frames = []

    for year in YEARS:
        path = os.path.join(data_dir, f"BPBuoyData_{year}_Cleaned.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found – skipping {year}.")
            continue
        df = pd.read_csv(path, parse_dates=["DateTime"], low_memory=False)
        df["year"] = year
        year_dict[year] = df
        frames.append(df)
        print(f"  Loaded {year}: {len(df)} rows")

    if not frames:
        raise FileNotFoundError(f"No annual CSV files found in '{data_dir}'.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows across all years: {len(combined)}")
    return year_dict, combined


# ---------------------------------------------------------------------------
# B. Apply quality flags
# ---------------------------------------------------------------------------

def apply_quality_flags(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Set ChlRFUShallow_RFU to NaN wherever its flag indicates bad data.

    Flags treated as bad: ``B7`` (biofouling), ``C`` (faulty), ``M`` (missing).
    All flag columns are preserved so the provenance of each removal can be traced.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame containing ``ChlRFUShallow_RFU`` and ``ChlRFUShallow_RFU_Flag``.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of *df* with bad-quality target values replaced by ``NaN``.
    stats : dict[str, int]
        Number of rows affected per flag code.
    """
    df_out = df.copy()
    stats: Dict[str, int] = {}

    for flag in BAD_FLAGS:
        mask = df_out[FLAG_COL] == flag
        count = int(mask.sum())
        stats[flag] = count
        if count > 0:
            df_out.loc[mask, TARGET_COL] = np.nan
            print(f"  Flag '{flag}': {count} rows set to NaN")
        else:
            print(f"  Flag '{flag}': 0 rows affected")

    total = sum(stats.values())
    print(f"  Total rows set to NaN by quality flags: {total}")
    return df_out, stats


# ---------------------------------------------------------------------------
# C. Resample to regular grid
# ---------------------------------------------------------------------------

def resample_to_regular_grid(
    df: pd.DataFrame, freq: str = "15min"
) -> pd.DataFrame:
    """Resample the DataFrame to a regular time grid by averaging within each interval.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DateTime`` column (not yet set as index).
    freq : str
        Pandas offset alias for the target frequency (default ``'15min'``).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with ``DateTime`` as the index and numeric columns
        aggregated by mean.  The ``year`` and flag columns are forward-filled
        after resampling so that metadata is retained.
    """
    df_ts = df.copy()
    df_ts = df_ts.set_index("DateTime").sort_index()

    # Separate numeric and non-numeric columns
    numeric_cols = df_ts.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df_ts.columns if c not in numeric_cols]

    df_num = df_ts[numeric_cols].resample(freq).mean()

    # For flag / string columns keep the most frequent value in each bin
    if non_numeric_cols:
        df_cat = (
            df_ts[non_numeric_cols]
            .resample(freq)
            .first()
        )
        df_resampled = pd.concat([df_num, df_cat], axis=1)
    else:
        df_resampled = df_num

    # Restore year as integer derived from the DatetimeIndex (avoids cross-boundary propagation)
    if "year" in df_resampled.columns:
        df_resampled["year"] = df_resampled.index.year

    print(f"  Resampled to {freq}: {len(df_resampled)} rows")
    return df_resampled


# ---------------------------------------------------------------------------
# D. Handle missing values
# ---------------------------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame, max_gap_hours: float = 6.0
) -> pd.DataFrame:
    """Linearly interpolate small gaps in ChlRFUShallow_RFU; leave large gaps as NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``DateTime`` as the index (output of
        :func:`resample_to_regular_grid`).
    max_gap_hours : float
        Maximum gap length (in hours) to fill by linear interpolation.
        Gaps longer than this are left as ``NaN``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with small gaps filled in ``ChlRFUShallow_RFU``.
    """
    df_out = df.copy()
    col = TARGET_COL

    nan_before = int(df_out[col].isna().sum())

    # Determine the time step in hours
    if len(df_out) >= 2:
        time_diffs = df_out.index.to_series().diff().dropna()
        median_step_hours = time_diffs.median().total_seconds() / 3600.0
    else:
        median_step_hours = 0.25  # fallback: 15 min

    max_gap_steps = int(np.ceil(max_gap_hours / median_step_hours))

    # Identify contiguous NaN runs and their lengths
    is_nan = df_out[col].isna()
    gap_lengths = []
    in_gap = False
    gap_len = 0
    for v in is_nan:
        if v:
            in_gap = True
            gap_len += 1
        else:
            if in_gap:
                gap_lengths.append(gap_len)
                gap_len = 0
                in_gap = False
    if in_gap:
        gap_lengths.append(gap_len)

    # Linear interpolation limited to max_gap_steps
    df_out[col] = df_out[col].interpolate(
        method="linear", limit=max_gap_steps, limit_direction="forward"
    )

    nan_after = int(df_out[col].isna().sum())

    print(f"  NaN before interpolation : {nan_before}")
    print(f"  NaN after  interpolation : {nan_after}")
    print(f"  Rows filled              : {nan_before - nan_after}")
    print(f"  Number of detected gaps  : {len(gap_lengths)}")
    if gap_lengths:
        print(f"  Gap length distribution  : "
              f"min={min(gap_lengths)}, max={max(gap_lengths)}, "
              f"mean={np.mean(gap_lengths):.1f} steps")

    return df_out


# ---------------------------------------------------------------------------
# E. Compute time features
# ---------------------------------------------------------------------------

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical and categorical time features derived from the DatetimeIndex.

    New columns added
    -----------------
    ``day_of_year``, ``hour_of_day``, ``week_of_year``, ``month``,
    ``sin_doy``, ``cos_doy``, ``sin_hod``, ``cos_hod``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the additional time-feature columns appended.
    """
    df_out = df.copy()
    idx = df_out.index

    df_out["day_of_year"] = idx.day_of_year
    df_out["hour_of_day"] = idx.hour
    df_out["week_of_year"] = idx.isocalendar().week.astype(int)
    df_out["month"] = idx.month

    df_out["sin_doy"] = np.sin(2 * np.pi * df_out["day_of_year"] / 366)
    df_out["cos_doy"] = np.cos(2 * np.pi * df_out["day_of_year"] / 366)
    df_out["sin_hod"] = np.sin(2 * np.pi * df_out["hour_of_day"] / 24)
    df_out["cos_hod"] = np.cos(2 * np.pi * df_out["hour_of_day"] / 24)

    print(f"  Time features added: day_of_year, hour_of_day, week_of_year, "
          f"month, sin_doy, cos_doy, sin_hod, cos_hod")
    return df_out


# ---------------------------------------------------------------------------
# F. Normalise target
# ---------------------------------------------------------------------------

def normalize_target(
    df: pd.DataFrame,
    method: str = "robust",
    scaler_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, RobustScaler]:
    """Normalise ``ChlRFUShallow_RFU`` using a RobustScaler and persist parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``ChlRFUShallow_RFU``.
    method : str
        Normalisation method; currently only ``'robust'`` is supported.
    scaler_path : str or None
        Destination JSON file for the scaler parameters (median and IQR).
        Defaults to ``FRDR_dataset_1095/chl_scaler_params.json``.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of *df* with an additional ``ChlRFUShallow_RFU_normalized`` column.
    scaler : RobustScaler
        Fitted scaler instance.
    """
    if scaler_path is None:
        scaler_path = os.path.join(DATA_DIR, "chl_scaler_params.json")
    df_out = df.copy()
    col = TARGET_COL

    valid = df_out[col].dropna().values.reshape(-1, 1)
    scaler = RobustScaler()
    scaler.fit(valid)

    df_out[col + "_normalized"] = scaler.transform(
        df_out[col].values.reshape(-1, 1)
    ).flatten()

    params = {
        "median": float(scaler.center_[0]),
        "iqr": float(scaler.scale_[0]),
    }
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"  RobustScaler fitted: median={params['median']:.4f}, IQR={params['iqr']:.4f}")
    print(f"  Scaler parameters saved to {scaler_path}")
    return df_out, scaler


# ---------------------------------------------------------------------------
# G. Train / validation / test split
# ---------------------------------------------------------------------------

def create_train_val_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset by year into train (2014–2019), val (2020), test (2021).

    The split is strictly temporal (no shuffling).

    Parameters
    ----------
    df : pd.DataFrame
        Fully processed DataFrame with a ``DatetimeIndex``.

    Returns
    -------
    df_train, df_val, df_test : tuple of pd.DataFrame
    """
    df_train = df[df.index.year <= 2019]
    df_val = df[df.index.year == 2020]
    df_test = df[df.index.year == 2021]

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        if split.empty:
            print(f"  {name}: 0 samples (no data)")
        else:
            print(
                f"  {name}: {len(split)} samples | "
                f"{split.index.min().date()} → {split.index.max().date()}"
            )

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# H. Preprocessing report
# ---------------------------------------------------------------------------

def generate_preprocessing_report(
    df_raw: pd.DataFrame,
    df_processed: pd.DataFrame,
    report_path: str = os.path.join(DATA_DIR, "preprocessing_report.txt"),
) -> None:
    """Write a human-readable preprocessing summary to *report_path*.

    The report includes row counts, NaN statistics, per-year flag distributions,
    descriptive statistics for ``ChlRFUShallow_RFU``, and the measurement period
    of each year.

    Parameters
    ----------
    df_raw : pd.DataFrame
        The combined raw DataFrame (before any processing).
    df_processed : pd.DataFrame
        The fully processed DataFrame (with DatetimeIndex).
    report_path : str
        Destination path for the text report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("PREPROCESSING REPORT — Buffalo Pound Lake Chlorophyll (2014–2021)")
    lines.append("=" * 70)

    lines.append(f"\n[1] Row counts")
    lines.append(f"  Raw rows       : {len(df_raw)}")
    lines.append(f"  Processed rows : {len(df_processed)}")

    lines.append(f"\n[2] NaN statistics for {TARGET_COL}")
    raw_nan = int(df_raw[TARGET_COL].isna().sum()) if TARGET_COL in df_raw.columns else "N/A"
    proc_nan = int(df_processed[TARGET_COL].isna().sum())
    lines.append(f"  NaN in raw data       : {raw_nan}")
    lines.append(f"  NaN after processing  : {proc_nan}")

    lines.append(f"\n[3] Flag distribution per year (raw data)")
    if FLAG_COL in df_raw.columns:
        raw_copy = df_raw.copy()
        # Derive year from DateTime if not present
        if "year" not in raw_copy.columns:
            raw_copy["year"] = pd.to_datetime(raw_copy["DateTime"]).dt.year
        flag_pivot = (
            raw_copy.groupby(["year", FLAG_COL])
            .size()
            .unstack(fill_value=0)
        )
        lines.append(flag_pivot.to_string())
    else:
        lines.append("  Flag column not found.")

    lines.append(f"\n[4] Descriptive statistics for {TARGET_COL} (processed)")
    desc = df_processed[TARGET_COL].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    lines.append(desc.to_string())

    lines.append(f"\n[5] Measurement period per year")
    for year in YEARS:
        subset = df_processed[df_processed.index.year == year]
        if subset.empty:
            lines.append(f"  {year}: no data")
        else:
            lines.append(
                f"  {year}: {subset.index.min().date()} → {subset.index.max().date()} "
                f"({len(subset)} samples)"
            )

    lines.append("\n" + "=" * 70)
    report_text = "\n".join(lines)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to {report_path}")


# ---------------------------------------------------------------------------
# I. Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the full preprocessing pipeline and save all outputs."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    year_dict, df_raw = load_all_years(DATA_DIR)

    print("\n" + "=" * 60)
    print("STEP 2: Applying quality flags")
    print("=" * 60)
    df_flagged, flag_stats = apply_quality_flags(df_raw)

    print("\n" + "=" * 60)
    print("STEP 3: Resampling to 15-minute grid")
    print("=" * 60)
    df_resampled = resample_to_regular_grid(df_flagged, freq="15min")

    print("\n" + "=" * 60)
    print("STEP 4: Handling missing values (max gap = 6 h)")
    print("=" * 60)
    df_interp = handle_missing_values(df_resampled, max_gap_hours=6)

    print("\n" + "=" * 60)
    print("STEP 5: Computing time features")
    print("=" * 60)
    df_features = compute_time_features(df_interp)

    print("\n" + "=" * 60)
    print("STEP 6: Normalising target")
    print("=" * 60)
    df_norm, scaler = normalize_target(
        df_features,
        scaler_path=os.path.join(DATA_DIR, "chl_scaler_params.json"),
    )

    print("\n" + "=" * 60)
    print("STEP 7: Creating train / val / test splits")
    print("=" * 60)
    df_train, df_val, df_test = create_train_val_test_split(df_norm)

    print("\n" + "=" * 60)
    print("STEP 8: Saving outputs")
    print("=" * 60)
    all_path = os.path.join(DATA_DIR, "ChlData_AllYears_Preprocessed.csv")
    df_norm.to_csv(all_path)
    print(f"  Saved: {all_path}")

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        path = os.path.join(DATA_DIR, f"ChlData_{name}.csv")
        split.to_csv(path)
        print(f"  Saved: {path}")

    print("\n" + "=" * 60)
    print("STEP 9: Generating preprocessing report")
    print("=" * 60)
    generate_preprocessing_report(
        df_raw,
        df_norm,
        report_path=os.path.join(DATA_DIR, "preprocessing_report.txt"),
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
