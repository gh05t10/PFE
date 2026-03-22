"""Visualisation utilities for Buffalo Pound Lake chlorophyll time-series data.

Each public function saves its figure to the ``figures/`` directory (created
automatically) and calls ``plt.close()`` to free memory.
"""

import os
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = "FRDR_dataset_1095"
FIGURES_DIR = "figures"
TARGET_COL = "ChlRFUShallow_RFU"
FLAG_COL = "ChlRFUShallow_RFU_Flag"
BAD_FLAGS = {"B7", "C", "M"}
FLAG_COLORS = {"B7": "#e74c3c", "C": "#e67e22", "M": "#9b59b6"}
YEARS = list(range(2014, 2022))


def _ensure_figures_dir() -> None:
    """Create the figures output directory if it does not already exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def load_all_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Read every BPBuoyData_*_Cleaned.csv file in *data_dir* and concatenate.

    The function parses the ``DateTime`` column as a pandas datetime and
    adds a ``Year`` column so that plots can be grouped by year.
    """
    frames: list[pd.DataFrame] = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith("BPBuoyData") and fname.endswith("_Cleaned.csv"):
            path = os.path.join(data_dir, fname)
            df = pd.read_csv(path, parse_dates=["DateTime"], low_memory=False)
            df["Year"] = df["DateTime"].dt.year
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def list_variables(df: pd.DataFrame) -> list[str]:
    """Return a list of non-flag, non-datetime columns that can be plotted."""
    drop = {"DateTime", "Year"}
    return [c for c in df.columns if c not in drop and not c.endswith("_Flag")]


def read_flag_definitions(path: str) -> dict[str, str]:
    """Read *path* (CSV) and return a mapping of flag code to its description."""
    flags: dict[str, str] = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("Flag"):
                    continue
                parts = line.split(",", 1)
                if len(parts) == 2:
                    key = parts[0].strip().strip('"')
                    val = parts[1].strip().strip('"')
                    flags[key] = val
    except FileNotFoundError:
        pass
    return flags


# ---------------------------------------------------------------------------
# A. Raw vs preprocessed comparison per year
# ---------------------------------------------------------------------------

def plot_raw_vs_preprocessed(year: int, data_dir: str = DATA_DIR) -> None:
    """Compare raw and preprocessed chlorophyll for a single *year*.

    Points removed by quality flags (B7, C, M) are overlaid in distinct
    colours so the reader can see where and how much data was discarded.

    Parameters
    ----------
    year : int
        Calendar year (2014–2021).
    data_dir : str
        Directory containing both the raw ``*_Cleaned.csv`` and the
        preprocessed ``ChlData_AllYears_Preprocessed.csv`` file.
    """
    _ensure_figures_dir()
    raw_path = os.path.join(data_dir, f"BPBuoyData_{year}_Cleaned.csv")
    preprocessed_path = os.path.join(data_dir, "ChlData_AllYears_Preprocessed.csv")

    if not os.path.exists(raw_path):
        print(f"Raw file not found: {raw_path}")
        return

    df_raw = pd.read_csv(raw_path, parse_dates=["DateTime"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- Top panel: raw data with flag highlights ---
    ax = axes[0]
    ax.plot(df_raw["DateTime"], df_raw[TARGET_COL], color="steelblue",
            linewidth=0.7, label="Raw")
    for flag, color in FLAG_COLORS.items():
        mask = df_raw[FLAG_COL] == flag
        if mask.any():
            ax.scatter(
                df_raw.loc[mask, "DateTime"],
                df_raw.loc[mask, TARGET_COL],
                color=color, s=6, zorder=3,
                label=f"Flag {flag}",
            )
    ax.set_title(f"{year} — Raw data with quality flags")
    ax.set_ylabel("ChlRFUShallow (RFU)")
    ax.legend(loc="upper right", fontsize=8)

    # --- Bottom panel: preprocessed data ---
    ax2 = axes[1]
    if os.path.exists(preprocessed_path):
        df_proc = pd.read_csv(preprocessed_path, parse_dates=["DateTime"],
                              index_col="DateTime")
        subset = df_proc[df_proc.index.year == year]
        ax2.plot(subset.index, subset[TARGET_COL], color="forestgreen",
                 linewidth=0.7, label="Preprocessed")
        ax2.set_title(f"{year} — After quality filtering & interpolation")
        ax2.set_ylabel("ChlRFUShallow (RFU)")
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax2.set_title(f"{year} — Preprocessed file not found")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"raw_vs_preprocessed_{year}.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# B. All-years overview
# ---------------------------------------------------------------------------

def plot_all_years_overview(df: pd.DataFrame) -> None:
    """Plot ChlRFUShallow_RFU for all years in one continuous figure.

    Vertical dashed lines mark the boundary between consecutive years.
    ``df`` must have a ``DatetimeIndex``.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame with a ``DatetimeIndex``.
    """
    _ensure_figures_dir()
    fig, ax = plt.subplots(figsize=(20, 5))

    ax.plot(df.index, df[TARGET_COL], color="steelblue", linewidth=0.5, alpha=0.8)

    # Year boundary lines
    for year in YEARS[1:]:
        boundary = pd.Timestamp(f"{year}-01-01")
        if df.index.min() < boundary < df.index.max():
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(boundary, ax.get_ylim()[1], str(year),
                    ha="left", va="top", fontsize=8, color="gray")

    ax.set_title("ChlRFUShallow_RFU — All years (2014–2021)")
    ax.set_xlabel("Date")
    ax.set_ylabel("ChlRFUShallow (RFU)")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "all_years_overview.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# C. Missing-data heatmap
# ---------------------------------------------------------------------------

def plot_missing_data_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of the fraction of missing ChlRFUShallow_RFU by year × month.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` (may include NaN in target column).
    """
    _ensure_figures_dir()

    df_copy = df[[TARGET_COL]].copy()
    df_copy["year"] = df_copy.index.year
    df_copy["month"] = df_copy.index.month
    df_copy["is_nan"] = df_copy[TARGET_COL].isna().astype(int)

    pivot = df_copy.groupby(["year", "month"])["is_nan"].mean().unstack(fill_value=np.nan)
    pivot.columns = [
        pd.Timestamp(f"2000-{m}-01").strftime("%b") for m in pivot.columns
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.4,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Fraction missing"},
        vmin=0,
        vmax=1,
    )
    ax.set_title("Missing-data fraction — ChlRFUShallow_RFU (by year & month)")
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "missing_data_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# D. Seasonal patterns
# ---------------------------------------------------------------------------

def plot_seasonal_patterns(df: pd.DataFrame) -> None:
    """Violin plot of ChlRFUShallow_RFU by month, with separate colour per year.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex``.
    """
    _ensure_figures_dir()

    df_copy = df[[TARGET_COL]].dropna().copy()
    df_copy["month"] = df_copy.index.month
    df_copy["year"] = df_copy.index.year
    df_copy["month_name"] = df_copy["month"].apply(
        lambda m: pd.Timestamp(f"2000-{m}-01").strftime("%b")
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    present_months = sorted(df_copy["month"].unique())
    sns.boxplot(
        data=df_copy,
        x="month_name",
        y=TARGET_COL,
        hue="year",
        ax=ax,
        palette="tab10",
        order=[pd.Timestamp(f"2000-{m}-01").strftime("%b") for m in present_months],
        flierprops={"marker": ".", "markersize": 2},
        linewidth=0.7,
    )
    ax.set_title("Seasonal distribution of ChlRFUShallow_RFU by month and year")
    ax.set_xlabel("Month")
    ax.set_ylabel("ChlRFUShallow (RFU)")
    ax.legend(title="Year", loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "seasonal_patterns.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# E. Train / val / test split visualisation
# ---------------------------------------------------------------------------

def plot_train_val_test_split(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    """Visualise the temporal train / validation / test split in one figure.

    Each partition is shown in a distinct colour against a shared time axis.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training set (DatetimeIndex, 2014–2019).
    df_val : pd.DataFrame
        Validation set (DatetimeIndex, 2020).
    df_test : pd.DataFrame
        Test set (DatetimeIndex, 2021).
    """
    _ensure_figures_dir()

    fig, ax = plt.subplots(figsize=(20, 5))

    splits = [
        (df_train, "Train (2014–2019)", "#2196F3"),
        (df_val,   "Val   (2020)",      "#FF9800"),
        (df_test,  "Test  (2021)",      "#F44336"),
    ]
    for split_df, label, color in splits:
        if not split_df.empty:
            ax.plot(split_df.index, split_df[TARGET_COL],
                    color=color, linewidth=0.6, label=label, alpha=0.9)

    ax.set_title("ChlRFUShallow_RFU — Train / Validation / Test split")
    ax.set_xlabel("Date")
    ax.set_ylabel("ChlRFUShallow (RFU)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "train_val_test_split.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Legacy: per-year flag-band plots (kept for backward compatibility)
# ---------------------------------------------------------------------------

def plot_with_flag_bands(
    df: pd.DataFrame,
    variable: str,
    flag_column: str,
    year: int,
    flag_defs: Optional[dict[str, str]] = None,
) -> None:
    """Plot *variable* for one *year* adding background bands for *flag_column*.

    ``flag_defs`` may be a mapping from flag code to description; it will be
    used to label the legend.  Figures are saved to the ``figures/`` directory.
    """
    _ensure_figures_dir()

    data = df[df["Year"] == year].sort_values("DateTime")
    if data.empty:
        print(f"no data for year {year}")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data["DateTime"], data[variable], color="brown", linewidth=0.7)

    fixed_flags = {
        "M":  "M - Missing data",
        "C":  "C - Faulty",
        "A2": "A2 - Low Light",
        "A3": "A3 - Inverted/Obstruction",
        "A4": "A4 - Shading",
        "B7": "B7 - Spike/Fouling",
        "A5": "A5 - Local Rain",
        "A6": "A6 - Conductivity Corrected",
        "B6": "B6 - Conductivity Step",
    }

    palette = sns.color_palette("bright", len(fixed_flags))
    color_map = {flag: palette[i] for i, flag in enumerate(fixed_flags)}

    present_flags = set(data[flag_column].dropna().unique())
    present_flags = {f for f in present_flags if f in fixed_flags}

    for flag in present_flags:
        color = color_map[flag]
        subset = data[data[flag_column] == flag]
        if subset.empty:
            continue
        times = subset["DateTime"].sort_values()
        spans = []
        start = times.iloc[0]
        prev = start
        for t in times.iloc[1:]:
            if t - prev > pd.Timedelta("1h"):
                spans.append((start, prev))
                start = t
            prev = t
        spans.append((start, prev))
        for s, e in spans:
            ax.axvspan(s, e, color=color, alpha=0.3)

    handles = [plt.Line2D([0], [0], color="brown", linewidth=0.7,
                           label="Shallow Chlorophyll (RFU)")]
    for flag in sorted(present_flags):
        handles.append(mpatches.Patch(color=color_map[flag], label=fixed_flags[flag]))
    ax.legend(handles=handles, loc="upper right")

    ax.set_title(f"{variable} ({year})")
    ax.set_xlabel("DateTime")
    ax.set_ylabel(variable)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"figure{year}.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all standard visualisations for the chlorophyll dataset."""
    _ensure_figures_dir()

    # Load raw data (used for legacy flag-band plots and raw vs preprocessed)
    df_raw = load_all_data()
    flag_defs = read_flag_definitions(os.path.join(DATA_DIR, "data_flags.csv"))

    print("columns available for plotting:", list_variables(df_raw))

    # Legacy per-year flag-band plots
    for yr in sorted(df_raw["Year"].unique()):
        plot_with_flag_bands(
            df_raw,
            variable=TARGET_COL,
            flag_column=FLAG_COL,
            year=yr,
            flag_defs=flag_defs,
        )

    # New visualisations that require the preprocessed dataset
    preprocessed_path = os.path.join(DATA_DIR, "ChlData_AllYears_Preprocessed.csv")
    if os.path.exists(preprocessed_path):
        df_proc = pd.read_csv(preprocessed_path, parse_dates=["DateTime"],
                              index_col="DateTime")

        plot_all_years_overview(df_proc)
        plot_missing_data_heatmap(df_proc)
        plot_seasonal_patterns(df_proc)

        # Raw vs preprocessed per year
        for yr in YEARS:
            plot_raw_vs_preprocessed(yr)

        # Train / val / test
        train_path = os.path.join(DATA_DIR, "ChlData_Train.csv")
        val_path = os.path.join(DATA_DIR, "ChlData_Val.csv")
        test_path = os.path.join(DATA_DIR, "ChlData_Test.csv")
        if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            df_train = pd.read_csv(train_path, parse_dates=["DateTime"],
                                   index_col="DateTime")
            df_val = pd.read_csv(val_path, parse_dates=["DateTime"],
                                 index_col="DateTime")
            df_test = pd.read_csv(test_path, parse_dates=["DateTime"],
                                  index_col="DateTime")
            plot_train_val_test_split(df_train, df_val, df_test)
        else:
            print("Split files not found — skipping train/val/test plot.")
    else:
        print(
            "Preprocessed file not found — run data_preprocessing.py first "
            "to enable the new visualisations."
        )


if __name__ == "__main__":
    main()