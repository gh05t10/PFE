import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

DATA_DIR = "FRDR_dataset_1095"
YEARS = list(range(2014, 2022))
TARGET_COL = "ChlRFUShallow_RFU"


def load_preprocessed_year(year: int, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load the preprocessed CSV for *year* (30-min grid by default).

    Parameters
    ----------
    year : int
        Study year (2014–2021).
    data_dir : str
        Folder containing ``BPBuoyData_{year}_Preprocessed.csv`` files.

    Returns
    -------
    pd.DataFrame
        DataFrame with DateTime parsed and set as index.

    Raises
    ------
    FileNotFoundError
        If the preprocessed file does not exist for *year*.
    """
    fname = f"BPBuoyData_{year}_Preprocessed.csv"
    path  = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed file not found: {path}")
    df = pd.read_csv(path, parse_dates=["DateTime"])
    df = df.set_index("DateTime").sort_index()
    return df


def _get_nan_spans(series: pd.Series, gap_threshold: pd.Timedelta) -> list[tuple]:
    """Return list of (start, end) datetime pairs for consecutive NaN runs.

    Consecutive NaN blocks separated by more than *gap_threshold* are
    treated as distinct spans.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex.
    gap_threshold : pd.Timedelta
        Maximum index gap within a single NaN span.

    Returns
    -------
    list of (start, end) tuples
        Each tuple marks the datetime boundaries of one NaN span.
    """
    nan_times = series.index[series.isna()]
    if len(nan_times) == 0:
        return []

    spans = []
    start = nan_times[0]
    prev  = nan_times[0]
    for t in nan_times[1:]:
        if t - prev > gap_threshold:
            spans.append((start, prev))
            start = t
        prev = t
    spans.append((start, prev))
    return spans


def plot_preprocessed_year(
    year: int,
    data_dir: str = DATA_DIR,
    output_dir: str = ".",
) -> None:
    """Plot ChlRFUShallow_RFU for one preprocessed year.

    The chart mirrors the style of the existing ``figure{year}.png`` files:
    a brown line for valid measurements and red-shaded bands where NaN
    values remain after preprocessing (i.e. long gaps that were not
    interpolated).

    Parameters
    ----------
    year : int
        Study year to plot (2014–2021).
    data_dir : str
        Folder containing the preprocessed CSVs.
    output_dir : str
        Folder where the PNG is saved.
    """
    try:
        df = load_preprocessed_year(year, data_dir)
    except FileNotFoundError as exc:
        print(f"[SKIP] {exc}")
        return

    series = df[TARGET_COL]

    fig, ax = plt.subplots(figsize=(12, 4))

    # --- Line plot (valid values only) ---
    ax.plot(series.index, series.values, color="brown", linewidth=0.7,
            label="Chlorophyll RFU (preprocessed)")

    # --- Red bands for remaining NaN gaps (long gaps not interpolated) ---
    nan_spans = _get_nan_spans(series, gap_threshold=pd.Timedelta("45min"))
    for s, e in nan_spans:
        ax.axvspan(s, e, color="red", alpha=0.25)

    # Build legend
    handles = [
        plt.Line2D([0], [0], color="brown", linewidth=0.7,
                   label="Shallow Chlorophyll (RFU)"),
    ]
    if nan_spans:
        handles.append(
            mpatches.Patch(color="red", alpha=0.4, label="NaN gap (long, not interpolated)")
        )
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    # Formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    ax.set_title(f"ChlRFUShallow_RFU {year} (Preprocessed — 30 min)", fontsize=11)
    ax.set_xlabel("DateTime")
    ax.set_ylabel("ChlRFUShallow_RFU [RFU]")
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"figure{year}_preprocessed.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_all_years(
    data_dir: str = DATA_DIR,
    output_dir: str = ".",
) -> None:
    """Plot preprocessed ChlRFUShallow_RFU for all years (2014–2021).

    Generates one PNG per year named ``figure{year}_preprocessed.png``.

    Parameters
    ----------
    data_dir : str
        Folder containing the preprocessed CSVs.
    output_dir : str
        Folder where PNGs are saved (created if it does not exist).
    """
    for year in YEARS:
        print(f"Plotting {year} ...")
        plot_preprocessed_year(year, data_dir=data_dir, output_dir=output_dir)


def main() -> None:
    """Entry point: generate preprocessed plots for all years."""
    plot_all_years(data_dir=DATA_DIR, output_dir=".")


if __name__ == "__main__":
    main()