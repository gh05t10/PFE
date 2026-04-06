import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns

# directory containing the buoy CSVs
DATA_DIR = "FRDR_dataset_1095"
OUTPUT_DIR = "shallow-chlorophyll-viz"


def load_all_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Read every BPBuoyData_*_Cleaned.csv file in *data_dir* and concatenate.

    The function parses the ``DateTime`` column as a pandas datetime and
    adds a ``Year`` column so that plots can be grouped by year.
    """
    frames: list[pd.DataFrame] = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith("BPBuoyData") and fname.endswith("_Cleaned.csv"):
            path = os.path.join(data_dir, fname)
            df = pd.read_csv(path, parse_dates=["DateTime"], infer_datetime_format=True)
            df["Year"] = df["DateTime"].dt.year
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no CSV files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def list_variables(df: pd.DataFrame) -> list[str]:
    """Return a list of non-flag, non-datetime columns that can be plotted."""
    drop = {"DateTime", "Year"}
    return [c for c in df.columns if c not in drop and not c.endswith("_Flag")]


def plot_variable_by_year(df: pd.DataFrame, variable: str = "AirTemp_C") -> None:
    """Create a time-series line plot of *variable* with one colour per year."""
    if variable not in df.columns:
        raise KeyError(f"variable '{variable}' not found in dataframe")

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df,
        x="DateTime",
        y=variable,
        hue="Year",
        palette="tab10",
        linewidth=0.8,
        alpha=0.7,
    )
    plt.title(f"{variable} over time by year")
    plt.xlabel("Date")
    plt.ylabel(variable)
    plt.legend(title="Year", loc="best")
    plt.tight_layout()
    plt.show()


def read_flag_definitions(path: str) -> dict[str, str]:
    """Read *path* (CSV) and return a mapping of flag to its description."""
    flags = {}
    try:
        with open(path) as f:
            # simple parsing because file is two columns
            for line in f:
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


def plot_with_flag_bands(
    df: pd.DataFrame,
    variable: str,
    flag_column: str,
    year: int,
    flag_defs: dict[str, str] | None = None,
):
    """Plot *variable* for one *year* adding background bands for *flag_column*.

    ``flag_defs`` may be a mapping from flag code to description; it will be used
    to label the legend.
    """
    data = df[df["Year"] == year].sort_values("DateTime")
    if data.empty:
        print(f"no data for year {year}")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data["DateTime"], data[variable], color="brown", linewidth=0.7)

    # Define fixed flags and their descriptions
    fixed_flags = {
        'M': 'M - Missing data',
        'C': 'C - Faulty',
        'A2': 'A2 - Low Light',
        'A3': 'A3 - Inverted/Obstruction',
        'A4': 'A4 - Shading',
        'B7': 'B7 - Spike/Fouling',
        'A5': 'A5 - Local Rain',
        'A6': 'A6 - Conductivity Corrected',
        'B6': 'B6 - Conductivity Step'
    }

    # Use a palette for the fixed flags
    palette = sns.color_palette("bright", len(fixed_flags))
    color_map = {flag: palette[i] for i, flag in enumerate(fixed_flags)}

    # Get unique flags present in this year's data
    present_flags = set(data[flag_column].dropna().unique())
    present_flags = {f for f in present_flags if f in fixed_flags}

    # Color background only where flag is present and variable is NA
    for flag in present_flags:
        color = color_map[flag]
        subset = data[data[flag_column] == flag]
        if subset.empty:
            continue
        # Group into contiguous intervals
        times = subset["DateTime"].sort_values()
        spans = []
        start = times.iloc[0]
        prev = start
        for t in times.iloc[1:]:
            # Treat gaps larger than 1h as breaks (sampling ~15min)
            if t - prev > pd.Timedelta("1h"):
                spans.append((start, prev))
                start = t
            prev = t
        spans.append((start, prev))
        for s, e in spans:
            ax.axvspan(s, e, color=color, alpha=0.3)

    # Create legend handles with only present flags
    handles = []
    # Add handle for the line
    line_handle = plt.Line2D([0], [0], color='brown', linewidth=0.7, label='Shallow Chlorophyll (RFU)')
    handles.append(line_handle)
    # Add handles for present flags only
    for flag in sorted(present_flags):
        color = color_map[flag]
        label = fixed_flags[flag]
        handles.append(mpatches.Patch(color=color, label=label))
    ax.legend(handles=handles, loc="upper right")

    ax.set_title(f"{variable} ({year})")
    ax.set_xlabel("DateTime")
    ax.set_ylabel(variable)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"figure{year}.png"))
    plt.close(fig)


def main():
    df = load_all_data()
    print("columns available for plotting:", list_variables(df))

    # create yearly plots for ChlRFUShallow_RFU
    flag_defs = read_flag_definitions(os.path.join(DATA_DIR, "data_flags.csv"))
    for yr_idx, yr in enumerate(sorted(df["Year"].unique()), 1):
        plot_with_flag_bands(
            df,
            variable="ChlRFUShallow_RFU",
            flag_column="ChlRFUShallow_RFU_Flag",
            year=yr,
            flag_defs=flag_defs,
        )


if __name__ == "__main__":
    main()