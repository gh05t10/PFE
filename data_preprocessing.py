import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_2014_data(input_path: str = "FRDR_dataset_1095/BPBuoyData_2014_Cleaned.csv",
                         output_path: str = "FRDR_dataset_1095/BPBuoyData_2014_Preprocessed.csv",
                         flag_value: str = "B7") -> int:
    """Remove biofouling-affected chlorophyll measurements from the 2014 dataset.

    Rows whose ``ChlRFUShallow_RFU_Flag`` equals *flag_value* (default ``'B7'``)
    have their ``ChlRFUShallow_RFU`` value replaced with ``NaN``.  All other
    columns and rows are preserved unchanged.

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

    # Identify rows flagged with the biofouling quality flag
    b7_mask = df["ChlRFUShallow_RFU_Flag"] == flag_value
    affected_count = int(b7_mask.sum())

    if affected_count > 0:
        # Replace only the chlorophyll measurement; keep all other columns intact
        df.loc[b7_mask, "ChlRFUShallow_RFU"] = np.nan

        affected_rows = df.loc[b7_mask, "DateTime"]
        print(f"Flag '{flag_value}' detected in {affected_count} rows.")
        print(f"  Affected period: {affected_rows.min()} → {affected_rows.max()}")
    else:
        print(f"No rows with flag '{flag_value}' found.")

    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    print(f"Total rows: {len(df)}, rows with chlorophyll set to NaN: {affected_count}")

    return affected_count

def plot_preprocessed_2014():
    # Load the preprocessed 2014 data
    data_path = "FRDR_dataset_1095/BPBuoyData_2014_Preprocessed.csv"
    df = pd.read_csv(data_path, parse_dates=["DateTime"])
    
    # Plot ChlRFUShallow_RFU over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="DateTime", y="ChlRFUShallow_RFU", color="blue")
    plt.title("ChlRFUShallow_RFU in 2014 (Preprocessed)")
    plt.xlabel("DateTime")
    plt.ylabel("ChlRFUShallow_RFU")
    plt.tight_layout()
    plt.savefig("ChlRFUShallow_RFU_2014_Preprocessed.png")
    plt.close()
    print("Plot saved as ChlRFUShallow_RFU_2014_Preprocessed.png")

if __name__ == "__main__":
    preprocess_2014_data()
    plot_preprocessed_2014()
