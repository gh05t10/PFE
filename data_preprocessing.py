import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_2014_data():
    # Load the 2014 data
    data_path = "FRDR_dataset_1095/BPBuoyData_2014_Cleaned.csv"
    df = pd.read_csv(data_path, parse_dates=["DateTime"])
    
    # Based on the visualizations, which highlight flagged periods,
    # we preprocess by removing rows where the ChlRFUShallow_RFU_Flag is not NaN
    # and also remove rows where ChlRFUShallow_RFU is NaN
    
    # Remove rows with NaN in ChlRFUShallow_RFU
    df_clean = df.dropna(subset=['ChlRFUShallow_RFU'])
    
    # Remove rows where flag is present (indicating problematic data)
    df_clean = df_clean[df_clean['ChlRFUShallow_RFU_Flag'].isna()]
    
    # Save the cleaned data
    output_path = "FRDR_dataset_1095/BPBuoyData_2014_Preprocessed.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    print(f"Original rows: {len(df)}, Cleaned rows: {len(df_clean)}")

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
