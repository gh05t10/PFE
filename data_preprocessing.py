from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("FRDR_dataset_1095")
OUT_DIR = Path("processed/step1_clean_resampled")
OUT_DIR.mkdir(parents=True, exist_ok=True)


RAW_FILE = DATA_DIR / "BPBuoyData_2014_Preprocessed.csv"
RESAMPLED_FILE = OUT_DIR / "BPBuoyData_2014_Preprocessed_10min.csv"

TARGET_COL = "ChlRFUShallow_RFU"

def main():
    raw = pd.read_csv(RAW_FILE)
    raw["DateTime"] = pd.to_datetime(raw["DateTime"], errors="coerce")
    raw = raw.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()

    rs = pd.read_csv(RESAMPLED_FILE)
    rs["DateTime"] = pd.to_datetime(rs["DateTime"], errors="coerce")
    rs = rs.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()

    # số mẫu
    print("Raw rows:", len(raw))
    print("Resampled rows:", len(rs))

    # đánh dấu điểm nội suy: có trong resampled nhưng không có trong raw
    interpolated_idx = rs.index.difference(raw.index)
    print("Interpolated rows:", len(interpolated_idx))

    # tỷ lệ nội suy cho chlorophyll
    if TARGET_COL in rs.columns:
        total = rs[TARGET_COL].notna().sum()
        interp = rs.loc[interpolated_idx, TARGET_COL].notna().sum()
        ratio = interp / total * 100 if total > 0 else 0
        print(f"Interpolated {TARGET_COL}: {interp}/{total} ({ratio:.2f}%)")
    else:
        print(f"Missing target column: {TARGET_COL}")

if __name__ == "__main__":
    main()