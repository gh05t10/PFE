"""
demo_usage.py
--------------
End-to-end demonstration script that runs the full chlorophyll ML pipeline:

  Step 1 – Remove B7 biofouling data → BPBuoyData_2014_B7Removed.csv
  Step 2 – Performance comparison (80/20 split)
  Step 3 – Cross-validation (k=5)
  Step 4 – Generate all comparison visualizations

Usage:
    cd /path/to/PFE
    python scripts/demo_usage.py
"""

import os
import sys

# Ensure the scripts package is importable regardless of CWD
sys.path.insert(0, os.path.dirname(__file__))

from remove_biofouling_b7 import remove_b7_biofouling
from performance_comparison import run_comparison
from cross_validation import run_cross_validation
from visualization_comparison import generate_all_plots


def main():
    print("=" * 65)
    print("Step 1 – Remove B7 biofouling data")
    print("=" * 65)
    remove_b7_biofouling()

    print("\n" + "=" * 65)
    print("Step 2 – Performance Comparison (80/20 split)")
    print("=" * 65)
    run_comparison()

    print("\n" + "=" * 65)
    print("Step 3 – Cross-Validation (k=5 folds)")
    print("=" * 65)
    run_cross_validation()

    print("\n" + "=" * 65)
    print("Step 4 – Generate Visualizations")
    print("=" * 65)
    generate_all_plots()

    print("\n✅ Pipeline complete.")
    print("   Cleaned CSV  : FRDR_dataset_1095/BPBuoyData_2014_B7Removed.csv")
    print("   Reports      : reports/")
    print("   Visualizations: reports/visualizations/")


if __name__ == "__main__":
    main()
