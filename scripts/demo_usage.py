"""
demo_usage.py
--------------
End-to-end demonstration of the chlorophyll B7 removal and
imputation pipeline.  Run this script to produce all outputs.

Usage
-----
    cd /path/to/PFE
    python scripts/demo_usage.py

Steps performed
---------------
1. Remove B7 flags      -> FRDR_dataset_1095/BPBuoyData_2014_B7Removed.csv
2. Fast interpolations  -> FRDR_dataset_1095/BPBuoyData_2014_Fast_*.csv
3. Advanced imputations -> FRDR_dataset_1095/BPBuoyData_2014_Advanced_*.csv
4. Performance comparison -> reports/performance_comparison.csv
5. Cross-validation      -> reports/cross_validation_results.csv
6. Visualizations        -> visualizations/*.png
"""

import os
import sys
import pandas as pd

# Allow import from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "FRDR_dataset_1095"
INPUT_B7REMOVED = os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")


def step1_remove_b7():
    print("\n" + "=" * 60)
    print("STEP 1: Remove B7 biofouling flags")
    print("=" * 60)
    from scripts.remove_biofouling_b7 import main as remove_main
    remove_main()


def step2_fast_methods():
    print("\n" + "=" * 60)
    print("STEP 2: Fast interpolation methods (< 1 minute each)")
    print("=" * 60)
    from scripts.chlorophyll_interpolation import ChlorophyllInterpolator

    df = pd.read_csv(INPUT_B7REMOVED, parse_dates=["DateTime"])
    interp = ChlorophyllInterpolator(df)

    methods = {
        "Linear":     interp.linear_interpolate,
        "Spline":     lambda: interp.spline_interpolate(order=3),
        "Polynomial": lambda: interp.polynomial_interpolate(degree=3),
        "kNN":        lambda: interp.knn_imputation(k=5),
        "LightGBM":   lambda: interp.lightgbm_imputation(n_estimators=50, max_iter=5),
    }

    for name, func in methods.items():
        print(f"\n  Running {name} ...")
        result_df = func()
        out = os.path.join(DATA_DIR, f"BPBuoyData_2014_Fast_{name}.csv")
        result_df.to_csv(out, index=False)
        print(f"  Saved -> {out}")


def step3_advanced_methods():
    print("\n" + "=" * 60)
    print("STEP 3: Advanced imputation methods (may take several minutes)")
    print("=" * 60)
    from scripts.chlorophyll_advanced_imputation import AdvancedChlorophyllImputation

    df = pd.read_csv(INPUT_B7REMOVED, parse_dates=["DateTime"])
    adv = AdvancedChlorophyllImputation(df)

    methods = {
        "MissForest":      lambda: adv.missforest_imputation(n_estimators=50, max_iter=5),
        "GaussianProcess": lambda: adv.gaussian_process_imputation(n_restarts_optimizer=3),
    }

    for name, func in methods.items():
        print(f"\n  Running {name} ...")
        result_df = func()
        out = os.path.join(DATA_DIR, f"BPBuoyData_2014_Advanced_{name}.csv")
        result_df.to_csv(out, index=False)
        print(f"  Saved -> {out}")


def step4_performance():
    print("\n" + "=" * 60)
    print("STEP 4: Performance comparison (all 7 methods)")
    print("=" * 60)
    from scripts.performance_comparison import main as perf_main
    perf_main()


def step5_cross_validation():
    print("\n" + "=" * 60)
    print("STEP 5: K-Fold Cross-Validation (5 folds)")
    print("=" * 60)
    from scripts.cross_validation import main as cv_main
    cv_main()


def step6_visualizations():
    print("\n" + "=" * 60)
    print("STEP 6: Generating visualizations")
    print("=" * 60)
    from scripts.visualization_comparison import main as viz_main
    viz_main()


def print_summary():
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    outputs = [
        ("B7-removed CSV",           os.path.join(DATA_DIR, "BPBuoyData_2014_B7Removed.csv")),
        ("Fast – Linear",            os.path.join(DATA_DIR, "BPBuoyData_2014_Fast_Linear.csv")),
        ("Fast – Spline",            os.path.join(DATA_DIR, "BPBuoyData_2014_Fast_Spline.csv")),
        ("Fast – Polynomial",        os.path.join(DATA_DIR, "BPBuoyData_2014_Fast_Polynomial.csv")),
        ("Fast – kNN",               os.path.join(DATA_DIR, "BPBuoyData_2014_Fast_kNN.csv")),
        ("Fast – LightGBM ⭐",       os.path.join(DATA_DIR, "BPBuoyData_2014_Fast_LightGBM.csv")),
        ("Advanced – MissForest ⭐⭐⭐", os.path.join(DATA_DIR, "BPBuoyData_2014_Advanced_MissForest.csv")),
        ("Advanced – GaussianProcess ⭐⭐", os.path.join(DATA_DIR, "BPBuoyData_2014_Advanced_GaussianProcess.csv")),
        ("Performance summary",      "reports/performance_summary.txt"),
        ("Performance CSV",          "reports/performance_comparison.csv"),
        ("CV results",               "reports/cross_validation_results.csv"),
        ("Overlay plot",             "visualizations/01_overlay_plot_all_methods.png"),
        ("Zoom B7 plot",             "visualizations/02_zoom_b7_region.png"),
        ("Error boxplot",            "visualizations/03_error_boxplot.png"),
        ("Performance bar chart",    "visualizations/04_performance_metrics_bar.png"),
        ("Speed vs accuracy",        "visualizations/05_speed_vs_accuracy_tradeoff.png"),
        ("CV heatmap",               "visualizations/06_cv_heatmap.png"),
    ]

    for label, path in outputs:
        status = "✓" if os.path.exists(path) else "✗ missing"
        print(f"  {status}  {label:<38} {path}")

    print("\nRecommendations:")
    print("  - Quick analysis / demo   : Fast – LightGBM (< 3 min) ⭐")
    print("  - Highest accuracy         : Advanced – MissForest (5-10 min) ⭐⭐⭐")
    print("  - Speed + good accuracy    : Fast – kNN (< 1 min) or Spline")
    print("\nSee reports/README_Recommendations.md for full guidance.")


def main():
    step1_remove_b7()
    step2_fast_methods()
    step3_advanced_methods()
    step4_performance()
    step5_cross_validation()
    step6_visualizations()
    print_summary()


if __name__ == "__main__":
    main()
