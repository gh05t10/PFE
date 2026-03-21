# Chlorophyll Imputation – Method Selection Guide

## Overview

This repository provides **7 interpolation / imputation methods** for
recovering chlorophyll (ChlRFUShallow_RFU) values that were removed
because of biofouling (quality flag **B7**) in the 2014 buoy dataset.

The methods are grouped into two tiers:

| Tier | Methods | Typical runtime |
|------|---------|----------------|
| 🚀 **FAST** | Linear, Spline, Polynomial, kNN, LightGBM | < 1 min |
| ⭐ **ADVANCED** | MissForest, Gaussian Process | 3–10 min |

---

## Method Descriptions

### 🚀 Fast Methods

#### 1. Linear Interpolation
Connect adjacent valid data points with a straight line.
- **Best for**: Very short gaps, quick sanity checks.
- **Weakness**: Over-smooths curves; no multivariate information used.

#### 2. Spline Interpolation (cubic)
Fit a smooth spline curve through valid data.
- **Best for**: Short to medium gaps, smooth trends.
- **Weakness**: Can oscillate near gap boundaries for long gaps.

#### 3. Polynomial Interpolation
Fit a degree-3 polynomial to bridge the gap.
- **Best for**: Gaps with a clear trend direction.
- **Weakness**: Runge's phenomenon for long gaps; can overshoot.

#### 4. k-Nearest Neighbors (kNN)
Use the k most similar environmental conditions (temperature, DO, pH…)
to estimate the missing chlorophyll value.
- **Best for**: Multivariate pattern-matching; moderate gaps.
- **Recommended k**: 5 (default).

#### 5. LightGBM Iterative Imputation ⭐ **(Recommended FAST)**
Gradient-boosted trees inside scikit-learn's IterativeImputer (MICE style).
- **Best for**: Long gaps, complex covariate relationships, production-ready FAST output.
- **3–5× faster** than MissForest with comparable accuracy.

---

### ⭐ Advanced Methods

#### 6. MissForest ⭐⭐⭐ **(Best Overall)**
Random Forest iterative imputation – the gold standard for multivariate
tabular missing data.
- **Best for**: Highest-accuracy, publication-quality results.
- **Parallel processing** (`n_jobs=-1`) uses all CPU cores.
- Runtime: ~5–10 min with `n_estimators=50, max_iter=5`.

#### 7. Gaussian Process Regression ⭐⭐
Kernel-based probabilistic regression; provides uncertainty estimates.
- **Best for**: When confidence intervals on imputed values matter.
- Runtime: ~3–5 min (subsamples training data for large datasets).

---

## Decision Guide

```
Do you need results in < 1 minute?
  YES → Are short gaps (~hours)?
          YES → Spline or Linear
          NO  → kNN or LightGBM ⭐
  NO  → Do you need uncertainty estimates?
          YES → Gaussian Process ⭐⭐
          NO  → MissForest ⭐⭐⭐
```

---

## How to Run

### Step 1: Remove B7 data
```bash
python scripts/remove_biofouling_b7.py
```

### Step 2a: Run all FAST methods
```bash
python scripts/chlorophyll_interpolation.py
```

### Step 2b: Run ADVANCED methods
```bash
python scripts/chlorophyll_advanced_imputation.py
```

### Step 3: Compare performance
```bash
python scripts/performance_comparison.py
```

### Step 4: Cross-validation
```bash
python scripts/cross_validation.py
```

### Step 5: Visualizations
```bash
python scripts/visualization_comparison.py
```

### Or run everything at once
```bash
python scripts/demo_usage.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `FRDR_dataset_1095/BPBuoyData_2014_B7Removed.csv` | B7 values replaced with NaN |
| `FRDR_dataset_1095/BPBuoyData_2014_Fast_Linear.csv` | Linear interpolation |
| `FRDR_dataset_1095/BPBuoyData_2014_Fast_Spline.csv` | Cubic spline |
| `FRDR_dataset_1095/BPBuoyData_2014_Fast_Polynomial.csv` | Polynomial degree-3 |
| `FRDR_dataset_1095/BPBuoyData_2014_Fast_kNN.csv` | kNN imputation |
| `FRDR_dataset_1095/BPBuoyData_2014_Fast_LightGBM.csv` | LightGBM iterative |
| `FRDR_dataset_1095/BPBuoyData_2014_Advanced_MissForest.csv` | MissForest RF |
| `FRDR_dataset_1095/BPBuoyData_2014_Advanced_GaussianProcess.csv` | GPR |
| `reports/B7_removal_report.txt` | B7 removal statistics |
| `reports/performance_summary.txt` | Ranked method comparison |
| `reports/performance_comparison.csv` | Detailed metrics CSV |
| `reports/cross_validation_results.csv` | Per-fold CV metrics |
| `reports/cross_validation_summary.csv` | Mean ± std CV summary |
| `visualizations/01_overlay_plot_all_methods.png` | All methods overlay |
| `visualizations/02_zoom_b7_region.png` | B7 gap zoom |
| `visualizations/03_error_boxplot.png` | Error distributions |
| `visualizations/04_performance_metrics_bar.png` | MAE/RMSE/MAPE bar |
| `visualizations/05_speed_vs_accuracy_tradeoff.png` | Speed vs accuracy |
| `visualizations/06_cv_heatmap.png` | CV RMSE heatmap |

---

## Dependencies

```
pandas
numpy
scikit-learn
lightgbm
scipy
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn lightgbm scipy matplotlib seaborn
```

---

## Reusing for Other Years

The scripts are parameterised.  To process another year:

```python
from scripts.chlorophyll_interpolation import ChlorophyllInterpolator
import pandas as pd

df = pd.read_csv("FRDR_dataset_1095/BPBuoyData_2015_Cleaned.csv",
                 parse_dates=["DateTime"])

# Mask whatever flag you consider erroneous
flag_col = "ChlRFUShallow_RFU_Flag"
df.loc[df[flag_col] == "B7", "ChlRFUShallow_RFU"] = float("nan")

interp = ChlorophyllInterpolator(df)
df_best = interp.lightgbm_imputation()
df_best.to_csv("BPBuoyData_2015_LightGBM_Imputed.csv", index=False)
```
