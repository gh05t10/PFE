# Outlier handling for shallow chlorophyll ground truth (refs synthesis)

## References used (folder `refs/`)

### Scientific Data — water quality dataset methodology (`s41597-025-04715-4.pdf`)

- Uses **Tukey’s method** with **inner fences** at **Q1 − 1.5·IQR** and **Q3 + 1.5·IQR** (“potential outliers”), and **outer fences** at **±3·IQR** (“possible outliers”).
- Motivation: more robust than mean±k·SD when distributions are **skewed**; explicitly contrasts with normality assumptions.
- Notes that outliers may be **left in** if impact is small; users may apply **trimming, winsorization, or imputation** depending on sensitivity.

### Buffalo Pound dataset / QA narrative (`data_info.pdf`)

- Pigment variables can show **spikes** tied to **site visits**, fouling, or **extreme but real** conditions.
- Automated screening can flag values that are later accepted as **environmental events** (blooms, wind, rain) after manual review.
- Implication for ML ground truth: **do not treat every extreme Chl RFU as a sensor error** without temporal context.

### Aquaculture / WQ time series (`peerj-cs-3515.pdf`)

- Emphasizes **temporal continuity**: deviations relative to **neighboring time steps** motivate outlier definitions in streaming water-quality data.

## Recommended strategy for `chl_shallow_rule_a_timeseries.csv`

1. **Keep existing QC** (B7/C/M already removed upstream).
2. **Prefer time-local robust methods** (Hampel / MAD on a rolling window) to catch **isolated spikes** typical of fouling/glitches.
3. **Avoid global Tukey on the full multi-year marginal distribution** as the sole rule: seasonal blooms skew the bulk distribution and inflate false positives.
4. **Add season-aware Tukey**: compute fences **within month-of-year** (across years) on **daily medians** or subseries to compare apples-to-apples seasonally.
5. **Label-first for GT**: export `*_flag_robust_spike` and produce **winsorized** (inner fence) or **masked** variants rather than silent deletion, so Transformer training can use masks.

## Implementation

See `src/chl_gt_outlier.py` and `run_gt_outlier_pipeline.py`.

### Transformer training file

`processed/chl_shallow/chl_shallow_transformer_gt.csv` keeps **raw** `ChlRFUShallow_RFU` and adds:

- `weight_chl_gt`: **1.0** on timesteps to keep in loss, **0.0** on Hampel spike timesteps (no neighbour imputation).
- `weight_chl_gt_conservative`: same but using the stricter combined rule (Hampel ∧ monthly-daily Tukey).

Use `sum((pred - y)^2 * weight) / sum(weight)` (or PyTorch masked loss) instead of replacing values.
