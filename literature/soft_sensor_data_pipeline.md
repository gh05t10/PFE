# Soft-sensor preprocessing aligned with slides 9–10 + FRDR + PatchTST/TDAlign refs

## Slide `Aquaculture-report17_1.pdf` (pages 9–10) — operational meaning

- **Soft-sensor path (inference):** inputs are **five** water variables only → concurrent backbone → **Q** → **cross-attention** with **K,V** from calibration branch → prediction head → **Chlorophyll_RFU**.
- **Calibration / training branch:** uses **six** channels: the **same five** + **Chlorophyll_RFU** (target) to train **PatchTST encoder** (normalize → patch → positional encoding → **Transformer encoder**), producing **K,V**; slide mentions **TDAlign** alongside this branch.
- **Ground truth:** used for **calibration** (training / teacher), not as input in the **red** soft-sensor path at inference.

This matches your requirement to **separate** shallow chlorophyll GT from input features.

## Correlation figure `correlation_slide_plots/page06_water_correlation.png`

The five non-target water variables (Pearson/Spearman vs `ChlRFUShallow_RFU`) are:

| Slide label (generic) | FRDR column (CSV) |
|----------------------|-------------------|
| Conductivity | `SpCondShallow_uS/cm` |
| DO | `ODOShallow_mg/L` |
| Temp | `TempShallow_C` |
| pH | `pHShallow` |
| Turbidity | `TurbShallow_NTU+` |

Target: `ChlRFUShallow_RFU`.

## `data_info.pdf` (Baulch et al., *Data in Brief*)

- Sampling: **15 min in 2014** and **10 min from 2015–2021** during open-water season.
- QA: field flags + automated screening; **erroneous measurements removed** (your `data/` preprocessed CSVs follow this).

**Implication for PatchTST:** build a **single regular time grid** (e.g. 5 min / 10 min / 15 min) + **missing masks**; do **not** fabricate Chl targets for GT.

## `s41598-025-00741-9.pdf` (Scientific Reports — PatchTST–GRU hybrid)

- This is **not** the original PatchTST paper (ICLR); it is an **application** that uses:
  - **PatchTST encoder** for local temporal patterns,
  - **seq2seq** / decoder for multi-step forecasting,
  - fusion / attention mechanisms.
- Takeaways for implementation: **patching + encoder** on long windows, **multi-step** heads, attention-based fusion — align with your **cross-attention** block in slide 10.

## `2406.04777v3.pdf` (TDAlign)

- **Plug-in** objective: align **changes between adjacent predicted steps** with **changes in the target** (plus standard LTSF loss).
- Use as **auxiliary loss** on the **predicted Chl horizon** (adjacent steps), with **train-only** scaling.

## Recommended next preprocessing steps (PFE-specific)

1. **Export aligned tensors** from `data/BPBuoyData_*_Preprocessed.csv`:
   - `X`: shape `(T, 5)` for the five columns above (same QC flags B7/C/M per column).
   - `y`: `ChlRFUShallow_RFU` **same timestamps**, stored separately.
2. **Unified grid** (train split): resample to chosen `Δt` (e.g. 10 min or 5 min) with **NaN + mask** (no neighbour mean for **y**).
3. **Calibration split** (training only): optional tensor `(T, 6)` = concat(X, y) for **teacher / K,V branch** (slide 10).
4. **Horizon**: build **windowed** sequences from the unified grid (`run_build_window_dataset.py`): `context_len` + `pred_len` in steps (e.g. 14 days @30 min → 672 steps); masks/weights for missing GT.
5. **PatchTST**: patch length & stride chosen **after** grid is fixed; **normalize** per-channel using **train** statistics only.
6. **TDAlign**: add **Δ-step loss** on predicted Chl sequence vs target sequence.
7. **Evaluation**: time-based split (e.g. 2014–2019 / 2020 / 2021) to respect reservoir seasonality.

## Code

See `src/soft_sensor_columns.py` for canonical column names.
