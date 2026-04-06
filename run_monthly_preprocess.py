#!/usr/bin/env python3
"""Build slide-aligned monthly multivariate panel (soft sensor + horizon ≥ 1 month)."""

from pathlib import Path

from src.monthly_forecast_preprocess import run_full_preprocess

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    run_full_preprocess(base / "data", base / "processed" / "chl_shallow", min_days=3, calibration_trailing_days=14)
    print(f"Wrote monthly_multivariate_panel*.csv under {base / 'processed' / 'chl_shallow'}")
