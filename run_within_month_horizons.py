#!/usr/bin/env python3
"""Build multi-horizon (≤31 days) supervision table from Rule-A + masked Chl series."""

from pathlib import Path

from src.within_month_horizons import HorizonTableConfig, run_horizon_table_export

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    cfg = HorizonTableConfig(
        input_csv=base / "processed" / "chl_shallow" / "chl_shallow_transformer_gt.csv",
        out_csv=base / "processed" / "chl_shallow" / "horizon_supervision_daily.csv",
    )
    run_horizon_table_export(cfg)
    print(f"Wrote {cfg.out_csv} and chl_daily_for_horizons.csv")
