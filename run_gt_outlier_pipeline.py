#!/usr/bin/env python3
"""Run outlier labeling + optional winsorization on Rule-A shallow Chl timeseries."""

from pathlib import Path

from src.chl_gt_outlier import GtOutlierExportConfig, run_gt_outlier_export

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    cfg = GtOutlierExportConfig(
        input_csv=base / "processed" / "chl_shallow" / "chl_shallow_rule_a_timeseries.csv",
        out_dir=base / "processed" / "chl_shallow",
        hampel_window=97,
        hampel_n_sigma=3.0,
        winsor_inner=True,
    )
    run_gt_outlier_export(cfg)
    print(f"Wrote chl_gt_outlier_flags.csv and related outputs under {cfg.out_dir}")
