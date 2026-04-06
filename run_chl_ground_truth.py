#!/usr/bin/env python3
"""Build shallow-Chl RFU ground truth and monthly targets from `data/` preprocessed CSVs."""

from pathlib import Path

from src.chl_shallow_pipeline import PipelineConfig, run_pipeline, summarize_quality

BASE = Path(__file__).resolve().parent


def main() -> None:
    cfg = PipelineConfig(
        data_dir=BASE / "data",
        out_dir=BASE / "processed" / "chl_shallow",
        monthly_method="two_stage",
        min_samples_per_month=24,
        min_days_with_data_per_month=3,
    )
    clean_ts, _daily, monthly = run_pipeline(cfg)
    print(summarize_quality(clean_ts, monthly))
    print(f"Wrote outputs under {cfg.out_dir}")


if __name__ == "__main__":
    main()
