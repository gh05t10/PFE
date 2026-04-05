"""
Tương thích ngược: pipeline ground truth đã chuyển sang
``chl_ground_truth_preprocessing``. File này re-export API cũ.
"""

from chl_ground_truth_preprocessing import (
    DEFAULT_DATA_DIR,
    DEFAULT_INVALID_FLAGS,
    DEFAULT_MAX_GAP_TIMESTEPS,
    DEFAULT_YEARS,
    FLAG_COL,
    GroundTruthConfig,
    TARGET_COL,
    clean_chlorophyll,
    interpolate_short_gaps,
    load_year,
    preprocess_all_years,
    resample_to_10min,
    resample_to_grid,
)

DATA_DIR = str(DEFAULT_DATA_DIR)
YEARS = list(DEFAULT_YEARS)
INVALID_FLAGS = set(DEFAULT_INVALID_FLAGS)
MAX_GAP_TIMESTEPS = DEFAULT_MAX_GAP_TIMESTEPS


def main() -> None:
    from pathlib import Path

    from chl_ground_truth_preprocessing import main as run_gt_main

    run_gt_main()


if __name__ == "__main__":
    main()
