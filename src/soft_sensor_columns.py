"""
Canonical feature/target names for FRDR buoy CSVs (aligned with slides + correlation figure).

- **Soft-sensor inputs (5):** shallow water variables only (no Chl).
- **Target / calibration / GT:** ``ChlRFUShallow_RFU`` kept separate.

Column names match ``data/BPBuoyData_*_Preprocessed.csv`` headers.
"""

from __future__ import annotations

from typing import Final

# Five inputs (slide 9–10 + page06_water_correlation.png)
FEATURE_COLS: Final[tuple[str, ...]] = (
    "SpCondShallow_uS/cm",
    "ODOShallow_mg/L",
    "TempShallow_C",
    "pHShallow",
    "TurbShallow_NTU+",
)

TARGET_COL: Final[str] = "ChlRFUShallow_RFU"


def flag_columns() -> list[str]:
    return [f"{c}_Flag" for c in FEATURE_COLS] + [f"{TARGET_COL}_Flag"]


def describe() -> str:
    lines = [
        "FEATURE_COLS (5):",
        *[f"  - {c}" for c in FEATURE_COLS],
        f"TARGET_COL (GT): {TARGET_COL}",
        "At inference: model input only FEATURE_COLS.",
        f"Calibration / teacher branch (training): FEATURE_COLS + {TARGET_COL}.",
    ]
    return "\n".join(lines)


__all__ = ["FEATURE_COLS", "TARGET_COL", "flag_columns", "describe"]
