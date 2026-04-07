"""
Rule A (calendar completeness) for shallow Chl ground truth.

**Rule A (strict):** A month is kept only if **every calendar day** in that month has
at least one valid ``ChlRFUShallow_RFU`` sample after excluding flags B7, C, M.

**Rule A(p) (relaxed):** A month is kept if the fraction
``n_days_with_data / n_days_required >= p`` (e.g. p=0.8).

Both imply **symmetric** handling of partial months at the **start** and **end** of
the record: a partial month is less likely to satisfy the threshold.

Data source: ``data/BPBuoyData_*_Preprocessed.csv`` (not ``ChlRFUShallow_RFU_GroundTruth.csv``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .chl_shallow_pipeline import (
    TARGET_COL,
    daily_table,
    load_trimmed_chl_frames,
    monthly_from_daily,
    PipelineConfig,
)


def _days_in_month(year: int, month: int) -> int:
    return int(pd.Timestamp(year=year, month=month, day=1).days_in_month)


def audit_rule_a(ts: pd.DataFrame, *, p: float = 1.0) -> pd.DataFrame:
    """
    For each calendar month overlapping *ts*, report whether Rule A(p) passes.

    *ts*: DataFrame indexed by DateTime with column TARGET_COL.
    """
    if not (0 < p <= 1.0):
        raise ValueError("p must be in (0, 1].")

    if ts.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "month",
                "year_month",
                "n_days_required",
                "n_days_with_data",
                "p",
                "coverage",
                "rule_a_pass",
            ]
        )

    idx_min = ts.index.min()
    idx_max = ts.index.max()
    start = pd.Timestamp(year=idx_min.year, month=idx_min.month, day=1)
    end = pd.Timestamp(year=idx_max.year, month=idx_max.month, day=1)
    months = pd.period_range(start=start, end=end, freq="M")

    rows = []
    for period in months:
        y, m = period.year, period.month
        need = _days_in_month(y, m)
        period_start = pd.Timestamp(year=y, month=m, day=1)
        period_end = period_start + pd.offsets.MonthEnd(0)
        sub = ts.loc[(ts.index >= period_start) & (ts.index <= period_end)]
        if sub.empty:
            days_with = 0
        else:
            days_with = sub.index.normalize().unique().size
        coverage = float(days_with) / float(need) if need else 0.0
        rows.append(
            {
                "year": y,
                "month": m,
                "year_month": f"{y:04d}-{m:02d}",
                "n_days_required": need,
                "n_days_with_data": int(days_with),
                "p": float(p),
                "coverage": coverage,
                "rule_a_pass": coverage >= p,
            }
        )
    return pd.DataFrame(rows)


def filter_to_rule_a_months(ts: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Keep only timestamps that fall in months where ``rule_a_pass`` is True."""
    good = set(zip(audit.loc[audit["rule_a_pass"], "year"], audit.loc[audit["rule_a_pass"], "month"]))
    if not good:
        return ts.iloc[0:0]
    ym = pd.Series(ts.index.year * 100 + ts.index.month, index=ts.index)
    ok = ts.index.map(lambda t: (t.year, t.month) in good)
    return ts.loc[ok]


def monthly_ground_truth_rule_a(ts: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Two-stage monthly mean (daily mean → monthly mean of dailies) for Rule-A months only."""
    passed = audit[audit["rule_a_pass"]].copy()
    if passed.empty:
        return pd.DataFrame()

    daily = daily_table(ts)
    cfg = PipelineConfig(
        data_dir=Path("."),
        out_dir=Path("."),
        monthly_method="two_stage",
        min_samples_per_month=0,
        min_days_with_data_per_month=0,
    )
    monthly_all = monthly_from_daily(daily, cfg)
    keep = set(passed["year_month"].tolist())
    monthly_all = monthly_all[monthly_all["year_month"].isin(keep)]
    return monthly_all


def run_rule_a_export(data_dir: Path, out_dir: Path, *, p: float = 1.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = load_trimmed_chl_frames(data_dir)
    audit = audit_rule_a(ts, p=p)
    audit.to_csv(out_dir / "rule_a_month_audit.csv", index=False)

    passed = audit[audit["rule_a_pass"]]
    passed.to_csv(out_dir / "rule_a_eligible_months.csv", index=False)

    ts_a = filter_to_rule_a_months(ts, audit)
    ts_a.reset_index().to_csv(out_dir / "chl_shallow_rule_a_timeseries.csv", index=False)

    monthly = monthly_ground_truth_rule_a(ts, audit)
    if not monthly.empty:
        monthly.to_csv(out_dir / "chl_shallow_monthly_gt_rule_a.csv")

    # Short summary for thesis / logs
    summary_lines = [
        f"Total calendar months in range: {len(audit)}",
        f"Rule-A(p) eligible months (p={p}): {len(passed)}",
        f"Rows in trimmed series: {len(ts):,}",
        f"Rows kept (only eligible months): {len(ts_a):,}",
    ]
    (out_dir / "rule_a_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


__all__ = [
    "audit_rule_a",
    "filter_to_rule_a_months",
    "monthly_ground_truth_rule_a",
    "run_rule_a_export",
]
