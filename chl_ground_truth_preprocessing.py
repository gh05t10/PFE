"""
Tiền xử lý **ChlRFUShallow_RFU** làm ground truth cho mô hình dự báo / soft sensor.

Bối cảnh
--------
- Dataset FRDR Buffalo Pound (Baulch et al., 2025): chất lượng nước + khí tượng
  tần số cao; ``ChlRFUShallow_RFU`` là chlorophyll theo đơn vị RFU tại ~0.8 m
  (README ``FRDR_dataset_1095/README.txt``).
- Cột ``ChlRFUShallow_RFU_Flag`` mô tả chất lượng; định nghĩa cờ trong
  ``data_flags.csv`` (ví dụ **B7** biofouling spike sau site visit, **C** faulty,
  **M** missing).

Pipeline (chuẩn 30 phút)
------------------------
1. Chuẩn hóa ``DateTime``: parse thống nhất; nếu có timezone thì quy về UTC rồi
   lưu dạng naive; làm tròn tới giây; khi ghi CSV dùng ``YYYY-MM-DD HH:MM:SS``.
2. Loại giá trị target không tin cậy theo cờ (mặc định B7, C, M).
3. Resample toàn bộ cột lên lưới **30 phút** (mean cho đo số; với ``ChlRFUShallow_RFU``
   dùng mean **cắt** tại lần cờ B7/C/M đầu tiên trong bin để không trộn miss và
   phần hồi phục; *_Flag* lấy mẫu **đầu theo thời gian** trong bin), phủ từ min–max.
4. Nội suy tuyến tính theo thời gian **chỉ** cho ``ChlRFUShallow_RFU`` tại các
   đoạn NaN liên tiếp có độ dài ≤ *max_gap_timesteps* (mặc định 2 bước = 1 giờ),
   để không “bịa” dữ liệu trong mất tín hiệu dài.

Đầu ra: ``BPBuoyData_{year}_Preprocessed.csv`` (cùng schema gần như bản Cleaned,
  lưới đều 30 phút, cột thời gian cùng một định dạng giữa các năm).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("FRDR_dataset_1095")
DEFAULT_YEARS = tuple(range(2014, 2022))

TARGET_COL = "ChlRFUShallow_RFU"
FLAG_COL = "ChlRFUShallow_RFU_Flag"

# Theo data_flags.csv + phân bố thực tế trong CSV (B7, C, M).
DEFAULT_INVALID_FLAGS = frozenset({"B7", "C", "M"})

DEFAULT_RESAMPLE_RULE = "30min"

# Tối đa 2 bước × 30 phút = 1 giờ
DEFAULT_MAX_GAP_TIMESTEPS = 2

# Chuỗi strftime khi ghi CSV (đồng bộ mọi năm)
DATETIME_CSV_FORMAT = "%Y-%m-%d %H:%M:%S"


def _rule_minutes(rule: str) -> int:
    """Độ dài một bước lưới resample, tính bằng phút (vd. ``30min`` → 30)."""
    return int(pd.Timedelta(rule).total_seconds() // 60)


@dataclass
class GroundTruthConfig:
    """Cấu hình một lần chạy pipeline ground truth."""

    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_DATA_DIR
    years: tuple[int, ...] = DEFAULT_YEARS
    target_col: str = TARGET_COL
    flag_col: str = FLAG_COL
    invalid_flags: frozenset[str] = field(default_factory=lambda: DEFAULT_INVALID_FLAGS)
    resample_rule: str = DEFAULT_RESAMPLE_RULE
    max_gap_timesteps: int = DEFAULT_MAX_GAP_TIMESTEPS
    manifest_path: Path | None = None


def normalize_datetime_column(series: pd.Series) -> pd.Series:
    """Parse datetime thống nhất; tz-aware → UTC rồi naive; làm tròn tới giây."""
    s = pd.to_datetime(series, errors="coerce")
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.dt.floor("s")


def load_year(year: int, data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    data_dir = Path(data_dir)
    path = data_dir / f"BPBuoyData_{year}_Cleaned.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    df["DateTime"] = normalize_datetime_column(df["DateTime"])
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime")
    return df


def clean_chlorophyll(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    flag_col: str = FLAG_COL,
    invalid_flags: frozenset[str] | set[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Đặt *target_col* = NaN khi *flag_col* thuộc tập cờ không hợp lệ."""
    invalid_flags = invalid_flags or DEFAULT_INVALID_FLAGS
    df = df.copy()
    total_affected = 0
    for flag in sorted(invalid_flags):
        mask = df[flag_col].astype(str).str.strip() == flag
        count = int(mask.sum())
        if count > 0:
            df.loc[mask, target_col] = np.nan
            if verbose:
                print(f"  Flag '{flag}': {count} row(s) → {target_col} set to NaN")
        total_affected += count
    if verbose:
        print(f"  Total rows invalidated: {total_affected} / {len(df)}")
    return df


def _truncated_mean_until_first_bad_flag(
    win: pd.DataFrame,
    value_col: str,
    flag_col: str,
    bad_flags: frozenset[str],
) -> float:
    """Trung bình *value_col* chỉ trên các mẫu **trước** mẫu đầu tiên mang cờ xấu trong bin.

    Nếu mẫu đầu tiên theo thời gian trong bin đã có cờ thuộc *bad_flags* → trả về NaN
    (không lấy mean của phần sau khi cảm biến “bật lại” trong cùng bin — lỗi 2015-05-17 10:00).

    Nếu không có cờ xấu trong bin → ``mean`` toàn bin (như cũ).
    """
    if win.empty:
        return np.nan
    flags = win[flag_col]
    vals = win[value_col]
    for i in range(len(win)):
        f = flags.iloc[i]
        if pd.notna(f) and str(f).strip() in bad_flags:
            if i == 0:
                return np.nan
            return float(vals.iloc[:i].mean(skipna=True))
    return float(vals.mean(skipna=True))


def _chronological_first(series: pd.Series) -> Any:
    """Giá trị tại mốc thời gian sớm nhất trong bin (kể cả NaN).

    ``Resampler.first()`` của pandas bỏ qua NaN, nên cờ ở mẫu sau trong cùng bin
    (vd. B7 lúc 05:15) bị gán nhầm lên nhãn đầu bin (05:00). Dùng ``iloc[0]`` sau
    khi đã sort index để khớp bản ghi gốc.
    """
    if series.empty:
        return np.nan
    return series.iloc[0]


def resample_to_grid(df: pd.DataFrame, rule: str = DEFAULT_RESAMPLE_RULE) -> pd.DataFrame:
    """Resample lên lưới đều *rule* (vd. ``30min``).

    - Cột đo số (không phải *_Flag*): ``mean``.
    - Riêng ``ChlRFUShallow_RFU``: mean chỉ trên đoạn **trước** lần đầu xuất hiện
      cờ B7/C/M trong bin (xem ``_truncated_mean_until_first_bad_flag``).
    - Mọi cột ``*_Flag`` và cột non-numeric khác: lấy **mẫu đầu theo thời gian**
      trong bin (``iloc[0]``), không dùng ``.first()`` của pandas.
    """
    df = df.copy().set_index("DateTime").sort_index()
    start = df.index.min().floor(rule)
    end = df.index.max().ceil(rule)
    full_index = pd.date_range(start=start, end=end, freq=rule)

    flag_cols = [c for c in df.columns if c.endswith("_Flag")]
    numeric_cols = [
        c
        for c in df.columns
        if c not in flag_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    other_cols = [c for c in df.columns if c not in flag_cols and c not in numeric_cols]

    use_trunc_chl = TARGET_COL in numeric_cols and FLAG_COL in df.columns
    numeric_for_mean = [c for c in numeric_cols if not (use_trunc_chl and c == TARGET_COL)]

    parts: list[pd.DataFrame] = []
    if numeric_for_mean:
        parts.append(df[numeric_for_mean].resample(rule).mean())
    if use_trunc_chl:
        grouper = pd.Grouper(freq=rule, label="left", closed="left")

        def _agg_chl_window(w: pd.DataFrame) -> float:
            return _truncated_mean_until_first_bad_flag(w, TARGET_COL, FLAG_COL, DEFAULT_INVALID_FLAGS)

        gb = df.groupby(grouper, group_keys=True)[[TARGET_COL, FLAG_COL]]
        try:
            chl_agg = gb.apply(_agg_chl_window, include_groups=False)
        except TypeError:
            chl_agg = gb.apply(_agg_chl_window)
        if isinstance(chl_agg, pd.Series):
            chl_series = chl_agg
        else:
            chl_series = chl_agg.squeeze(axis=1)
        parts.append(chl_series.rename(TARGET_COL).to_frame())
    if flag_cols:
        flag_df = pd.concat(
            [df[c].resample(rule).apply(_chronological_first).rename(c) for c in flag_cols],
            axis=1,
        )
        parts.append(flag_df)
    if other_cols:
        other_df = pd.concat(
            [df[c].resample(rule).apply(_chronological_first).rename(c) for c in other_cols],
            axis=1,
        )
        parts.append(other_df)

    if not parts:
        resampled = pd.DataFrame(index=full_index)
    else:
        resampled = pd.concat(parts, axis=1)

    resampled = resampled.reindex(full_index)
    resampled = resampled[[c for c in df.columns if c in resampled.columns]]
    resampled.index.name = "DateTime"
    return resampled


def resample_to_10min(df: pd.DataFrame) -> pd.DataFrame:
    """Alias tương thích cũ — hiện gọi ``resample_to_grid`` với ``DEFAULT_RESAMPLE_RULE``."""
    return resample_to_grid(df, DEFAULT_RESAMPLE_RULE)


def interpolate_short_gaps(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    max_gap_timesteps: int = DEFAULT_MAX_GAP_TIMESTEPS,
    step_minutes: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Nội suy theo thời gian các đoạn NaN ngắn của *target_col* (không đụng cột khác)."""
    if step_minutes is None:
        step_minutes = _rule_minutes(DEFAULT_RESAMPLE_RULE)
    df = df.copy()
    series = df[target_col]
    is_nan = series.isna()
    interpolate_mask = pd.Series(False, index=series.index)

    gap_start: int | None = None
    run_indices: list[int] = []
    for idx, val in enumerate(is_nan):
        if val:
            if gap_start is None:
                gap_start = idx
            run_indices.append(idx)
        else:
            if gap_start is not None:
                if len(run_indices) <= max_gap_timesteps:
                    interpolate_mask.iloc[run_indices] = True
                gap_start = None
                run_indices = []
    if gap_start is not None and len(run_indices) <= max_gap_timesteps:
        interpolate_mask.iloc[run_indices] = True

    interpolated = series.interpolate(method="time", limit=max_gap_timesteps)
    series_filled = series.copy()
    series_filled[interpolate_mask] = interpolated[interpolate_mask]
    df[target_col] = series_filled

    if verbose:
        n_filled = int(interpolate_mask.sum())
        n_remaining = int(df[target_col].isna().sum())
        max_wall = max_gap_timesteps * step_minutes
        print(f"  Gaps filled (≤{max_gap_timesteps} steps = {max_wall} min): {n_filled} timestep(s)")
        print(f"  NaN remaining (long gaps)  : {n_remaining}")

    return df


def _year_summary(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    s = df[target_col]
    idx = df.index
    return {
        "n_rows": int(len(df)),
        "datetime_min": str(idx.min()) if len(idx) else None,
        "datetime_max": str(idx.max()) if len(idx) else None,
        "target_non_nan": int(s.notna().sum()),
        "target_nan": int(s.isna().sum()),
        "target_valid_fraction": float(s.notna().mean()) if len(df) else 0.0,
    }


def save_preprocessed_csv(df: pd.DataFrame, path: Path, dt_format: str = DATETIME_CSV_FORMAT) -> None:
    """Ghi CSV với cột DateTime cùng định dạng chuỗi giữa các năm."""
    out = df.reset_index()
    out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce").dt.strftime(dt_format)
    out.to_csv(path, index=False)


def preprocess_all_years(cfg: GroundTruthConfig | None = None) -> dict[int, pd.DataFrame]:
    """Chạy chuẩn hóa thời gian → clean → resample → nội suy gap ngắn; ghi CSV."""
    cfg = cfg or GroundTruthConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, pd.DataFrame] = {}
    per_year_manifest: dict[str, dict[str, Any]] = {}

    for year in cfg.years:
        print(f"\n{'=' * 50}")
        print(f"  Processing year {year} ...")
        print(f"{'=' * 50}")

        try:
            df_raw = load_year(year, cfg.data_dir)
        except FileNotFoundError as exc:
            print(f"  [SKIP] {exc}")
            continue

        n_raw = len(df_raw)
        print(f"  Rows loaded (raw)        : {n_raw}")

        df_clean = clean_chlorophyll(
            df_raw,
            target_col=cfg.target_col,
            flag_col=cfg.flag_col,
            invalid_flags=cfg.invalid_flags,
        )
        n_nan_after_clean = int(df_clean[cfg.target_col].isna().sum())
        print(f"  NaN after cleaning       : {n_nan_after_clean} ({n_nan_after_clean / n_raw * 100:.1f}%)")

        step_min = _rule_minutes(cfg.resample_rule)
        df_resampled = resample_to_grid(df_clean, cfg.resample_rule)
        n_resampled = len(df_resampled)
        n_nan_resampled = int(df_resampled[cfg.target_col].isna().sum())
        pct_nan = n_nan_resampled / n_resampled * 100 if n_resampled > 0 else 0.0
        print(f"  Rows after resample      : {n_resampled}")
        print(f"  NaN after resample       : {n_nan_resampled} ({pct_nan:.1f}%)")

        df_final = interpolate_short_gaps(
            df_resampled,
            target_col=cfg.target_col,
            max_gap_timesteps=cfg.max_gap_timesteps,
            step_minutes=step_min,
        )

        out_path = cfg.output_dir / f"BPBuoyData_{year}_Preprocessed.csv"
        save_preprocessed_csv(df_final, out_path)
        print(f"  Saved → {out_path}")

        results[year] = df_final
        per_year_manifest[str(year)] = _year_summary(df_final, cfg.target_col)

    if cfg.manifest_path is not None:
        step_min = _rule_minutes(cfg.resample_rule)
        global_summary = {
            "target_col": cfg.target_col,
            "flag_col": cfg.flag_col,
            "invalid_flags": sorted(cfg.invalid_flags),
            "resample_rule": cfg.resample_rule,
            "resample_step_minutes": step_min,
            "datetime_csv_format": DATETIME_CSV_FORMAT,
            "max_gap_timesteps": cfg.max_gap_timesteps,
            "max_gap_wall_minutes": cfg.max_gap_timesteps * step_min,
            "years_processed": sorted(results.keys()),
            "per_year": per_year_manifest,
        }
        cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.manifest_path.write_text(json.dumps(global_summary, indent=2), encoding="utf-8")
        print(f"\nManifest written → {cfg.manifest_path}")

    return results


def main() -> None:
    """CLI mặc định: xử lý mọi năm, in tóm tắt toàn cục."""
    manifest = Path("processed/chl_ground_truth/manifest.json")
    cfg = GroundTruthConfig(manifest_path=manifest)
    results = preprocess_all_years(cfg)

    print(f"\n{'=' * 50}")
    print("  GLOBAL SUMMARY")
    print(f"{'=' * 50}")
    total_rows = sum(len(v) for v in results.values())
    total_nan = sum(int(v[cfg.target_col].isna().sum()) for v in results.values())
    pct = total_nan / total_rows * 100 if total_rows > 0 else 0.0
    print(f"  Years processed          : {sorted(results.keys())}")
    print(f"  Total rows ({cfg.resample_rule} grid) : {total_rows}")
    print(f"  Total NaN remaining      : {total_nan} ({pct:.1f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
