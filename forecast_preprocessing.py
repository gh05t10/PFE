"""
Tiền xử lý cho dự báo nồng độ shallow chlorophyll (ChlRFUShallow_RFU), ưu tiên horizon dài.

Pipeline gợi ý:
- Đầu vào: các file ``BPBuoyData_{year}_Preprocessed.csv`` (lưới 10 phút) từ
  ``chl_ground_truth_preprocessing.preprocess_all_years`` hoặc ``data_preprocessing.py``.
- Bước 1: nối đa năm theo thời gian.
- Bước 2: tổng hợp theo ngày (mean) — giảm nhiễu tần số cao, phù hợp dự báo tuần/tháng/mùa.
- Bước 3: chọn cột số, bỏ cột *_Flag; giữ target; có thể giữ biến hồ làm covariate trễ hoặc chỉ dùng khí tượng nếu muốn kịch bản "chỉ dùng ngoại sinh khí hậu".
- Bước 4 (tùy chọn): với mỗi năm dương lịch, loại tháng biên nếu không trọn (ngày đầu tiên trong năm không phải ngày 1, hoặc ngày cuối trong năm không phải ngày cuối tháng).

Xuất: CSV (và tùy chọn Parquet nếu có pyarrow/fastparquet) + manifest JSON mô tả cột và khoảng thời gian.
"""

from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATA_DIR = Path("FRDR_dataset_1095")
OUT_DIR = Path("processed/forecasting")
DEFAULT_YEARS = tuple(range(2014, 2022))

TARGET_COL = "ChlRFUShallow_RFU"


@dataclass
class ForecastPreprocessConfig:
    """Cấu hình một lần chạy tiền xử lý."""

    data_dir: Path = DATA_DIR
    out_dir: Path = OUT_DIR
    years: tuple[int, ...] = DEFAULT_YEARS
    target_col: str = TARGET_COL
    # Tổng hợp theo ngày — chuẩn cho dự báo dài hạn (giảm chiều thời gian, ổn định hơn)
    daily_rule: str = "1D"
    # Cột khí tượng / môi trường (có thể mở rộng); không gồm target
    meteo_cols: tuple[str, ...] = (
        "BarometricPress_kPa",
        "RelativeHum_%",
        "WindDir_Degree",
        "WindSp_km/h",
        "DailyRain_mm",
        "AirTemp_C",
        "PARAir_umol/s/m2",
        "PARW1_umol/s/m2",
        "PARW2_umol/s/m2",
    )
    # Biến hồ nông (dùng làm covariate *trễ* trong mô hình sequence; không dùng làm input đồng thời nếu tránh leakage)
    lake_shallow_cols: tuple[str, ...] = (
        "TempShallow_C",
        "pHShallow",
        "pHmVShallow_mV",
        "ODOSatShallow_%",
        "ODOShallow_mg/L",
        "SpCondShallow_uS/cm",
        "TurbShallow_NTU+",
        "BGPCShallowRFU_RFU",
    )
    # Nếu True, chỉ xuất khí tượng + target (kịch bản ngoại sinh thuần)
    meteo_only_features: bool = False
    # Loại bỏ ngày không có target sau aggregate
    drop_target_nan_days: bool = True
    # Nội suy tuyến tính theo thời gian cho khoảng trống ngắn ở feature (sau daily), tối đa N ngày
    max_ffill_gap_days: int = 3
    # Mỗi năm dương lịch: bỏ tháng chứa ngày quan trắc đầu tiên nếu ngày đó không phải ngày 1;
    # bỏ tháng chứa ngày quan trắc cuối nếu không phải ngày cuối tháng → chỉ giữ tháng “trọn” theo lịch ở biên năm.
    trim_partial_boundary_months_per_year: bool = True


def load_preprocessed_year(year: int, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"BPBuoyData_{year}_Preprocessed.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["DateTime"], low_memory=False)
    df = df.set_index("DateTime").sort_index()
    for c in df.columns:
        if not c.endswith("_Flag"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_concat_years(years: Iterable[int], data_dir: Path) -> pd.DataFrame:
    frames = []
    for y in years:
        try:
            frames.append(load_preprocessed_year(y, data_dir))
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError(f"No BPBuoyData_*_Preprocessed.csv under {data_dir}")
    return pd.concat(frames, axis=0).sort_index()


def aggregate_daily_mean(df: pd.DataFrame, rule: str = "1D") -> pd.DataFrame:
    """Trung bình theo ngày (UTC/local theo index đã có; giả định datetime không đổi múi)."""
    cols = [c for c in df.columns if not c.endswith("_Flag")]
    daily = df[cols].resample(rule).mean(numeric_only=True)
    daily.index.name = "DateTime"
    return daily


def build_feature_frame(
    daily: pd.DataFrame,
    cfg: ForecastPreprocessConfig,
) -> pd.DataFrame:
    feat_order: list[str] = list(cfg.meteo_cols)
    if not cfg.meteo_only_features:
        feat_order.extend(cfg.lake_shallow_cols)
    feat_order = [c for c in feat_order if c in daily.columns]
    missing = [c for c in cfg.meteo_cols if c not in daily.columns]
    if not cfg.meteo_only_features:
        missing.extend([c for c in cfg.lake_shallow_cols if c not in daily.columns])
    if cfg.target_col not in daily.columns:
        missing.append(cfg.target_col)
    if missing:
        raise KeyError(f"Missing expected columns after daily agg: {sorted(set(missing))}")
    # Target cột cuối: X = feat_order, y = target (dễ tách khi train)
    out = daily[feat_order + [cfg.target_col]].copy()
    return out


def trim_partial_boundary_months_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Bỏ tháng không trọn ở đầu/cuối mỗi năm dương lịch (sau khi đã có một dòng / ngày).

    Với mỗi năm *Y*:
    - Nếu ngày sớm nhất trong *Y* không phải ngày 1 của tháng → loại toàn bộ các dòng thuộc tháng đó.
    - Nếu ngày muộn nhất trong *Y* không phải ngày cuối của tháng → loại toàn bộ các dòng thuộc tháng đó.

    Không yêu cầu mọi ngày trong tháng đều có mẫu (tránh quá khắt khe khi thiếu vài ngày giữa tháng).
    """
    if df.empty:
        return df
    df = df.sort_index()
    drop = np.zeros(len(df), dtype=bool)
    idx = df.index
    for year in idx.year.unique():
        m_year = idx.year == year
        sub = idx[m_year]
        if len(sub) == 0:
            continue
        first = sub.min()
        if first.day != 1:
            drop |= m_year & (idx.month == first.month)
        last = sub.max()
        _, last_dom = calendar.monthrange(last.year, last.month)
        if last.day != last_dom:
            drop |= m_year & (idx.month == last.month)
    return df.loc[~drop]


def fill_short_feature_gaps(df: pd.DataFrame, target_col: str, max_gap_days: int) -> pd.DataFrame:
    """Nội suy theo thời gian + ffill/bfill giới hạn cho feature (không đụng target)."""
    df = df.copy()
    feat_cols = [c for c in df.columns if c != target_col]
    for c in feat_cols:
        s = df[c]
        s = s.interpolate(method="time", limit=max_gap_days)
        df[c] = s.ffill(limit=max_gap_days).bfill(limit=max_gap_days)
    return df


def run_preprocess(cfg: ForecastPreprocessConfig | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or ForecastPreprocessConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    full = load_concat_years(cfg.years, cfg.data_dir)
    daily = aggregate_daily_mean(full, cfg.daily_rule)
    frame = build_feature_frame(daily, cfg)

    if cfg.drop_target_nan_days:
        frame = frame.loc[frame[cfg.target_col].notna()].copy()

    n_before_trim = len(frame)
    if cfg.trim_partial_boundary_months_per_year:
        frame = trim_partial_boundary_months_per_year(frame)

    frame = fill_short_feature_gaps(frame, cfg.target_col, cfg.max_ffill_gap_days)

    manifest = {
        "target_col": cfg.target_col,
        "freq": "daily",
        "trim_partial_boundary_months_per_year": cfg.trim_partial_boundary_months_per_year,
        "n_rows_before_boundary_trim": n_before_trim,
        "date_min": str(frame.index.min()) if len(frame) else None,
        "date_max": str(frame.index.max()) if len(frame) else None,
        "n_rows": len(frame),
        "feature_columns": [c for c in frame.columns if c != cfg.target_col],
        "meteo_only": cfg.meteo_only_features,
        "years_used": list(cfg.years),
    }

    csv_path = cfg.out_dir / "bp_daily_forecasting.csv"
    frame.reset_index().to_csv(csv_path, index=False)

    manifest_path = cfg.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Parquet tùy chọn
    try:
        pq_path = cfg.out_dir / "bp_daily_forecasting.parquet"
        frame.to_parquet(pq_path)
        manifest["parquet_path"] = str(pq_path)
    except Exception:
        manifest["parquet_path"] = None

    return frame, manifest


def time_based_split_indices(
    idx: pd.DatetimeIndex,
    train_end: str,
    val_end: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chỉ số vị trí cho train / val / test theo ngày cuối (inclusive train_end, val)."""
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    pos = np.arange(len(idx))
    train_mask = idx <= train_end_ts
    val_mask = (idx > train_end_ts) & (idx <= val_end_ts)
    test_mask = idx > val_end_ts
    return pos[train_mask], pos[val_mask], pos[test_mask]


def build_sequence_arrays(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    input_days: int,
    horizon_days: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    X: (N, input_days, n_features), y: (N, horizon_days) — mỗi mẫu dự báo horizon ngày phía sau cửa sổ.

    Chỉ dùng quá khứ làm input; y là chuỗi target liên tiếp sau ngày cuối cửa sổ.
    """
    values = frame[feature_cols].values.astype(np.float64)
    target = frame[target_col].values.astype(np.float64)
    n = len(frame)
    max_t = n - input_days - horizon_days + 1
    if max_t <= 0:
        raise ValueError("Series too short for input_days + horizon_days")

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    end_dates: list[pd.Timestamp] = []

    for t in range(max_t):
        X_list.append(values[t : t + input_days])
        y_list.append(target[t + input_days : t + input_days + horizon_days])
        end_dates.append(frame.index[t + input_days - 1])

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    end_index = pd.DatetimeIndex(end_dates)
    return X, y, end_index


def main() -> None:
    frame, manifest = run_preprocess()
    print(json.dumps(manifest, indent=2))
    print(f"\nSaved CSV → {OUT_DIR / 'bp_daily_forecasting.csv'}")
    print("\nGợi ý chia train/val/test (điều chỉnh theo năm bạn muốn):")
    print("  train: 2014-01-01 .. 2018-12-31")
    print("  val  : 2019-01-01 .. 2020-12-31")
    print("  test : 2021-01-01 .. cuối series")
    tr, va, te = time_based_split_indices(
        frame.index, train_end="2018-12-31", val_end="2020-12-31"
    )
    print(f"  sizes (days): train={len(tr)}, val={len(va)}, test={len(te)}")


if __name__ == "__main__":
    main()
