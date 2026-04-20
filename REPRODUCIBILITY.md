# Khôi phục & tái lập kết quả (PFE)

## 1. Git là “điểm mốc” chính

- **Commit** sau mỗi bước quan trọng (preprocess xong, train xong một baseline/slide).
- Ghi **message** rõ: ví dụ `research(results): window L96 H1 S1 + snapshot`.
- Để lấy lại đúng **code** đã chạy: `git checkout <commit>`.

## 2. Snapshot pipeline (metadata + hash file nhỏ)

Sau khi chạy xong resample / normalize / window / train:

```bash
cd /home/peter/PFE
python run_pipeline_snapshot.py
```

Tạo:

- `artifacts/pipeline_snapshots/snapshot_<timestamp>.json` — bản ghi đầy đủ lần đó;
- `artifacts/pipeline_snapshots/latest.json` — bản mới nhất (ghi đè).

Trong file có:

- **Git**: `commit`, `branch`, `dirty` (có thay đổi chưa commit không).
- **Môi trường**: Python, `PFE_RESAMPLE_FREQ` nếu có.
- **File theo dõi**: `resample_meta.txt`, `split_manifest.json`, `scaler_params.json`, `baseline_metrics.json`, `train_config*.json`, `metrics*.csv`, v.v. — kèm **SHA256** (file nhỏ) hoặc prefix 1MB (file lớn).
- **Artifact lớn**: `.npz`, `.pt` — chỉ **kích thước + mtime** (đủ để biết có đổi file không).

**Nên** add & commit các file `artifacts/pipeline_snapshots/*.json` nếu muốn lưu lịch sử trong repo (dung lượng nhỏ).

## 3. Tái lại **đúng** dữ liệu đã xử lý

Thứ tự phụ thuộc **cùng** tham số đã ghi trong `split_manifest.json`, `resample_meta.txt`, thư mục `windowed_L*_H*_S*`.

1. Checkout commit (hoặc đọc snapshot để biết branch/commit).
2. Chạy lại pipeline với **cùng** `--freq`, `--rule-a`, `--train-end`, `--val-end`, `context_len`, `horizon_steps`, `stride`.
3. So khớp **SHA256** trong snapshot với file mới tạo (nếu khác → dữ liệu hoặc code đã lệch).

## 4. Tái lại kết quả train

- Trong snapshot có đường dẫn tới `checkpoints/best_gru.pt`, `best_slide.pt` và `train_config*.json`.
- **Seed**: `run_train_baseline.py` / `run_train_slide.py` dùng `--seed` (mặc định 42); cùng seed + cùng data → gần như tái lập (GPU vẫn có thể khác nhẹ).
- Lưu **cùng** `requirements.txt`; có thể thêm `pip freeze > requirements-lock.txt` khi chốt luận văn.

## 5. Không commit toàn bộ `processed/`?

- Dữ liệu lớn thường **không** đưa lên git; khi đó snapshot JSON + **mô tả lệnh** trong báo cáo là bằng chứng “đã chạy gì”.
- Nếu cần archive: nén thư mục `processed/chl_shallow/` và lưu cùng `snapshot_*.json` trên ổ/đám mây.
