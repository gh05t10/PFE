"""
PyTorch Dataset wrappers around windowed NPZ files produced by ``run_build_window_dataset.py``.

Each sample: X_z (L×5), X_mask (L×5), and either:
  - legacy scalar target: y_z (scalar), y_mask (scalar), target_time
  - multi-step target:   Y_z (H,), Y_mask (H,), target_times

Both are exposed when present in the NPZ. Training scripts can choose which to use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowNPZDataset(Dataset):
    """
    Dataset reading a single NPZ bundle, keeping arrays in memory.

    Expected keys: ``X_z``, ``X_mask``, and at least one of:
      - ``y_z`` / ``y_mask`` (legacy scalar)
      - ``Y_z`` / ``Y_mask`` (multi-step)
    """

    def __init__(self, npz_path: str | Path) -> None:
        super().__init__()
        self.path = Path(npz_path)
        data = np.load(self.path, allow_pickle=False)
        self.X = torch.from_numpy(data["X_z"]).float()
        self.X_mask = torch.from_numpy(data["X_mask"]).bool()
        self.y = (
            torch.from_numpy(data["y_z"]).float()
            if "y_z" in data.files
            else None
        )
        self.y_mask = (
            torch.from_numpy(data["y_mask"]).bool()
            if "y_mask" in data.files
            else None
        )
        self.Y = (
            torch.from_numpy(data["Y_z"]).float()
            if "Y_z" in data.files
            else None
        )
        self.Y_mask = (
            torch.from_numpy(data["Y_mask"]).bool()
            if "Y_mask" in data.files
            else None
        )
        self.Y_w = (
            torch.from_numpy(data["Y_w"]).float()
            if "Y_w" in data.files
            else None
        )

        n = self.X.shape[0]
        if self.y is not None and self.y.shape[0] != n:
            raise ValueError("X and y have different number of samples.")
        if self.Y is not None and self.Y.shape[0] != n:
            raise ValueError("X and Y have different number of samples.")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            "x": self.X[idx],          # (L, 5)
            "x_mask": self.X_mask[idx],
        }
        if self.y is not None:
            out["y"] = self.y[idx]  # ()
        if self.y_mask is not None:
            out["y_mask"] = self.y_mask[idx]
        if self.Y is not None:
            out["y_seq"] = self.Y[idx]  # (H,)
        if self.Y_mask is not None:
            out["y_seq_mask"] = self.Y_mask[idx]
        if self.Y_w is not None:
            out["y_seq_w"] = self.Y_w[idx]
        return out


class SlideWindowNPZDataset(Dataset):
    """
    Same as ``WindowNPZDataset`` plus ``X6_z`` / ``X6_mask`` for the calibration branch (slide 10).

    Requires NPZ built after ``window_dataset`` includes ``X6_z`` — re-run ``run_build_window_dataset.py``.
    """

    def __init__(self, npz_path: str | Path) -> None:
        super().__init__()
        self.path = Path(npz_path)
        data = np.load(self.path, allow_pickle=False)
        if "X6_z" not in data.files:
            raise KeyError(
                f"{npz_path} has no X6_z — rebuild windows with the current run_build_window_dataset.py"
            )
        self.X = torch.from_numpy(data["X_z"]).float()
        self.X_mask = torch.from_numpy(data["X_mask"]).bool()
        self.X6 = torch.from_numpy(data["X6_z"]).float()
        self.X6_mask = torch.from_numpy(data["X6_mask"]).bool()
        self.y = torch.from_numpy(data["y_z"]).float() if "y_z" in data.files else None
        self.y_mask = torch.from_numpy(data["y_mask"]).bool() if "y_mask" in data.files else None
        self.Y = torch.from_numpy(data["Y_z"]).float() if "Y_z" in data.files else None
        self.Y_mask = torch.from_numpy(data["Y_mask"]).bool() if "Y_mask" in data.files else None
        self.Y_w = torch.from_numpy(data["Y_w"]).float() if "Y_w" in data.files else None
        self.chl_end = (
            torch.from_numpy(data["chl_z_at_window_end"]).float()
            if "chl_z_at_window_end" in data.files
            else None
        )
        # Keep times internally; we will expose them as int64 nanoseconds for DataLoader collation.
        self.context_end_time = data["context_end_time"] if "context_end_time" in data.files else None
        self.target_time = data["target_time"] if "target_time" in data.files else None
        self.target_times = data["target_times"] if "target_times" in data.files else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out: dict[str, object] = {
            "x": self.X[idx],
            "x_mask": self.X_mask[idx],
            "x6": self.X6[idx],
            "x6_mask": self.X6_mask[idx],
        }
        if self.y is not None:
            out["y"] = self.y[idx]
        if self.y_mask is not None:
            out["y_mask"] = self.y_mask[idx]
        if self.Y is not None:
            out["y_seq"] = self.Y[idx]
        if self.Y_mask is not None:
            out["y_seq_mask"] = self.Y_mask[idx]
        if self.Y_w is not None:
            out["y_seq_w"] = self.Y_w[idx]
        if self.chl_end is not None:
            ce = self.chl_end[idx]
            out["chl_end"] = ce
            out["chl_end_mask"] = torch.isfinite(ce)
        if self.context_end_time is not None:
            # int64 nanoseconds since epoch; collates cleanly.
            out["context_end_time_ns"] = int(np.asarray(self.context_end_time[idx]).astype("datetime64[ns]").astype("int64"))
        if self.target_time is not None:
            out["target_time_ns"] = int(np.asarray(self.target_time[idx]).astype("datetime64[ns]").astype("int64"))
        if self.target_times is not None:
            # (H,) datetime64[ns] -> int64[ns]
            out["target_times_ns"] = np.asarray(self.target_times[idx]).astype("datetime64[ns]").astype("int64")
        return out  # type: ignore[return-value]


__all__ = ["WindowNPZDataset", "SlideWindowNPZDataset"]

