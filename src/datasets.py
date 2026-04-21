"""
PyTorch Dataset wrappers around windowed NPZ files produced by ``run_build_window_dataset.py``.

Each sample: X_z (L×5), X_mask (L×5), y_z (scalar), y_mask (scalar), context_end_time, target_time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowNPZDataset(Dataset):
    """
    Dataset reading a single NPZ bundle, keeping arrays in memory.

    Expected keys: ``X_z``, ``X_mask``, ``y_z``, ``y_mask``.
    """

    def __init__(self, npz_path: str | Path) -> None:
        super().__init__()
        self.path = Path(npz_path)
        data = np.load(self.path, allow_pickle=False)
        self.X = torch.from_numpy(data["X_z"]).float()
        self.X_mask = torch.from_numpy(data["X_mask"]).bool()
        self.y = torch.from_numpy(data["y_z"]).float()
        self.y_mask = torch.from_numpy(data["y_mask"]).bool()

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y have different number of samples.")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.X[idx],          # (L, 5)
            "x_mask": self.X_mask[idx],
            "y": self.y[idx],          # ()
            "y_mask": self.y_mask[idx],
        }


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
        self.y = torch.from_numpy(data["y_z"]).float()
        self.y_mask = torch.from_numpy(data["y_mask"]).bool()
        # Keep times internally; we will expose them as int64 nanoseconds for DataLoader collation.
        self.context_end_time = data["context_end_time"] if "context_end_time" in data.files else None
        self.target_time = data["target_time"] if "target_time" in data.files else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out: dict[str, object] = {
            "x": self.X[idx],
            "x_mask": self.X_mask[idx],
            "x6": self.X6[idx],
            "x6_mask": self.X6_mask[idx],
            "y": self.y[idx],
            "y_mask": self.y_mask[idx],
        }
        if self.context_end_time is not None:
            # int64 nanoseconds since epoch; collates cleanly.
            out["context_end_time_ns"] = int(np.asarray(self.context_end_time[idx]).astype("datetime64[ns]").astype("int64"))
        if self.target_time is not None:
            out["target_time_ns"] = int(np.asarray(self.target_time[idx]).astype("datetime64[ns]").astype("int64"))
        return out  # type: ignore[return-value]


__all__ = ["WindowNPZDataset", "SlideWindowNPZDataset"]

