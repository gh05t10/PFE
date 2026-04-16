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


__all__ = ["WindowNPZDataset"]

