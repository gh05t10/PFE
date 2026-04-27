from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GRUStudentConfig:
    input_dim: int = 5
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    pred_len: int = 48


class GRUStudent(nn.Module):
    """
    Simple deployable student: GRU over X5 -> horizon vector in z-space.
    Designed to be numerically stable for long multi-step horizons.
    """

    def __init__(self, cfg: GRUStudentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.pred_len),
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # Avoid NaN*0 propagation
        x_valid = torch.where(x_mask & torch.isfinite(x), x, torch.zeros_like(x))
        _, h_n = self.gru(x_valid)
        last_h = h_n[-1]
        y = self.head(last_h)
        if self.cfg.pred_len == 1:
            return y.squeeze(-1)
        return y


__all__ = ["GRUStudent", "GRUStudentConfig"]


@dataclass
class MLPStudentConfig:
    input_dim: int = 5
    context_len: int = 96
    hidden_dim: int = 512
    dropout: float = 0.1
    pred_len: int = 48


class MLPStudent(nn.Module):
    """
    Very simple student: flatten (L,5) -> MLP -> horizon vector.
    This avoids RNN/Transformer backward instabilities on some CPU builds.
    """

    def __init__(self, cfg: MLPStudentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.context_len * cfg.input_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.pred_len),
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x_valid = torch.where(x_mask & torch.isfinite(x), x, torch.zeros_like(x))
        flat = x_valid.reshape(x_valid.shape[0], -1)
        y = self.net(flat)
        if self.cfg.pred_len == 1:
            return y.squeeze(-1)
        return y


__all__ = ["GRUStudent", "GRUStudentConfig", "MLPStudent", "MLPStudentConfig"]

