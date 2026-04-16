"""
Simple baseline models for shallow-Chl soft sensor.

Current default: GRU-based sequence-to-one regressor on z-scored inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GRUBaselineConfig:
    input_dim: int = 5
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1


class GRUBaseline(nn.Module):
    """
    GRU over (L, C) with simple masking: masked timesteps get zeroed before GRU.
    """

    def __init__(self, cfg: GRUBaselineConfig) -> None:
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
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C), x_mask: (B, L, C) True=valid.
        Returns y_pred_z: (B,)
        """
        # Invalid positions must be 0 without multiplying NaN*0 (PyTorch keeps NaN).
        x_valid = torch.where(x_mask, x, torch.zeros_like(x))
        out, h_n = self.gru(x_valid)
        # Use last hidden state from top layer
        last_h = h_n[-1]  # (B, hidden_dim)
        y = self.head(last_h).squeeze(-1)
        return y


__all__ = ["GRUBaselineConfig", "GRUBaseline"]

