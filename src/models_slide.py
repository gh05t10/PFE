"""
Slide-inspired architecture (Aquaculture-report17_1.pdf ~pages 9–10):

  - **Calibration / teacher:** 6 channels (5 water + Chl_z along the window) → patch embed → Transformer encoder → **K, V**.
  - **Soft-sensor:** 5 channels → patch embed → Transformer encoder → **Q**.
  - **Cross-attention:** Q attends to K, V; pooled → MLP head → scalar Chl (z-space).

Training uses full ``X6_z`` (teacher). Inference without Chl is not implemented here (would need a separate forward or masking strategy).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SlidePatchCrossAttnConfig:
    patch_len: int = 16
    d_model: int = 64
    nhead: int = 4
    encoder_layers: int = 2
    dropout: float = 0.1
    n_vars5: int = 5
    n_vars6: int = 6


class SlidePatchCrossAttn(nn.Module):
    def __init__(self, cfg: SlidePatchCrossAttnConfig) -> None:
        super().__init__()
        self.cfg = cfg
        pl = cfg.patch_len
        dm = cfg.d_model
        nh = cfg.nhead
        df = 4 * dm
        drop = cfg.dropout

        self.embed5 = nn.Linear(pl * cfg.n_vars5, dm)
        self.embed6 = nn.Linear(pl * cfg.n_vars6, dm)

        el5 = nn.TransformerEncoderLayer(dm, nh, df, drop, batch_first=True)
        el6 = nn.TransformerEncoderLayer(dm, nh, df, drop, batch_first=True)
        self.enc5 = nn.TransformerEncoder(el5, num_layers=cfg.encoder_layers)
        self.enc6 = nn.TransformerEncoder(el6, num_layers=cfg.encoder_layers)

        self.cross = nn.MultiheadAttention(dm, nh, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dm)
        self.head = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, 1))

    @staticmethod
    def _fold(
        x: torch.Tensor,
        mask: torch.Tensor,
        patch_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,L,C) → (B,P,C*patch_len), patch_valid (B,P)."""
        B, L, C = x.shape
        if L % patch_len != 0:
            raise ValueError(f"L={L} not divisible by patch_len={patch_len}")
        P = L // patch_len
        x = torch.where(mask, x, torch.zeros_like(x))
        x = x.view(B, P, patch_len, C).reshape(B, P, patch_len * C)
        m = mask.view(B, P, patch_len, C).float().mean(dim=(2, 3)) > 0.5
        return x, m

    def forward(
        self,
        x5: torch.Tensor,
        mask5: torch.Tensor,
        x6: torch.Tensor,
        mask6: torch.Tensor,
    ) -> torch.Tensor:
        """
        x5, mask5: (B, L, 5); x6, mask6: (B, L, 6).
        Returns ŷ_z: (B,).
        """
        pl = self.cfg.patch_len
        x5p, v5 = self._fold(x5, mask5, pl)
        x6p, v6 = self._fold(x6, mask6, pl)
        pad5 = ~v5
        pad6 = ~v6

        h5 = self.embed5(x5p)
        h6 = self.embed6(x6p)
        h5 = self.enc5(h5, src_key_padding_mask=pad5)
        h6 = self.enc6(h6, src_key_padding_mask=pad6)

        attn_out, _ = self.cross(h5, h6, h6, key_padding_mask=pad6)
        out = self.norm(attn_out + h5)
        pooled = out.mean(dim=1)
        return self.head(pooled).squeeze(-1)


__all__ = ["SlidePatchCrossAttn", "SlidePatchCrossAttnConfig"]
