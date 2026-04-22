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
    # Student-only (setup B): build K/V from x5 only (no Chl in-window).
    kv_encoder_layers: int | None = None  # default: same as encoder_layers
    share_q_kv_encoder: bool = False


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
        # mask True but value NaN/Inf still poisons Transformer — zero those out
        ok = mask & torch.isfinite(x)
        x = torch.where(ok, x, torch.zeros_like(x))
        x = x.view(B, P, patch_len, C).reshape(B, P, patch_len * C)
        # At least one valid timestep×channel in the patch (not ">50% valid")
        m = mask.view(B, P, patch_len, C).any(dim=(2, 3))
        return x, m

    @staticmethod
    def _no_all_masked(pad: torch.Tensor) -> torch.Tensor:
        """
        Transformer / MHA: if every position is padding (all True), softmax can become NaN.
        Unmask one slot per row that was fully masked.
        ``pad``: True = ignore (PyTorch convention for key_padding / src_key_padding).
        """
        if not pad.any():
            return pad
        all_bad = pad.all(dim=1)
        if not all_bad.any():
            return pad
        out = pad.clone()
        out[all_bad, -1] = False
        return out

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
        pad5 = self._no_all_masked(~v5)
        pad6 = self._no_all_masked(~v6)

        h5 = self.embed5(x5p)
        h6 = self.embed6(x6p)
        h5 = self.enc5(h5, src_key_padding_mask=pad5)
        h6 = self.enc6(h6, src_key_padding_mask=pad6)

        pad6_k = self._no_all_masked(pad6)
        attn_out, _ = self.cross(h5, h6, h6, key_padding_mask=pad6_k)
        out = self.norm(attn_out + h5)
        pooled = out.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class SlideStudentCrossAttn(nn.Module):
    """
    Setup B (deployable): only consumes x5/mask5 at inference.

    We keep the same patching + Transformer encoder structure, but learn a separate
    K/V path from x5 (instead of x6 which includes Chl in-window).
    """

    def __init__(self, cfg: SlidePatchCrossAttnConfig) -> None:
        super().__init__()
        self.cfg = cfg
        pl = cfg.patch_len
        dm = cfg.d_model
        nh = cfg.nhead
        df = 4 * dm
        drop = cfg.dropout

        self.embed5_q = nn.Linear(pl * cfg.n_vars5, dm)
        self.embed5_kv = nn.Linear(pl * cfg.n_vars5, dm)

        el_q = nn.TransformerEncoderLayer(dm, nh, df, drop, batch_first=True)
        self.enc_q = nn.TransformerEncoder(el_q, num_layers=cfg.encoder_layers)

        if cfg.share_q_kv_encoder:
            self.enc_kv = self.enc_q
        else:
            kv_layers = cfg.kv_encoder_layers if cfg.kv_encoder_layers is not None else cfg.encoder_layers
            el_kv = nn.TransformerEncoderLayer(dm, nh, df, drop, batch_first=True)
            self.enc_kv = nn.TransformerEncoder(el_kv, num_layers=kv_layers)

        self.cross = nn.MultiheadAttention(dm, nh, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dm)
        self.head = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, 1))

    @staticmethod
    def _fold(
        x: torch.Tensor,
        mask: torch.Tensor,
        patch_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return SlidePatchCrossAttn._fold(x, mask, patch_len)

    @staticmethod
    def _no_all_masked(pad: torch.Tensor) -> torch.Tensor:
        return SlidePatchCrossAttn._no_all_masked(pad)

    def forward(self, x5: torch.Tensor, mask5: torch.Tensor) -> torch.Tensor:
        """
        x5, mask5: (B, L, 5).
        Returns ŷ_z: (B,).
        """
        pl = self.cfg.patch_len
        x5p, v5 = self._fold(x5, mask5, pl)
        pad = self._no_all_masked(~v5)

        q = self.embed5_q(x5p)
        kv = self.embed5_kv(x5p)
        q = self.enc_q(q, src_key_padding_mask=pad)
        kv = self.enc_kv(kv, src_key_padding_mask=pad)

        pad_k = self._no_all_masked(pad)
        attn_out, _ = self.cross(q, kv, kv, key_padding_mask=pad_k)
        out = self.norm(attn_out + q)
        pooled = out.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class SlideStudentResidual(nn.Module):
    """
    Setup B + residual: deployable student predicts
      - current Chl at window end: y_end_hat_z (from x5 only)
      - delta to horizon: delta_hat_z
    and returns y_future_hat_z = y_end_hat_z + delta_hat_z.
    """

    def __init__(self, cfg: SlidePatchCrossAttnConfig) -> None:
        super().__init__()
        self.cfg = cfg
        pl = cfg.patch_len
        dm = cfg.d_model
        nh = cfg.nhead
        df = 4 * dm
        drop = cfg.dropout

        self.embed5_q = nn.Linear(pl * cfg.n_vars5, dm)
        self.embed5_kv = nn.Linear(pl * cfg.n_vars5, dm)

        el_q = nn.TransformerEncoderLayer(dm, nh, df, drop, batch_first=True)
        self.enc_q = nn.TransformerEncoder(el_q, num_layers=cfg.encoder_layers)

        if cfg.share_q_kv_encoder:
            self.enc_kv = self.enc_q
        else:
            kv_layers = cfg.kv_encoder_layers if cfg.kv_encoder_layers is not None else cfg.encoder_layers
            el_kv = nn.TransformerEncoderLayer(dm, nh, df, drop, batch_first=True)
            self.enc_kv = nn.TransformerEncoder(el_kv, num_layers=kv_layers)

        self.cross = nn.MultiheadAttention(dm, nh, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dm)

        self.head_end = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, 1))
        self.head_delta = nn.Sequential(nn.Linear(dm, dm), nn.ReLU(), nn.Linear(dm, 1))

    @staticmethod
    def _fold(x: torch.Tensor, mask: torch.Tensor, patch_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return SlidePatchCrossAttn._fold(x, mask, patch_len)

    @staticmethod
    def _no_all_masked(pad: torch.Tensor) -> torch.Tensor:
        return SlidePatchCrossAttn._no_all_masked(pad)

    def forward(self, x5: torch.Tensor, mask5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (y_future_hat_z, y_end_hat_z, delta_hat_z), each (B,).
        """
        pl = self.cfg.patch_len
        x5p, v5 = self._fold(x5, mask5, pl)
        pad = self._no_all_masked(~v5)

        q = self.embed5_q(x5p)
        kv = self.embed5_kv(x5p)
        q = self.enc_q(q, src_key_padding_mask=pad)
        kv = self.enc_kv(kv, src_key_padding_mask=pad)

        pad_k = self._no_all_masked(pad)
        attn_out, _ = self.cross(q, kv, kv, key_padding_mask=pad_k)
        out = self.norm(attn_out + q)
        pooled = out.mean(dim=1)

        y_end = self.head_end(pooled).squeeze(-1)
        delta = self.head_delta(pooled).squeeze(-1)
        return y_end + delta, y_end, delta


__all__ = [
    "SlidePatchCrossAttn",
    "SlideStudentCrossAttn",
    "SlideStudentResidual",
    "SlidePatchCrossAttnConfig",
]
