from __future__ import annotations

import torch


def _first_differences_target(y_true: torch.Tensor, y_end: torch.Tensor) -> torch.Tensor:
    """
    TDAlign TDT for a horizon vector.

    y_true: (B, H)
    y_end:  (B,) last observed target at window end
    returns d_true: (B, H) where
      d[:,0] = y_true[:,0] - y_end
      d[:,i] = y_true[:,i] - y_true[:,i-1]
    """
    d = torch.empty_like(y_true)
    d[:, 0] = y_true[:, 0] - y_end
    if y_true.shape[1] > 1:
        d[:, 1:] = y_true[:, 1:] - y_true[:, :-1]
    return d


def _first_differences_pred(y_hat: torch.Tensor, y_end: torch.Tensor) -> torch.Tensor:
    """TDAlign TDP for a horizon vector (same convention as the paper)."""
    d = torch.empty_like(y_hat)
    d[:, 0] = y_hat[:, 0] - y_end
    if y_hat.shape[1] > 1:
        d[:, 1:] = y_hat[:, 1:] - y_hat[:, :-1]
    return d


def tdalign_losses(
    *,
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
    y_end: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    loss: str = "mse",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute (L_Y, L_D, rho) for TDAlign.

    y_true/y_hat: (B,H)
    y_end: (B,)
    mask: optional (B,H) bool indicating which horizon entries are valid
    weights: optional (B,H) float weights in [0,1] for each horizon entry
    loss: 'mse' or 'mae'
    """
    if y_true.ndim != 2 or y_hat.ndim != 2:
        raise ValueError("y_true and y_hat must be 2D (B,H)")
    if y_true.shape != y_hat.shape:
        raise ValueError("y_true and y_hat must have the same shape")
    if y_end.ndim != 1 or y_end.shape[0] != y_true.shape[0]:
        raise ValueError("y_end must be shape (B,)")

    if mask is None:
        mask = torch.isfinite(y_true) & torch.isfinite(y_hat)
    else:
        mask = mask & torch.isfinite(y_true) & torch.isfinite(y_hat)

    if weights is not None:
        if weights.shape != y_true.shape:
            raise ValueError("weights must have shape (B,H)")
        w = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights)).clamp(min=0.0)
        mask = mask & (w > 0)
    else:
        w = None

    if mask.sum() == 0:
        nan = torch.tensor(float("nan"), device=y_true.device)
        return nan, nan, nan

    err = y_hat - y_true
    if loss == "mse":
        if w is None:
            l_y = (err[mask] ** 2).mean()
        else:
            ww = w[mask]
            l_y = ((err[mask] ** 2) * ww).sum() / ww.sum().clamp_min(1e-12)
    elif loss == "mae":
        if w is None:
            l_y = err[mask].abs().mean()
        else:
            ww = w[mask]
            l_y = (err[mask].abs() * ww).sum() / ww.sum().clamp_min(1e-12)
    else:
        raise ValueError(loss)

    d_true = _first_differences_target(y_true, y_end)
    d_hat = _first_differences_pred(y_hat, y_end)
    d_mask = mask.clone()

    d_err = d_hat - d_true
    if loss == "mse":
        if w is None:
            l_d = (d_err[d_mask] ** 2).mean()
        else:
            ww = w[d_mask]
            l_d = ((d_err[d_mask] ** 2) * ww).sum() / ww.sum().clamp_min(1e-12)
    else:
        if w is None:
            l_d = d_err[d_mask].abs().mean()
        else:
            ww = w[d_mask]
            l_d = (d_err[d_mask].abs() * ww).sum() / ww.sum().clamp_min(1e-12)

    # rho: sign inconsistency ratio on valid entries
    rho = (torch.sign(d_true[d_mask]) != torch.sign(d_hat[d_mask])).float().mean()
    return l_y, l_d, rho


def tdalign_total_loss(
    *,
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
    y_end: torch.Tensor,
    mask: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    loss: str = "mse",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (L_total, L_Y, L_D, rho) with adaptive mixing:
      L = rho * L_Y + (1-rho) * L_D
    """
    l_y, l_d, rho = tdalign_losses(
        y_true=y_true, y_hat=y_hat, y_end=y_end, mask=mask, weights=weights, loss=loss
    )
    l = rho * l_y + (1.0 - rho) * l_d
    return l, l_y, l_d, rho


__all__ = ["tdalign_losses", "tdalign_total_loss"]

