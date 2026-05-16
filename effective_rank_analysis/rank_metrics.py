"""Effective rank and stable rank from singular values."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def effective_rank_from_singular_values(s: torch.Tensor) -> float:
    """
    Shannon / von Neumann style effective rank:
        exp( - sum_i p_i log p_i ),  p_i = sigma_i / sum_j sigma_j,
    with sigma_i >= 0 singular values of the matrix.
    """
    s = s.float()
    s = s.clamp(min=0.0)
    total = float(s.sum().item())
    if total <= 0.0 or s.numel() == 0:
        return 0.0
    p = (s / total).double()
    p = p[p > 0]
    if p.numel() == 0:
        return 0.0
    entropy = float(-(p * p.log()).sum().item())
    return float(math.exp(entropy))


def stable_rank_from_singular_values(s: torch.Tensor) -> float:
    """||W||_F^2 / ||W||_2^2 = (sum sigma_i^2) / sigma_max^2."""
    s = s.float()
    if s.numel() == 0:
        return 0.0
    smax = float(s[0].item()) if s.numel() else 0.0
    if smax <= 0.0:
        return 0.0
    num = float((s * s).sum().item())
    return num / (smax * smax)


def ranks_for_matrix_2d(
    w: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, int]:
    """
    Move 2D float tensor to `device`, compute singular values, return
    (effective_rank, stable_rank, rank_width) where rank_width = min(m, n).
    """
    if w.ndim != 2:
        raise ValueError("ranks_for_matrix_2d expects a 2D tensor")
    w32 = w.float().to(device)
    s = torch.linalg.svdvals(w32)
    eff = effective_rank_from_singular_values(s.cpu())
    stab = stable_rank_from_singular_values(s.cpu())
    k = int(s.numel())
    del w32, s
    return eff, stab, k
