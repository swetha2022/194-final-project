"""Norms and streaming accumulation for weight deltas."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch


def rms_rms_induced_2d(delta: torch.Tensor) -> float:
    """
    Operator norm of ΔW as a map (R^{n_in}, ||·||_rms) -> (R^{n_out}, ||·||_rms).

    For W with shape (out_features, in_features) = (m, n):
        ||ΔW||_{rms→rms} = sqrt(n/m) * σ_max(ΔW)
    """
    if delta.ndim != 2:
        raise ValueError("rms→rms induced norm is defined for 2D tensors only")
    m, n = delta.shape
    sigma = spectral_norm_chunked(delta)
    return math.sqrt(n / m) * sigma


def spectral_norm_chunked(
    w: torch.Tensor,
    n_iter: int = 32,
    row_chunk: int = 8192,
) -> float:
    """
    Largest singular value of W (spectral / operator norm), float32 power iteration.
    Row-chunked matmuls so large matrices do not need the full W @ W.T in memory.
    """
    if w.ndim != 2:
        raise ValueError
    w = w.float()
    m, n = w.shape
    device = w.device
    v = torch.randn(n, 1, device=device, dtype=torch.float32)
    v /= v.norm() + 1e-12
    for _ in range(n_iter):
        u = torch.zeros(m, 1, device=device, dtype=torch.float32)
        for r0 in range(0, m, row_chunk):
            r1 = min(m, r0 + row_chunk)
            u[r0:r1] = w[r0:r1] @ v
        nu = u.norm() + 1e-12
        u /= nu
        v_new = torch.zeros(n, 1, device=device, dtype=torch.float32)
        for r0 in range(0, m, row_chunk):
            r1 = min(m, r0 + row_chunk)
            v_new += w[r0:r1].T @ u[r0:r1]
        nv = v_new.norm() + 1e-12
        v = v_new / nv
    u = torch.zeros(m, 1, device=device, dtype=torch.float32)
    for r0 in range(0, m, row_chunk):
        r1 = min(m, r0 + row_chunk)
        u[r0:r1] = w[r0:r1] @ v
    return float(u.norm().item())


def accumulate_multi_dot_and_norms(
    base: torch.Tensor,
    fts: Sequence[torch.Tensor],
    chunk_elems: int = 4_000_000,
) -> Tuple[List[float], List[List[float]], List[float]]:
    """
    For tensors base, fts[k] of identical shape, compute per-run ||Δ_k||^2,
    pairwise <Δ_i, Δ_j> (upper triangle), and max |Δ_k| on this tensor.

    Returns:
      norm_sq: length len(fts)
      dots: symmetric matrix as nested lists dots[i][j] for i<=j
      linf_this_tensor: max absolute delta per run on this tensor
    """
    r = len(fts)
    norm_sq = [0.0] * r
    dots = [[0.0] * r for _ in range(r)]
    linf_t = [0.0] * r
    if base.shape != fts[0].shape:
        raise ValueError
    flat_b = base.reshape(-1)
    n = flat_b.numel()
    i = 0
    while i < n:
        j = min(n, i + chunk_elems)
        bchunk = flat_b[i:j].float()
        deltas = [fts[k].reshape(-1)[i:j].float() - bchunk for k in range(r)]
        for k in range(r):
            dk = deltas[k]
            norm_sq[k] += float(dk.pow(2).sum().item())
            linf_t[k] = max(linf_t[k], float(dk.abs().max().item()))
            for l in range(k, r):
                dots[k][l] += float((dk * deltas[l]).sum().item())
        i = j
    return norm_sq, dots, linf_t


def global_max_rms_induced_for_tensor(
    base: torch.Tensor,
    ft: torch.Tensor,
    spectral_device: torch.device | None = None,
) -> float:
    """Return rms→rms induced norm for this tensor if 2D, else 0."""
    if base.ndim != 2:
        return 0.0
    delta = ft.float() - base.float()
    dev = spectral_device or torch.device("cpu")
    if dev.type == "cuda":
        delta = delta.to(dev)
    return rms_rms_induced_2d(delta)
