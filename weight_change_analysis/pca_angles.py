"""PCA on weight-update vectors using only the m×m Gram matrix of inner products.

We never materialize full parameter vectors (dimension d). For m finetune runs in a
pretrain group, G_ij = <δ_i, δ_j> is accumulated during the tensor pass. Classical
PCA on centered rows would use the d×d covariance; the dual formulation uses the
double-centered Gram H G H (m×m), whose top eigenvectors yield m-dimensional score
vectors Z ∈ R^{m×k} with rows z_i ∈ R^k. Pairwise angles in R^k use cos∠(z_i,z_j).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def gram_from_upper_triangle(dots_upper: List[List[float]], r: int) -> np.ndarray:
    """Symmetric Gram G[i,j] = <delta_i, delta_j> from upper-triangle accumulation."""
    g = np.zeros((r, r), dtype=np.float64)
    for i in range(r):
        for j in range(i, r):
            v = float(dots_upper[i][j])
            g[i, j] = v
            g[j, i] = v
    return g


def double_center_gram(g: np.ndarray) -> np.ndarray:
    """H G H with H = I - (1/m) 11^T (same as centering rows/columns of the implied data)."""
    m = g.shape[0]
    if m == 0:
        return g
    row_mean = g.mean(axis=1, keepdims=True)
    col_mean = g.mean(axis=0, keepdims=True)
    grand = float(g.mean())
    return g - row_mean - col_mean + grand


def pca_scores_from_centered_gram(gc: np.ndarray, k: int) -> Tuple[np.ndarray, int]:
    """
    Dual PCA: scores Z (m × k_eff) with rows in R^{k_eff}, preserving inner products
    on the leading eigenspace (up to k components with positive eigenvalues).

    Returns (Z, k_eff_used).
    """
    m = gc.shape[0]
    if m < 2 or k < 1:
        return np.zeros((m, 0)), 0

    w, v = np.linalg.eigh(gc)
    # descending eigenvalues
    idx = np.argsort(w)[::-1]
    w = w[idx].astype(np.float64)
    v = v[:, idx]

    if w[0] <= 0.0:
        return np.zeros((m, 0)), 0

    thresh = max(1e-12, 1e-10 * w[0])
    k_eff = 0
    for t in range(min(k, m)):
        if w[t] <= thresh:
            break
        k_eff = t + 1
    if k_eff == 0:
        return np.zeros((m, 0)), 0

    lam = np.clip(w[:k_eff], 0.0, None)
    z = v[:, :k_eff] * np.sqrt(lam)
    return z, k_eff


def cosine_and_angle_degrees(u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return float("nan"), float("nan")
    c = float(np.dot(u, v) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return c, math.degrees(math.acos(c))
