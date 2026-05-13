#!/usr/bin/env python3
"""
Diagnostics for PCA dimension selection and layer-type filtering utilities.

Addresses reviewer critiques:
  - Justify the choice of PCA k (log cumulative variance explained)
  - Identify which layer types contribute to RMS norms
  - Filter analysis to specific layer categories
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Layer-type classification
# ---------------------------------------------------------------------------

# Patterns that identify layer categories from safetensors key names.
# Keys typically look like:
#   model.layers.0.self_attn.q_proj.weight
#   model.layers.0.mlp.gate_proj.weight
#   model.embed_tokens.weight
#   lm_head.weight
LAYER_CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "attention_qkv": ["q_proj", "k_proj", "v_proj"],
    "attention_out": ["o_proj", "out_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj", "fc1", "fc2", "ffn"],
    "embed": ["embed_tokens", "embed", "wte", "wpe"],
    "lm_head": ["lm_head"],
    "norm": ["norm", "ln_"],
}


def classify_key(key: str) -> str:
    """
    Return a layer-category label for a safetensors tensor key.

    Categories (in priority order):
      attention_qkv, attention_out, mlp, embed, lm_head, norm, other
    """
    k = key.lower()
    for category, patterns in LAYER_CATEGORY_PATTERNS.items():
        if any(p in k for p in patterns):
            return category
    return "other"


def filter_keys_by_category(
    keys: List[str],
    categories: Optional[List[str]] = None,
) -> List[str]:
    """
    Return only the keys whose layer category is in `categories`.
    If `categories` is None, all keys are returned (no filtering).

    Example:
        filter_keys_by_category(keys, ["attention_qkv", "attention_out"])
    """
    if categories is None:
        return keys
    cat_set = set(categories)
    return [k for k in keys if classify_key(k) in cat_set]


def summarize_key_categories(keys: List[str]) -> Dict[str, int]:
    """Return a dict mapping category -> count of keys in that category."""
    counts: Dict[str, int] = {}
    for k in keys:
        cat = classify_key(k)
        counts[cat] = counts.get(cat, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# PCA variance-explained diagnostics
# ---------------------------------------------------------------------------


def variance_explained_ratios(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Cumulative fraction of variance explained by the first k components,
    considering only positive eigenvalues (negative ones arise from numerical
    noise in the double-centred Gram and carry no variance).

    Parameters
    ----------
    eigenvalues : 1-D array, in descending order (as returned by eigh after argsort).

    Returns
    -------
    cumvar : 1-D array of length == number of positive eigenvalues,
             cumvar[i] = fraction of total variance explained by components 0..i.
    """
    pos = eigenvalues[eigenvalues > 0].astype(np.float64)
    if pos.size == 0:
        return np.array([], dtype=np.float64)
    total = pos.sum()
    return np.cumsum(pos) / total


def pca_scores_with_diagnostics(
    gc: np.ndarray,
    k: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Dual PCA on the double-centred Gram matrix, with variance-explained logging.

    Parameters
    ----------
    gc      : double-centred Gram matrix (m × m).
    k       : target number of PCA components.
    verbose : if True, print a variance-explained table to stderr.

    Returns
    -------
    z       : score matrix (m × k_eff).
    k_eff   : number of components actually used (≤ k).
    cumvar  : cumulative variance ratios for all positive components.
    """
    import sys

    m = gc.shape[0]
    if m < 2 or k < 1:
        return np.zeros((m, 0)), 0, np.array([])

    w, v = np.linalg.eigh(gc)
    # descending
    idx = np.argsort(w)[::-1]
    w = w[idx].astype(np.float64)
    v = v[:, idx]

    cumvar = variance_explained_ratios(w)

    if verbose:
        n_pos = len(cumvar)
        print(
            f"\n[PCA] {m} runs → {n_pos} positive eigenvalue(s).",
            file=sys.stderr,
        )
        print(
            f"[PCA] Cumulative variance explained (up to k={min(k, n_pos)}):",
            file=sys.stderr,
        )
        header = f"  {'k':>4}  {'λ_k':>14}  {'cumvar':>8}"
        print(header, file=sys.stderr)
        for i in range(min(k, n_pos)):
            pct = cumvar[i] * 100.0
            print(f"  {i+1:>4}  {w[i]:>14.6g}  {pct:>7.2f}%", file=sys.stderr)
        if n_pos > k:
            pct_total = cumvar[min(k, n_pos) - 1] * 100.0
            print(
                f"[PCA] k={k} captures {pct_total:.2f}% of variance "
                f"({n_pos - k} positive component(s) discarded).",
                file=sys.stderr,
            )
        else:
            print(
                f"[PCA] k={k} ≥ {n_pos} positive components; using k_eff={n_pos}.",
                file=sys.stderr,
            )
        print("", file=sys.stderr)

    thresh = max(1e-12, 1e-10 * w[0]) if w[0] > 0 else 1e-12
    k_eff = 0
    for t in range(min(k, m)):
        if w[t] <= thresh:
            break
        k_eff = t + 1

    if k_eff == 0:
        return np.zeros((m, 0)), 0, cumvar

    lam = np.clip(w[:k_eff], 0.0, None)
    z = v[:, :k_eff] * np.sqrt(lam)
    return z, k_eff, cumvar


def recommend_k(cumvar: np.ndarray, threshold: float = 0.90) -> int:
    """
    Return the smallest k such that cumulative variance ≥ threshold.
    Returns len(cumvar) if the threshold is never reached.

    Useful for justifying a specific k choice in plots / logs.
    """
    for i, cv in enumerate(cumvar):
        if cv >= threshold:
            return i + 1
    return len(cumvar)


# ---------------------------------------------------------------------------
# RMS-norm layer inventory
# ---------------------------------------------------------------------------


def log_rms_contributions(
    key_rms_map: Dict[str, float],
    top_n: int = 10,
) -> None:
    """
    Print the top-N layers by RMS→RMS induced norm to stderr, so reviewers
    can see *which* weight matrices dominate the reported max.

    Parameters
    ----------
    key_rms_map : dict mapping tensor key -> rms_rms_induced value for that layer.
    top_n       : how many top layers to display.
    """
    import sys

    if not key_rms_map:
        return
    ranked = sorted(key_rms_map.items(), key=lambda x: x[1], reverse=True)
    print(f"\n[RMS] Top-{min(top_n, len(ranked))} layers by RMS→RMS induced norm:", file=sys.stderr)
    print(f"  {'rank':>4}  {'rms_induced':>14}  {'category':>16}  key", file=sys.stderr)
    for rank, (key, val) in enumerate(ranked[:top_n], 1):
        cat = classify_key(key)
        print(f"  {rank:>4}  {val:>14.6g}  {cat:>16}  {key}", file=sys.stderr)
    print("", file=sys.stderr)