#!/usr/bin/env bash
# Full weight-change analysis with PCA-based pairwise angles.
#
# PCA dimensionality (--pca-k 5)
# --------------------------------
# We project each run's weight-update vector Δᵢ into a K-dimensional score
# space via dual PCA on the double-centred Gram matrix G[i,j] = <Δᵢ,Δⱼ>.
# K=5 is a starting point: the script prints a cumulative variance-explained
# table to stderr during the run (see [PCA] log lines) so you can confirm
# how much variance K=5 captures and re-run with a larger or smaller K if
# needed.  The pca_variance_explained column in the output CSV records the
# fraction of variance captured for each pair.
#
# Recommended workflow:
#   1. Run once with --pca-k 5 (fast); check [PCA] stderr output.
#   2. If k=5 < 90% cumvar, re-run with a higher --pca-k.
#   3. The recommend_k() helper in pca_diagnostics.py automates this check.
#
# Layer filtering (--layer-categories)
# --------------------------------------
# By default the analysis runs over all shared tensors.  To restrict to
# specific layer types (e.g. to understand which sub-network drives the
# observed norms / angles), pass --layer-categories.  Examples:
#
#   Attention only:
#     --layer-categories attention_qkv,attention_out
#
#   MLP only:
#     --layer-categories mlp
#
#   Attention + MLP (excludes embed/norm/lm_head):
#     --layer-categories attention_qkv,attention_out,mlp
#
# The layer_filter column in both output CSVs records which filter was used.
# Available categories: attention_qkv, attention_out, mlp, embed, lm_head,
#                       norm, other  (see pca_diagnostics.py for patterns).
#
# Re-run from project root:
#   bash weight_change_analysis/run_analysis_pca32.sh
set -euo pipefail
cd "$(dirname "$0")/.."
 
exec python3 weight_change_analysis/run_analysis.py \
    --ft-root /scratch/celine/ft_out \
    --output-dir /home/swetharajkumar/weight_change_analysis_output \
    --cuda-spectral \
    --pca-k 5 \
    --progress
    # Add --layer-categories attention_qkv,attention_out,mlp to restrict to
    # attention and MLP layers only (recommended for layer-level comparison).