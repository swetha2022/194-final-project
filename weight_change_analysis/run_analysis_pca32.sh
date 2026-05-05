#!/usr/bin/env bash
# Full weight-change analysis with PCA-based pairwise angles (K=32).
# Re-run from project root: bash weight_change_analysis/run_analysis_pca32.sh

set -euo pipefail
cd "$(dirname "$0")/.."

exec python3 weight_change_analysis/run_analysis.py \
  --ft-root /scratch/celine/ft_out \
  --output-dir /home/ehharrison/projects/final_project/weight_change_analysis_output \
  --cuda-spectral \
  --pca-k 5 \
  --progress
