#!/usr/bin/env bash
# Build C4 web-text OpenAI JSONL and launch Moonlight SFT (NeMo RL), following the
# same openai_format + local JSONL pattern as celinetan's driving SFT pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DATA_DIR="${DATA_DIR:-${ROOT}/data/moonlight_c4_sft}"
CONFIG="${CONFIG:-${ROOT}/examples/configs/sft_moonlight_c4_webtext_openai.yaml}"

echo "[1/2] Preparing JSONL under ${DATA_DIR} ..."
uv run python scripts/prepare_webtext_openai_sft.py \
  --output_dir "${DATA_DIR}" \
  "$@"

echo "[2/2] Launching SFT ..."
uv run examples/run_sft.py --config "${CONFIG}" \
  data.train.data_path="${DATA_DIR}/train.jsonl" \
  data.validation.data_path="${DATA_DIR}/val.jsonl"
