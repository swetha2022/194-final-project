#!/usr/bin/env bash
# 2 (pretrain checkpoint) x 2 (finetune optimizer) x 3 (seeds) = 12 SFT runs for error bars.
# Moonlight YAMLs use full-parameter SFT (lora_cfg.enabled=false).
# Checkpoints default under YOUR scratch so runs stay off NFS home:
#   /scratch/katiewang/nemo_rl_results/sft_2x2_c4_fullft/...
# Override: CHECKPOINT_ROOT=/scratch/checkpoints/my_run ./examples/scripts/...
#
# Pretrain axis (celinetan-style HF trees on scratch):
#   MUON_PRETRAINED  – e.g. .../Moonlight_hf_step_42000
#   ADAM_PRETRAINED  – e.g. .../Moonlight_adam_hf_step_42000
#
# Finetune optimizer axis:
#   AdamW – examples/configs/sft_moonlight_c4_webtext_openai.yaml
#   Muon  – examples/configs/sft_moonlight_c4_webtext_openai_muon_ft.yaml
#           (torch.optim.Muon; requires PyTorch >= 2.9)
#
# From nemo-rl repo root (same env pattern as DPO: examples/scripts/run_dpo_nano_smoke.sh):
#   source ./env.sh                    # venv CUDA libs (nvjitlink) before system CUDA
#   export MUON_PRETRAINED=/scratch/katiewang/moonlight_weights/Moonlight_hf_step_42000
#   export ADAM_PRETRAINED=/scratch/katiewang/moonlight_weights/Moonlight_adam_hf_step_42000
#   export TRAIN_JSONL=data/moonlight_c4_sft/train.jsonl
#   export VAL_JSONL=data/moonlight_c4_sft/val.jsonl
#   ./examples/scripts/run_moonlight_c4_2x2_pretrain_optimizer_grid.sh
#
# Use real newlines between shell commands (not ``ray stop --forceexport ...`` on one line).
#
# DRY_RUN=1 prints commands only.
#
# Step budget (overrides yaml): default 400; e.g. MAX_NUM_STEPS=2000 ./examples/scripts/... for longer runs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

# CUDA wheel libs + Ray: mirrors run_dpo_nano_smoke.sh / ./env.sh (fixes nvJitLink import
# mismatches and accidental join of a remote Ray cluster).
if [[ -f "${ROOT}/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/env.sh"
else
  _SITE="${ROOT}/.venv/lib/python3.12/site-packages"
  export LD_LIBRARY_PATH="${_SITE}/nvidia/nvjitlink/lib:${_SITE}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
fi
unset RAY_ADDRESS
export NEMO_RL_RAY_LOCAL_ONLY="${NEMO_RL_RAY_LOCAL_ONLY:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export NEMO_RL_PY_EXECUTABLE_AUTOMODEL="${NEMO_RL_PY_EXECUTABLE_AUTOMODEL:-${ROOT}/.venv/bin/python}"

MUON_PRETRAINED="${MUON_PRETRAINED:?Set MUON_PRETRAINED to Muon-pretrained HF checkpoint dir}"
ADAM_PRETRAINED="${ADAM_PRETRAINED:?Set ADAM_PRETRAINED to Adam-pretrained HF checkpoint dir}"
TRAIN_JSONL="${TRAIN_JSONL:-data/moonlight_c4_sft/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/moonlight_c4_sft/val.jsonl}"

CFG_ADAMW="${ROOT}/examples/configs/sft_moonlight_c4_webtext_openai.yaml"
CFG_MUON="${ROOT}/examples/configs/sft_moonlight_c4_webtext_openai_muon_ft.yaml"

DRY_RUN="${DRY_RUN:-0}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-400}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/scratch/katiewang/nemo_rl_results}"
mkdir -p "${CHECKPOINT_ROOT}/sft_2x2_c4_fullft"

run_one() {
  local pretrain_label="$1"
  local pretrain_path="$2"
  local ft_label="$3"
  local cfg="$4"
  local seed="$5"

  local short="${pretrain_label}__${ft_label}__seed${seed}"
  # Separate tree from LoRA runs so resume/load_state_dict layouts cannot collide.
  local ckpt_dir="${CHECKPOINT_ROOT}/sft_2x2_c4_fullft/${short}"
  local wandb_name="sft-2x2-c4-fullft-${short}"

  local -a cmd=(
    uv run examples/run_sft.py
    --config "${cfg}"
    "policy.model_name=${pretrain_path}"
    "policy.tokenizer.name=${pretrain_path}"
    "data.train.data_path=${TRAIN_JSONL}"
    "data.validation.data_path=${VAL_JSONL}"
    "sft.seed=${seed}"
    "sft.max_num_steps=${MAX_NUM_STEPS}"
    "checkpointing.checkpoint_dir=${ckpt_dir}"
    "logger.wandb.name=${wandb_name}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi
  "${cmd[@]}"
}

for seed in 42 43 44; do
  run_one "muon_pt" "${MUON_PRETRAINED}" "adamw_ft" "${CFG_ADAMW}" "${seed}"
  run_one "muon_pt" "${MUON_PRETRAINED}" "muon_ft" "${CFG_MUON}" "${seed}"
  run_one "adam_pt" "${ADAM_PRETRAINED}" "adamw_ft" "${CFG_ADAMW}" "${seed}"
  run_one "adam_pt" "${ADAM_PRETRAINED}" "muon_ft" "${CFG_MUON}" "${seed}"
done

echo "Done: 12 runs (4 cells x 3 seeds)."
