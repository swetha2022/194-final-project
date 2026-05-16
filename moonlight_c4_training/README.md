# Moonlight C4 SFT

This folder mirrors the paths under a **NeMo RL** checkout so you can vendor it in your final-project repo or copy it back into `nemo-rl`.

## Contents

| Path in this bundle | Purpose |
|---------------------|---------|
| `scripts/prepare_webtext_openai_sft.py` | Build `train.jsonl` / `val.jsonl` from C4 (OpenAI chat format) |
| `examples/scripts/run_sft_moonlight_c4_webtext.sh` | Prep data + single SFT run |
| `examples/scripts/run_moonlight_c4_2x2_pretrain_optimizer_grid.sh` | Full-FT 2×2×3 grid (`sft_2x2_c4_fullft/`) |
| `examples/configs/sft_moonlight_c4_webtext_openai.yaml` | AdamW finetune recipe |
| `examples/configs/sft_moonlight_c4_webtext_openai_muon_ft.yaml` | Muon finetune recipe |

These YAMLs `defaults: - sft.yaml` — you still need a full **nemo-rl** tree with `examples/run_sft.py` and `examples/configs/sft.yaml`.

## Use inside nemo-rl

From your nemo-rl root, merge this bundle (overwrites same paths if present):

```bash
rsync -a cs194_moonlight_c4_bundle/ /path/to/nemo-rl/
```

Then (see script headers for env vars such as `MUON_PRETRAINED`, `ADAM_PRETRAINED`, `CHECKPOINT_ROOT`).
