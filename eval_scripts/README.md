# Evaluation & Plotting Pipeline

This repo contains two scripts for running LLM evaluations and visualizing forgetting across fine-tuning steps.

---

## 1. `run_evals.py` — Running Evaluations

### Setup

Follow the [lm-evaluation-harness setup instructions](https://github.com/EleutherAI/lm-evaluation-harness) to install the library. Then place `run_evals.py` into the root of the cloned `lm-evaluation-harness` repo and run it from there.

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
# Place run_evals.py here, then run:
python run_evals.py [OPTIONS]
```

### Configuration

Before running, update the following paths near the top of `run_evals.py` to match your environment:

| Variable | Description |
|---|---|
| `out_dir` | Directory where evaluation JSON results will be saved |
| `root_dir` | Root directory containing your fine-tuned checkpoint runs |
| `default_tokenizer` | Path to a fallback tokenizer if a checkpoint lacks one |
| `base_ckpts` | List of base (pre-fine-tuning) checkpoint paths to evaluate |

### Usage

```bash
# Evaluate only the preferred subset of steps (step_800, step_1200, step_2000)
python run_evals.py

# Evaluate ALL discovered checkpoints under root_dir
python run_evals.py --all_steps
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--all_steps` | flag | off | If set, evaluates every discovered checkpoint. Otherwise, evaluates only steps matching `step_800`, `step_1200`, or `step_2000`. |

### What it does

- Evaluates **base checkpoints** once (listed in `base_ckpts`).
- Discovers fine-tuned checkpoint runs under `root_dir`, finds `merged` or `consolidated` model weights inside each `step_*` subdirectory, and runs evaluation on the selected steps.
- Skips tasks that already have results saved in the output JSON (safe to resume after interruptions).
- Evaluates on the following tasks: `mmlu`, `mmlu_pro`, `bbh`, `triviaqa`, `humaneval`, `mbpp`, `gsm8k`, `hendrycks_math`, `ceval-valid`.
- Results are saved as JSON files in `out_dir`, one file per checkpoint.

### Expected checkpoint structure

```
root_dir/
  <run_name>/
    step_800/
      policy/weights/model/merged/   ← or consolidated/
    step_1200/
      policy/weights/model/merged/
    ...
```

---

## 2. `plot_forgetting.py` — Plotting Results

This script reads the JSON evaluation outputs from `run_evals.py` and produces forgetting curves across fine-tuning steps.

### Usage

```bash
python plot_forgetting.py <eval_dir> [--output_dir <output_dir>]
```

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `eval_dir` | positional (required) | — | Directory containing the evaluation JSON files produced by `run_evals.py` |
| `--output_dir` | optional | `.` (current directory) | Root directory where plots will be saved |

### Example

```bash
python plot_forgetting.py /home/user/eval_results/ --output_dir /home/user/plots/
```

### Outputs

Two subdirectories are created under `--output_dir`:

```
<output_dir>/
  individual_plots/   ← One plot per (task, metric, model config)
  group_plots/        ← One plot per (pretrain optimizer, FT type, task, metric)
                         comparing all fine-tuning optimizers on the same axes
```

### Expected filename format

The script infers model metadata from the JSON filenames produced by `run_evals.py`. Filenames should match one of these patterns:

**Base checkpoints:**
```
BASE_<optimizer>_hf_step_<N>.json
```

**Fine-tuned checkpoints:**
```
<pretrain_opt>_ckpt_driving_[fullft_]<finetune_opt>_step_<N>.json
```

Supported optimizer names: `adam`, `adamw`, `muon` (case-insensitive).

### Supported metrics

The script auto-selects the best available metric per task in this priority order:

1. `pass@1` / `pass_at_1`
2. `exact_match` (various extractors)
3. `acc,none`
4. `acc_norm,none`
