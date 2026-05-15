import lm_eval
import os
import json
import time
import subprocess
import requests
import signal
import argparse

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

tasks = [
    "mmlu","mmlu_pro","bbh","triviaqa","humaneval",
    "mbpp","gsm8k","hendrycks_math","ceval-valid"
]

out_dir = "/home/swetharajkumar/eval_results_katie/"
os.makedirs(out_dir, exist_ok=True)

root_dir = "/scratch/katiewang/ft_out/"

default_tokenizer = "/scratch/celine/moonlight_weights/Moonlight_adam_hf_step_42000"

# ----------------------------
# ARGPARSE
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--all_steps",
    action="store_true",
    help="Evaluate all checkpoints instead of selected subset"
)

args = parser.parse_args()

# ----------------------------
# BASE CHECKPOINTS (RUN ONCE)
# ----------------------------
base_ckpts = [
    "/scratch/celine/moonlight_weights/Moonlight_adam_hf_step_42000",
    "/scratch/celine/moonlight_weights/Moonlight_hf_step_42000",
]


# ----------------------------
# UTIL: STEP DISCOVERY
# ----------------------------
def get_step_ckpts(run_dir):
    steps = []

    for d in os.listdir(run_dir):
        if not d.startswith("step_"):
            continue

        merged_path = os.path.join(
            run_dir, d,
            "policy/weights/model/merged"
        )
        consolidated_path = os.path.join(
            run_dir, d,
            "policy/weights/model/consolidated"
        )

        if os.path.exists(merged_path):
            steps.append(merged_path)
        if os.path.exists(consolidated_path):
            steps.append(consolidated_path)

    def step_key(p):
        return int(p.split("step_")[1].split("/")[0])

    return sorted(steps, key=step_key)


def select_steps(step_ckpts):
    if len(step_ckpts) < 4:
        return step_ckpts

    preferred = {"step_800", "step_1200", "step_2000"}

    selected = []
    for ckpt in step_ckpts:
        step_name = None
        for p in ckpt.split("/"):
            if p.startswith("step_"):
                step_name = p
                break

        if step_name in preferred:
            selected.append(ckpt)

    return selected


# ----------------------------
# EVAL FUNCTION
# ----------------------------
def run_eval(ckpt, out_file, tasks_to_run):

    existing_results = {}
    missing_tasks = tasks_to_run.copy()

    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                existing_results = json.load(f)
            missing_tasks = [t for t in tasks_to_run if t not in existing_results]
        except Exception as e:
            print(f"Failed loading {out_file}: {e}")

    if not missing_tasks:
        print(f"Already complete: {ckpt}")
        return

    print(f"Running eval: {ckpt}")
    print(f"Tasks: {missing_tasks}")
    
    tokenizer_path = (
        ckpt if os.path.exists(os.path.join(ckpt, "tokenizer_config.json"))
        else default_tokenizer
    )

    model_args = (
        f"pretrained={ckpt},"
        f"tokenizer={tokenizer_path},"
        f"trust_remote_code=True,"
        f"data_parallel_size=8,"
        f"gpu_memory_utilization=0.8,"
    )
    # model_args = (
    #     f"pretrained={ckpt},"
    #     f"trust_remote_code=True,"
    #     f"base_url=http://127.0.0.1:8000/v1/completions,"
    # )

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        device="cuda:0,1,2,3,4,5,6,7",
        batch_size="auto:8",
        confirm_run_unsafe_code=True,
        tasks=missing_tasks,
    )

    if "results" in results:
        filtered = {
            k: v for k, v in results["results"].items()
            if k in missing_tasks
        }
    else:
        filtered = results

    existing_results.update(filtered)

    with open(out_file, "w") as f:
        json.dump(existing_results, f, indent=4)

    print(f"Saved: {out_file}")


# ----------------------------
# RUN BASE CHECKPOINTS (ONCE)
# ----------------------------
print("\n=== BASE CHECKPOINTS ===")

for ckpt in base_ckpts:
    if not os.path.exists(ckpt):
        print(f"Missing base ckpt: {ckpt}")
        continue

    name = ckpt.split("/")[-1]
    out_file = os.path.join(out_dir, f"BASE_{name}.json")

    run_eval(ckpt, out_file, tasks)


# ----------------------------
# FIND RUNS
# ----------------------------
# runs = set()

# for root, _, files in os.walk(root_dir):
#     if any(f.endswith(".safetensors") for f in files) and "merged" in root:
#         run_root = root.split("/policy/weights/model/merged")[0]
#         runs.add(run_root)

runs = [
    os.path.join(root_dir, d)
    for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
]

# ----------------------------
# RUN EVALS
# ----------------------------
for run in sorted(runs):
    print(f"\n=== RUN: {run} ===")

    step_ckpts = get_step_ckpts(run)

    if not step_ckpts:
        print("No checkpoints found.")
        continue

    if args.all_steps:
        selected_ckpts = step_ckpts
    else:
        selected_ckpts = select_steps(step_ckpts)

    print("Selected ckpts:")
    for c in selected_ckpts:
        print(" ", c)

    for ckpt in selected_ckpts:

        rel = os.path.relpath(ckpt, root_dir)
        out_file = os.path.join(out_dir, rel.replace("/", "_") + ".json")

        run_eval(ckpt, out_file, tasks)