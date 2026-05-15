import os
import glob
import re
import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

# =========================================================
# Regex
# =========================================================

STEP_RE = re.compile(r"step_(\d+)")
BASE_RE = re.compile(r"BASE_(.+?)_hf_step_\d+")
CKPT_RE = re.compile(
    r"(.+?)_ckpt_driving_(fullft_)?(.+?)_step_\d+"
)

# =========================================================
# Output dirs
# =========================================================

INDIVIDUAL_DIR = "individual_plots"
GROUP_DIR = "group_plots"

# =========================================================
# Helpers
# =========================================================

def normalize_optimizer(name):
    name = name.lower()

    mapping = {
        "adam": "Adam",
        "adamw": "AdamW",
        "muon": "Muon",
    }

    return mapping.get(name, name)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_step(filename):
    m = STEP_RE.search(filename)
    return int(m.group(1)) if m else 0


def parse_filename(filename):
    """
    Returns:
        {
            pretrain_optimizer,
            finetune_optimizer,
            ft_type,
            is_base
        }
    """

    base_match = BASE_RE.search(filename)

    # -----------------------------------------------------
    # Base models
    # -----------------------------------------------------
    if base_match:
        name = base_match.group(1).lower()

        if "adam" in name:
            opt = "Adam"
        elif "muon" in name:
            opt = "Muon"
        else:
            opt = name

        return {
            "pretrain_optimizer": opt,
            "finetune_optimizer": opt,
            "ft_type": "Base",
            "is_base": True
        }

    # -----------------------------------------------------
    # Fine-tuned checkpoints
    # -----------------------------------------------------
    m = CKPT_RE.search(filename)

    if not m:
        return None

    pretrain_opt = m.group(1)
    fullft = m.group(2)
    finetune_opt = m.group(3)

    ft_type = "Full FT" if fullft else "LoRA"

    return {
        "pretrain_optimizer": normalize_optimizer(pretrain_opt),
        "finetune_optimizer": normalize_optimizer(finetune_opt),
        "ft_type": ft_type,
        "is_base": False
    }


# =========================================================
# Metric selection
# =========================================================

def pick_metric(task_dict):
    keys = set(task_dict.keys())

    def find(prefix):
        return [k for k in keys if k.startswith(prefix)]

    # pass@1
    for k in keys:
        if k.startswith("pass@1") or k.startswith("pass_at_1"):
            return k

    # exact match
    if "exact_match,flexible-extract" in keys:
        return "exact_match,flexible-extract"

    if "exact_match,custom-extract" in keys:
        return "exact_match,custom-extract"

    if "exact_match,strict-match" in keys:
        return "exact_match,strict-match"

    exacts = find("exact_match")
    if exacts:
        return exacts[0]

    # accuracy
    if "acc,none" in keys:
        return "acc,none"

    if "acc_norm,none" in keys:
        return "acc_norm,none"

    return None


def get_stderr_key(metric_key):
    return metric_key + "_stderr"


def clean_metric_name(metric_key):
    """
    Makes legend/title cleaner.
    """

    if metric_key.startswith("pass@1"):
        return "Pass@1"

    if metric_key.startswith("pass_at_1"):
        return "Pass@1"

    if metric_key.startswith("exact_match"):
        return "Exact Match"

    if metric_key.startswith("acc_norm"):
        return "Normalized Accuracy"

    if metric_key.startswith("acc"):
        return "Accuracy"

    return metric_key


# =========================================================
# Collect data
# =========================================================

def collect(eval_dir):
    pattern = os.path.join(eval_dir, "*.json")
    files = glob.glob(pattern)

    # -----------------------------------------------------
    # Individual plots
    #
    # key:
    # (task, metric_name, label)
    # -----------------------------------------------------

    individual_data = defaultdict(list)

    # -----------------------------------------------------
    # Group plots
    #
    # key:
    # (pretrain_opt, ft_type, task, metric_name)
    #
    # value:
    #   dict[label] -> runs
    # -----------------------------------------------------

    group_data = defaultdict(lambda: defaultdict(list))

    # -----------------------------------------------------
    # Base values cache
    # -----------------------------------------------------

    base_scores = {}

    for f in files:

        filename = os.path.basename(f)

        parsed = parse_filename(filename)

        if parsed is None:
            print(f"[WARN] Could not parse {filename}")
            continue

        step = extract_step(filename)

        j = load_json(f)

        pretrain_opt = parsed["pretrain_optimizer"]
        finetune_opt = parsed["finetune_optimizer"]
        ft_type = parsed["ft_type"]
        is_base = parsed["is_base"]

        label = f"{pretrain_opt}->{finetune_opt} ({ft_type})"

        for task, metrics in j.items():

            metric_key = pick_metric(metrics)

            if metric_key is None:
                continue

            metric_name = clean_metric_name(metric_key)

            value = metrics.get(metric_key)

            if value is None:
                continue

            stderr = metrics.get(get_stderr_key(metric_key), 0.0)

            # -------------------------------------------------
            # Store base checkpoints
            # -------------------------------------------------

            if is_base:

                base_scores[(pretrain_opt, task, metric_name)] = {
                    "step": 0,
                    "value": value,
                    "stderr": stderr
                }

                continue

            # -------------------------------------------------
            # Add FT checkpoint
            # -------------------------------------------------

            run = {
                "step": step,
                "value": value,
                "stderr": stderr
            }

            individual_key = (
                task,
                metric_name,
                label
            )

            individual_data[individual_key].append(run)

            group_key = (
                pretrain_opt,
                ft_type,
                task,
                metric_name
            )

            group_data[group_key][label].append(run)

    # =====================================================
    # Inject base points
    # =====================================================

    for (task, metric_name, label), runs in individual_data.items():

        pretrain_opt = label.split("->")[0]

        base_key = (pretrain_opt, task, metric_name)

        if base_key in base_scores:
            runs.append(base_scores[base_key])

        runs.sort(key=lambda x: x["step"])

    for group_key, models in group_data.items():

        pretrain_opt, _, task, metric_name = group_key

        base_key = (pretrain_opt, task, metric_name)

        base_point = base_scores.get(base_key)

        for label in models:

            if base_point is not None:
                models[label].append(base_point)

            models[label].sort(key=lambda x: x["step"])

    return individual_data, group_data


# =========================================================
# Plotting
# =========================================================

def safe_name(x):
    return (
        x.replace("/", "_")
         .replace(" ", "_")
         .replace("(", "")
         .replace(")", "")
         .replace(":", "_")
         .lower()
    )

# =========================================================
# Fixed colors
# =========================================================

# COLOR_MAP = {
#     "Adam->AdamW (LoRA)": "tab:purple",
#     "Adam->Muon (LoRA)": "tab:pink",
#     "Muon->AdamW (LoRA)": "tab:brown",
#     "Muon->Muon (LoRA)": "tab:gray",

#     "Adam->AdamW (Full FT)": "deepskyblue",
#     "Adam->Muon (Full FT)": "darkorange",
#     "Muon->AdamW (Full FT)": "limegreen",
#     "Muon->Muon (Full FT)": "darkred",
# }
COLOR_MAP = {
    "Adam->AdamW (LoRA)":   "#7C6FCD",
    "Adam->Muon (LoRA)":    "#C4699A",
    "Muon->AdamW (LoRA)":   "#3EA88A",
    "Muon->Muon (LoRA)":    "#7AB648",
    "Adam->AdamW (Full FT)": "#4A3DB5",
    "Adam->Muon (Full FT)":  "#A03070",
    "Muon->AdamW (Full FT)": "#0F7A5C",
    "Muon->Muon (Full FT)":  "#4E8A18",
}

def make_individual_plots(individual_data, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for (task, metric_name, label), runs in individual_data.items():

        plt.figure(figsize=(8, 5))

        x = [r["step"] for r in runs]
        y = [r["value"] for r in runs]
        yerr = [r["stderr"] for r in runs]

        color = COLOR_MAP.get(label, "black")

        # Main line
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=6,
            label=f"{label}, {task}",
            color=color
        )

        # Confidence band
        lower = [a - e for a, e in zip(y, yerr)]
        upper = [a + e for a, e in zip(y, yerr)]

        plt.fill_between(
            x,
            lower,
            upper,
            alpha=0.2
        )

        plt.title("Measuring Forgetting Over Fine-Tuning Steps")
        plt.xlabel("Fine-Tuning Steps")
        plt.ylabel(metric_name)

        plt.grid(True)
        plt.legend()

        filename = (
            f"{safe_name(task)}_"
            f"{safe_name(metric_name)}_"
            f"{safe_name(label)}.png"
        )

        out_path = os.path.join(output_dir, filename)

        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] {out_path}")


def make_group_plots(group_data, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for group_key, models in group_data.items():

        pretrain_opt, ft_type, task, metric_name = group_key

        plt.figure(figsize=(9, 6))

        for label, runs in models.items():

            x = [r["step"] for r in runs]
            y = [r["value"] for r in runs]
            yerr = [r["stderr"] for r in runs]

            color = COLOR_MAP.get(label, "black")
            if color == "black":
                print(label)
                raise ImportError()

            # Main line
            plt.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                markersize=6,
                label=label,
                color=color
            )

            # Confidence band
            lower = [a - e for a, e in zip(y, yerr)]
            upper = [a + e for a, e in zip(y, yerr)]

            plt.fill_between(
                x,
                lower,
                upper,
                alpha=0.2
            )

        plt.title(
            f"Forgetting with {pretrain_opt} Pretraining "
            f"({ft_type})\n{task}"
        )

        plt.xlabel("Fine-Tuning Steps")
        plt.ylabel(metric_name)

        plt.grid(True)
        plt.legend()

        filename = (
            f"{safe_name(pretrain_opt)}_"
            f"{safe_name(ft_type)}_"
            f"{safe_name(task)}_"
            f"{safe_name(metric_name)}.png"
        )

        out_path = os.path.join(output_dir, filename)

        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] {out_path}")

# def make_individual_plots(individual_data, output_dir):

#     os.makedirs(output_dir, exist_ok=True)

#     for (task, metric_name, label), runs in individual_data.items():

#         plt.figure(figsize=(8, 5))

#         x = [r["step"] for r in runs]
#         y = [r["value"] for r in runs]
#         yerr = [r["stderr"] for r in runs]

#         plt.errorbar(
#             x,
#             y,
#             yerr=yerr,
#             marker="o",
#             capsize=3,
#             label=f"{label}, {task}"
#         )

#         plt.title("Measuring Forgetting Over Fine-Tuning Steps")
#         plt.xlabel("Fine-Tuning Steps")
#         plt.ylabel(metric_name)

#         plt.grid(True)
#         plt.legend()

#         filename = (
#             f"{safe_name(task)}_"
#             f"{safe_name(metric_name)}_"
#             f"{safe_name(label)}.png"
#         )

#         out_path = os.path.join(output_dir, filename)

#         plt.savefig(out_path, dpi=200, bbox_inches="tight")
#         plt.close()

#         print(f"[SAVED] {out_path}")


# def make_group_plots(group_data, output_dir):

#     os.makedirs(output_dir, exist_ok=True)

#     for group_key, models in group_data.items():

#         pretrain_opt, ft_type, task, metric_name = group_key

#         plt.figure(figsize=(9, 6))

#         for label, runs in models.items():

#             x = [r["step"] for r in runs]
#             y = [r["value"] for r in runs]
#             yerr = [r["stderr"] for r in runs]

#             plt.errorbar(
#                 x,
#                 y,
#                 yerr=yerr,
#                 marker="o",
#                 capsize=3,
#                 label=label
#             )

#         plt.title(
#             f"Forgetting with {pretrain_opt} Pretraining "
#             f"({ft_type})\n{task}"
#         )

#         plt.xlabel("Fine-Tuning Steps")
#         plt.ylabel(metric_name)

#         plt.grid(True)
#         plt.legend()

#         filename = (
#             f"{safe_name(pretrain_opt)}_"
#             f"{safe_name(ft_type)}_"
#             f"{safe_name(task)}_"
#             f"{safe_name(metric_name)}.png"
#         )

#         out_path = os.path.join(output_dir, filename)

#         plt.savefig(out_path, dpi=200, bbox_inches="tight")
#         plt.close()

#         print(f"[SAVED] {out_path}")


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "eval_dir",
        type=str,
        help="Directory containing evaluation JSON files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Root output directory"
    )

    args = parser.parse_args()

    individual_dir = os.path.join(
        args.output_dir,
        INDIVIDUAL_DIR
    )

    group_dir = os.path.join(
        args.output_dir,
        GROUP_DIR
    )

    individual_data, group_data = collect(args.eval_dir)

    make_individual_plots(
        individual_data,
        individual_dir
    )

    make_group_plots(
        group_data,
        group_dir
    )

    print("\nDone.")


if __name__ == "__main__":
    main()