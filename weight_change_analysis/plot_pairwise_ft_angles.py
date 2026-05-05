#!/usr/bin/env python3
"""
Plot angle (degrees) between AdamW and Muon fine-tuning weight updates for LoRA vs full FT,
from weight_change_pairwise_angles.csv, for a chosen pretraining optimizer.

By default uses PCA score-space angles (angle_degrees_pca) when the CSV was produced with
run_analysis.py --pca-k. Use --full-space for the original full-parameter angles.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from weight_change_analysis.parsing import RunMeta, parse_run_folder  # noqa: E402


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def meta_pair(row: dict[str, str]) -> tuple[RunMeta, RunMeta] | None:
    """Parse run_i / run_j; return None if invalid."""
    try:
        a = parse_run_folder(row["run_i"])
        b = parse_run_folder(row["run_j"])
    except ValueError:
        return None
    if a.pretrain_optimizer != b.pretrain_optimizer:
        return None
    if a.finetune_type != b.finetune_type:
        return None
    if {a.finetune_optimizer, b.finetune_optimizer} != {"adamw", "muon"}:
        return None
    if a.name_suffix != b.name_suffix:
        return None
    return a, b


def pick_adamw_muon_angle_row(
    rows: list[dict[str, str]],
    pretrain: str,
    finetune_type: str,
    name_suffix: str | None,
) -> dict[str, str]:
    """
    Exactly one row: same pretrain (CSV + folder), lora or full, AdamW vs Muon, matching suffix.
    name_suffix None => require blank suffix for both runs.
    """
    pre = pretrain.strip().lower()
    ft = finetune_type.strip().lower()
    want_suf: str | None
    if name_suffix is not None:
        want_suf = name_suffix.strip()
    else:
        want_suf = ""

    matches: list[dict[str, str]] = []
    for row in rows:
        if row.get("pretrain_optimizer", "").lower() != pre:
            continue
        mp = meta_pair(row)
        if mp is None:
            continue
        a, b = mp
        if a.finetune_type != ft:
            continue
        suf = a.name_suffix
        if want_suf is not None and suf != want_suf:
            continue
        matches.append(row)

    if not matches:
        raise SystemExit(
            f"No AdamW–Muon angle row for pretrain={pre!r}, finetune_type={ft!r}, "
            f"name_suffix={want_suf!r}."
        )
    if len(matches) > 1:
        keys = [f"{m['run_i']} vs {m['run_j']}" for m in matches]
        raise SystemExit(
            f"Ambiguous: {len(matches)} rows match. Refine --name-suffix or CSV.\n"
            + "\n".join(f"  {k}" for k in keys)
        )
    return matches[0]


def csv_has_pca_columns(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    return "angle_degrees_pca" in rows[0] and "cosine_similarity_pca" in rows[0]


def pick_angle_cos(row: dict[str, str], use_pca: bool) -> tuple[float, float]:
    if use_pca:
        as_ = row.get("angle_degrees_pca", "").strip()
        cs = row.get("cosine_similarity_pca", "").strip()
        if not as_ or not cs:
            raise ValueError("missing PCA angle columns or empty values in this row")
        return float(as_), float(cs)
    return float(row["angle_degrees"]), float(row["cosine_similarity"])


def list_full_suffixes(rows: list[dict[str, str]], pretrain: str) -> None:
    pre = pretrain.lower()
    seen: set[str] = set()
    for row in rows:
        if row.get("pretrain_optimizer", "").lower() != pre:
            continue
        mp = meta_pair(row)
        if mp is None:
            continue
        a, _ = mp
        if a.finetune_type != "full":
            continue
        seen.add(a.name_suffix)
    if not seen:
        print(f"No full FT AdamW–Muon pairs for pretrain={pre!r}.", file=sys.stderr)
        return
    for suf in sorted(seen, key=lambda s: (s == "", s)):
        label = "(blank)" if suf == "" else repr(suf)
        print(label)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Bar plot: AdamW vs Muon FT update angles (LoRA vs full). "
        "Uses PCA score-space angles by default when CSV includes angle_degrees_pca; "
        "use --full-space for full-parameter angles."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("weight_change_analysis_output/weight_change_pairwise_angles.csv"),
        help="Path to weight_change_pairwise_angles.csv",
    )
    p.add_argument(
        "--pretrain",
        choices=("adam", "muon"),
        required=True,
        help="Pretraining optimizer family",
    )
    p.add_argument(
        "--name-suffix",
        default=None,
        metavar="SUFFIX",
        help="Select full FT pair by name_suffix (both runs). Blank primary: --name-suffix ''. "
        "LoRA uses the LoRA AdamW–Muon pair with blank name_suffix.",
    )
    p.add_argument(
        "--list-full-suffixes",
        action="store_true",
        help="Print available name_suffix values for full FT AdamW–Muon pairs, then exit",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path",
    )
    p.add_argument(
        "--full-space",
        action="store_true",
        help="Use full-parameter angles (angle_degrees) instead of PCA (angle_degrees_pca)",
    )
    args = p.parse_args(argv)

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    rows = load_rows(csv_path)
    if not rows:
        print("CSV has no data rows.", file=sys.stderr)
        return 1

    if args.list_full_suffixes:
        list_full_suffixes(rows, args.pretrain)
        return 0

    if args.full_space:
        use_pca = False
    else:
        if not csv_has_pca_columns(rows):
            print(
                "CSV has no PCA angle columns (angle_degrees_pca). "
                "Re-run: python3 weight_change_analysis/run_analysis.py ... --pca-k K\n"
                "Or pass --full-space to plot full-parameter angles.",
                file=sys.stderr,
            )
            return 1
        use_pca = True

    want_full_suffix: str | None
    if args.name_suffix is not None:
        want_full_suffix = args.name_suffix.strip()
    else:
        want_full_suffix = None

    try:
        row_lora = pick_adamw_muon_angle_row(rows, args.pretrain, "lora", name_suffix="")
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1

    try:
        row_full = pick_adamw_muon_angle_row(
            rows, args.pretrain, "full", name_suffix=want_full_suffix
        )
    except SystemExit as e:
        print(e, file=sys.stderr)
        print(
            "Hint: use --list-full-suffixes and pass --name-suffix for full FT if needed.",
            file=sys.stderr,
        )
        return 1

    try:
        ang_l, cos_l = pick_angle_cos(row_lora, use_pca)
        ang_f, cos_f = pick_angle_cos(row_full, use_pca)
    except (ValueError, KeyError) as e:
        print(f"Could not read angles from CSV row: {e}", file=sys.stderr)
        return 1

    for name, v in (("LoRA", ang_l), ("Full FT", ang_f)):
        if isinstance(v, float) and math.isnan(v):
            print(f"{name} angle is NaN in CSV; cannot plot.", file=sys.stderr)
            return 1

    ang_l_f = float(ang_l)
    ang_f_f = float(ang_f)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb
    except ImportError:
        print("matplotlib is required.", file=sys.stderr)
        return 1

    labels = ["LoRA", "Full FT"]
    values = [ang_l_f, ang_f_f]
    cos_vals = [cos_l, cos_f]
    # Same green family as norm plots; LoRA lighter (alpha), full FT solid — like --lora-and-full
    green = to_rgb("#2f855a")
    colors = [(*green, 0.55), (*green, 1.0)]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="tight")
    x = [0, 1]
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6, width=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Angle (degrees)" + (" (PCA score space)" if use_pca else " (full Δ)"))
    ymax = max(100.0, max(values) * 1.08)
    ax.set_ylim(0, ymax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.55, color="gray", zorder=1)

    suf_f = parse_run_folder(row_full["run_i"]).name_suffix
    suf_note = f", full FT suffix={suf_f!r}" if suf_f else ""
    if use_pca:
        kt = row_lora.get("pca_target_k", "?")
        ku = row_lora.get("pca_components_used", "?")
        pca_note = f" — PCA K={kt}, k_eff={ku}"
    else:
        pca_note = ""
    ax.set_title(
        f"Angle between AdamW vs Muon FT updates — {args.pretrain} pretrain{suf_note}{pca_note}",
        fontsize=11,
    )

    for bar, v, cos_t in zip(bars, values, cos_vals):
        ax.annotate(
            f"{v:.2f}°",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.annotate(
            f"cos = {cos_t:.4g}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_y()),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7,
            color="gray",
        )

    fig.subplots_adjust(bottom=0.14)

    out = args.output
    if out is None:
        suf_tag = suf_f or "default"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in suf_tag)
        tag = "pca" if use_pca else "full"
        out = csv_path.parent / f"angles_adamw_muon_{tag}_{args.pretrain}_{safe}.png"
    else:
        out = out.resolve()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
