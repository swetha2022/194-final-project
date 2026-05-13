#!/usr/bin/env python3
"""
Plot angle (degrees) between AdamW and Muon fine-tuning weight updates for LoRA vs full FT.

How the angles are computed
----------------------------
For each fine-tuning run i we form the weight-update vector:

    Δᵢ = θ_finetuned_i  −  θ_pretrained

(flattened concatenation of all shared parameter tensors).  The full-space angle between
two runs' updates is:

    θ = cos⁻¹( <Δᵢ, Δⱼ> / (‖Δᵢ‖ · ‖Δⱼ‖) )

Both the dot product and the norms are accumulated during the same streaming pass over
tensor shards, so the billion-dimensional vectors are never materialised.

PCA-space angles (default when --pca-k was used in run_analysis.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Because many fine-tuning runs may share a large common direction (e.g. task adaptation)
the full-space cosine is often dominated by that shared component.  The PCA-space angle
isolates the *relative* geometry:

  1. Build the m×m Gram matrix G[i,j] = <Δᵢ, Δⱼ> (already in the CSV).
  2. Double-centre G  →  H·G·H  (removes the mean update direction).
  3. Eigendecompose; take the top-K score vectors  z_i ∈ R^K.
  4. Report  cos⁻¹( <zᵢ, zⱼ> / (‖zᵢ‖ · ‖zⱼ‖) )  as the PCA-space angle.

The CSV column pca_variance_explained records the cumulative variance fraction captured
by the K components used, justifying the choice of K.  Use --full-space for the raw
full-parameter angle.
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

_METHODOLOGY_CAPTION_FULL = (
    "Full-space angle: cos⁻¹(⟨Δ_AdamW, Δ_Muon⟩ / (‖Δ_AdamW‖·‖Δ_Muon‖)).\n"
    "Δ = finetuned − pretrained, streamed over all shared tensors."
)
_METHODOLOGY_CAPTION_PCA = (
    "PCA-space angle: cos⁻¹(⟨z_AdamW, z_Muon⟩ / (‖z_AdamW‖·‖z_Muon‖)),  "
    "where z ∈ Rᴷ are dual-PCA score vectors from the double-centred Gram matrix.\n"
    "Δ = finetuned − pretrained, streamed over all shared tensors.  "
    "See pca_variance_explained column for variance captured by K components."
)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def meta_pair(row: dict[str, str]) -> tuple[RunMeta, RunMeta] | None:
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
    pre = pretrain.strip().lower()
    ft = finetune_type.strip().lower()
    want_suf: str = name_suffix.strip() if name_suffix is not None else ""

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
        if a.name_suffix != want_suf:
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
            f"Ambiguous: {len(matches)} rows match. Refine --name-suffix.\n"
            + "\n".join(f"  {k}" for k in keys)
        )
    return matches[0]


def csv_has_pca_columns(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    return "angle_degrees_pca" in rows[0] and "cosine_similarity_pca" in rows[0]


def pick_angle_cos(
    row: dict[str, str], use_pca: bool
) -> tuple[float, float, float | None]:
    """Return (angle_degrees, cosine_similarity, pca_variance_explained_or_None)."""
    if use_pca:
        as_ = row.get("angle_degrees_pca", "").strip()
        cs = row.get("cosine_similarity_pca", "").strip()
        if not as_ or not cs:
            raise ValueError("missing PCA angle columns or empty values in this row")
        var_exp_s = row.get("pca_variance_explained", "").strip()
        var_exp = float(var_exp_s) if var_exp_s else None
        return float(as_), float(cs), var_exp
    return float(row["angle_degrees"]), float(row["cosine_similarity"]), None


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
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("weight_change_analysis_output/weight_change_pairwise_angles.csv"),
    )
    p.add_argument("--pretrain", choices=("adam", "muon"), required=True)
    p.add_argument("--name-suffix", default=None, metavar="SUFFIX")
    p.add_argument("--list-full-suffixes", action="store_true")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument(
        "--full-space",
        action="store_true",
        help="Use full-parameter angles (angle_degrees) instead of PCA score-space angles",
    )
    p.add_argument(
        "--no-caption",
        action="store_true",
        help="Suppress the methodology caption below the plot",
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
                "CSV has no PCA angle columns.  Re-run run_analysis.py with --pca-k K, "
                "or pass --full-space to use full-parameter angles.",
                file=sys.stderr,
            )
            return 1
        use_pca = True

    want_full_suffix = args.name_suffix.strip() if args.name_suffix is not None else None

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
        print("Hint: use --list-full-suffixes and pass --name-suffix.", file=sys.stderr)
        return 1

    try:
        ang_l, cos_l, var_l = pick_angle_cos(row_lora, use_pca)
        ang_f, cos_f, var_f = pick_angle_cos(row_full, use_pca)
    except (ValueError, KeyError) as e:
        print(f"Could not read angles from CSV row: {e}", file=sys.stderr)
        return 1

    for name, v in (("LoRA", ang_l), ("Full FT", ang_f)):
        if isinstance(v, float) and math.isnan(v):
            print(f"{name} angle is NaN in CSV; cannot plot.", file=sys.stderr)
            return 1

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb
    except ImportError:
        print("matplotlib is required.", file=sys.stderr)
        return 1

    labels = ["LoRA", "Full FT"]
    values = [float(ang_l), float(ang_f)]
    cos_vals = [cos_l, cos_f]
    var_vals = [var_l, var_f]
    green = to_rgb("#2f855a")
    colors = [(*green, 0.55), (*green, 1.0)]

    fig, ax = plt.subplots(figsize=(6.5, 5.0), layout="tight")
    x = [0, 1]
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6, width=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    # Y-axis label — was missing before
    if use_pca:
        ax.set_ylabel(
            "Angle between AdamW and Muon weight updates (degrees)\n"
            "[PCA score space — see caption for methodology]",
            fontsize=9,
            labelpad=6,
        )
    else:
        ax.set_ylabel(
            "Angle between AdamW and Muon weight updates (degrees)\n"
            r"[full parameter space: $\cos^{-1}(\langle\Delta_i,\Delta_j\rangle"
            r"/ \|\Delta_i\|\|\Delta_j\|)$]",
            fontsize=9,
            labelpad=6,
        )

    ymax = max(100.0, max(values) * 1.10)
    ax.set_ylim(0, ymax)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.55, color="gray", zorder=1)

    suf_f = parse_run_folder(row_full["run_i"]).name_suffix
    suf_note = f", full FT suffix={suf_f!r}" if suf_f else ""
    lf = (row_lora.get("layer_filter") or "all").strip()
    layer_note = f", layers: {lf}" if lf and lf != "all" else ""

    if use_pca:
        kt = row_lora.get("pca_target_k", "?")
        ku = row_lora.get("pca_components_used", "?")
        pca_note = f" — PCA K={kt}, k_eff={ku}"
    else:
        pca_note = ""
    ax.set_title(
        f"AdamW vs Muon weight-update angle — {args.pretrain} pretrain"
        f"{suf_note}{layer_note}{pca_note}",
        fontsize=10,
        pad=6,
    )

    for bar, v, cos_t, var_exp in zip(bars, values, cos_vals, var_vals):
        cx = bar.get_x() + bar.get_width() / 2
        # Angle value above bar
        ax.annotate(
            f"{v:.2f}°",
            xy=(cx, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
        # Cosine similarity below bar top
        ax.annotate(
            f"cos = {cos_t:.4g}",
            xy=(cx, bar.get_height()),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center", va="top",
            fontsize=8, color="gray",
        )
        # PCA variance explained (if available)
        if use_pca and var_exp is not None and not math.isnan(var_exp):
            ax.annotate(
                f"var = {var_exp*100:.1f}%",
                xy=(cx, bar.get_height()),
                xytext=(0, -24),
                textcoords="offset points",
                ha="center", va="top",
                fontsize=7, color="#777777",
            )

    # Methodology caption
    if not args.no_caption:
        caption = _METHODOLOGY_CAPTION_PCA if use_pca else _METHODOLOGY_CAPTION_FULL
        fig.text(
            0.5, -0.03,
            caption,
            ha="center", va="top",
            fontsize=7, color="#555555",
            style="italic",
        )

    fig.subplots_adjust(bottom=0.18)

    out = args.output
    if out is None:
        suf_tag = suf_f or "default"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in suf_tag)
        tag = "pca" if use_pca else "full"
        out = csv_path.parent / f"angles_adamw_muon_{tag}_{args.pretrain}_{safe}.png"
    else:
        out = out.resolve()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())