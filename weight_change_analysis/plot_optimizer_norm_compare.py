#!/usr/bin/env python3
"""
Compare one norm of the weight update between AdamW and Muon fine-tuning optimizers.

Single mode: one of LoRA or full FT (--finetune-type).
Combined mode (--lora-and-full): grouped bars — per fine-tuning optimizer, LoRA and
Full FT side by side.

How norms are computed
-----------------------
Each bar shows ‖Δ‖ for one fine-tuning run, where Δ = finetuned − pretrained weights,
accumulated over all shared parameter tensors (streamed in chunks).

  L∞   : max |Δ[j]|  over all parameters j.
  L2   : √(Σ Δ[j]²).
  RMS→RMS : max over 2-D layers of √(n/m)·σ_max(ΔW),
             where σ_max is the largest singular value (power iteration).

Example commands
-----------------
  python3 weight_change_analysis/plot_optimizer_norm_compare.py \\
      --pretrain muon --lora-and-full --name-suffix '4-28' --norm rms
  python3 weight_change_analysis/plot_optimizer_norm_compare.py \\
      --pretrain adam --lora-and-full --norm l_inf
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from weight_change_analysis.plot_norms_bar import (  # noqa: E402
    NORM_ALIASES,
    NORM_SPECS,
    NORM_YLABEL,
    _METHODOLOGY_CAPTION,
)

_MULTI_NORM_YLABEL = r"$\|\Delta W\|$" + "  (see x-axis for norm type)"


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_finetune_type(s: str) -> str:
    t = s.strip().lower()
    if t in ("full", "fullft", "full_ft"):
        return "full"
    if t in ("lora",):
        return "lora"
    raise ValueError(f"finetune type must be lora or full, got {s!r}")


def canonical_norm(token: str) -> str:
    t = token.strip().lower().replace(" ", "")
    k = NORM_ALIASES.get(t)
    if k is None:
        allowed = ", ".join(sorted(set(NORM_ALIASES.keys())))
        raise SystemExit(f"Unknown norm {token!r}. Use one of: {allowed}")
    return k


def suffix_key(r: dict[str, str]) -> str:
    return (r.get("name_suffix") or "").strip()


def find_pairs(
    rows: list[dict[str, str]],
    pretrain: str,
    finetune_type: str,
) -> dict[str, tuple[dict[str, str], dict[str, str]]]:
    """Map name_suffix -> (row_adamw, row_muon) when both exist for that suffix."""
    pre = pretrain.strip().lower()
    ft = normalize_finetune_type(finetune_type)

    by_suffix: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"adamw": [], "muon": []}
    )
    for r in rows:
        if r.get("pretrain_optimizer", "").lower() != pre:
            continue
        if r.get("finetune_type", "").lower() != ft:
            continue
        opt = r.get("finetune_optimizer", "").lower()
        if opt not in ("adamw", "muon"):
            continue
        by_suffix[suffix_key(r)][opt].append(r)

    pairs: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
    for suf, d in by_suffix.items():
        aw, mu = d["adamw"], d["muon"]
        if not aw or not mu:
            continue
        if len(aw) > 1 or len(mu) > 1:
            raise ValueError(
                f"Ambiguous rows for pretrain={pre!r}, finetune_type={ft!r}, "
                f"name_suffix={suf!r}: multiple matches for the same optimizer."
            )
        pairs[suf] = (aw[0], mu[0])
    return pairs


def resolve_lora_full_rows(
    rows: list[dict[str, str]],
    pretrain: str,
    name_suffix: str | None,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str], str]:
    pairs_l = find_pairs(rows, pretrain, "lora")
    pairs_f = find_pairs(rows, pretrain, "full")

    if not pairs_l:
        raise SystemExit(f"No complete (AdamW, Muon) LoRA rows for pretrain={pretrain!r}.")
    if not pairs_f:
        raise SystemExit(f"No complete (AdamW, Muon) full FT rows for pretrain={pretrain!r}.")

    note = ""

    if name_suffix is not None:
        suf_f = name_suffix.strip()
        if suf_f not in pairs_f:
            have = ", ".join(
                repr(s) if s else "(blank)"
                for s in sorted(pairs_f.keys(), key=lambda s: (s == "", s))
            )
            raise SystemExit(
                f"No full FT pair for name_suffix={suf_f!r}. Available: {have}."
            )
        r_aw_f, r_mu_f = pairs_f[suf_f]
        if suf_f in pairs_l:
            r_aw_l, r_mu_l = pairs_l[suf_f]
        elif "" in pairs_l:
            r_aw_l, r_mu_l = pairs_l[""]
            if suf_f:
                note = " (LoRA: blank suffix; full FT: chosen suffix)"
        else:
            raise SystemExit(
                f"No LoRA (AdamW, Muon) pair for suffix={suf_f!r} or blank."
            )
    else:
        if len(pairs_f) > 1:
            have = ", ".join(
                repr(s) if s else "(blank)"
                for s in sorted(pairs_f.keys(), key=lambda s: (s == "", s))
            )
            raise SystemExit(
                "Multiple full FT pairs; set --name-suffix. "
                f"Available: {have}."
            )
        suf_f = next(iter(pairs_f.keys()))
        r_aw_f, r_mu_f = pairs_f[suf_f]
        if len(pairs_l) > 1:
            have = ", ".join(
                repr(s) if s else "(blank)"
                for s in sorted(pairs_l.keys(), key=lambda s: (s == "", s))
            )
            raise SystemExit(
                "Multiple LoRA pairs; set --name-suffix. "
                f"Available LoRA suffixes: {have}."
            )
        suf_l = next(iter(pairs_l.keys()))
        r_aw_l, r_mu_l = pairs_l[suf_l]
        if suf_l != suf_f and (suf_l or suf_f):
            note = f" (LoRA suffix={suf_l!r}, full FT suffix={suf_f!r})"

    return r_aw_l, r_mu_l, r_aw_f, r_mu_f, note


def _layer_filter_note(r: dict[str, str]) -> str:
    lf = (r.get("layer_filter") or "all").strip()
    return f"layers: {lf}" if lf and lf != "all" else "all layers"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("/home/swetharajkumar/weight_change_analysis_output/weight_change_norms.csv"),
    )
    p.add_argument("--pretrain", choices=("adam", "muon"), required=True)
    p.add_argument("--finetune-type", choices=("lora", "full"), default=None)
    p.add_argument("--lora-and-full", action="store_true")
    p.add_argument("--norm", type=str, default=None, metavar="NAME")
    p.add_argument("--name-suffix", default=None, metavar="SUFFIX")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--log-y", action="store_true")
    p.add_argument("--list-pairs", action="store_true")
    p.add_argument("--list-lora-full", action="store_true")
    p.add_argument(
        "--no-caption",
        action="store_true",
        help="Suppress the methodology caption below the plot",
    )
    args = p.parse_args(argv)

    if args.lora_and_full and args.finetune_type is not None:
        print("Do not pass --finetune-type with --lora-and-full.", file=sys.stderr)
        return 1
    if not args.lora_and_full and args.finetune_type is None and not args.list_lora_full:
        print(
            "Pass --finetune-type {lora,full}, or use --lora-and-full, or --list-lora-full.",
            file=sys.stderr,
        )
        return 1
    if args.list_pairs and args.list_lora_full:
        print("Use only one of --list-pairs or --list-lora-full.", file=sys.stderr)
        return 1

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    rows = load_rows(csv_path)
    if not rows:
        print("CSV has no data rows.", file=sys.stderr)
        return 1

    if args.list_pairs:
        if args.finetune_type is None:
            print("--list-pairs requires --finetune-type.", file=sys.stderr)
            return 1
        try:
            pairs = find_pairs(rows, args.pretrain, args.finetune_type)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        if not pairs:
            print(
                f"No (AdamW, Muon) pair for pretrain={args.pretrain!r}, "
                f"finetune_type={args.finetune_type!r}.",
                file=sys.stderr,
            )
            return 1
        for suf in sorted(pairs.keys(), key=lambda s: (s == "", s)):
            label = "(blank)" if suf == "" else repr(suf)
            ra, rm = pairs[suf]
            print(f"name_suffix {label}\t{ra.get('run_folder')}\t{rm.get('run_folder')}")
        return 0

    if args.list_lora_full:
        try:
            pairs_l = find_pairs(rows, args.pretrain, "lora")
            pairs_f = find_pairs(rows, args.pretrain, "full")
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        print("# LoRA (AdamW, Muon) pairs:")
        for suf in sorted(pairs_l.keys(), key=lambda s: (s == "", s)):
            label = "(blank)" if suf == "" else repr(suf)
            ra, rm = pairs_l[suf]
            print(f"  name_suffix {label}\t{ra.get('run_folder')}\t{rm.get('run_folder')}")
        print("# Full FT (AdamW, Muon) pairs:")
        for suf in sorted(pairs_f.keys(), key=lambda s: (s == "", s)):
            label = "(blank)" if suf == "" else repr(suf)
            ra, rm = pairs_f[suf]
            print(f"  name_suffix {label}\t{ra.get('run_folder')}\t{rm.get('run_folder')}")
        return 0

    if args.norm is None:
        print("--norm is required unless using a --list-* option.", file=sys.stderr)
        return 1

    norm_key = canonical_norm(args.norm)
    col, norm_label, _ = NORM_SPECS[norm_key]

    want_suffix: str | None = args.name_suffix.strip() if args.name_suffix is not None else None

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb
    except ImportError:
        print("matplotlib is required.", file=sys.stderr)
        return 1

    # ========== Combined LoRA + full ==========
    if args.lora_and_full:
        try:
            r_aw_l, r_mu_l, r_aw_f, r_mu_f, pair_note = resolve_lora_full_rows(
                rows, args.pretrain, args.name_suffix
            )
        except SystemExit as e:
            print(e, file=sys.stderr)
            return 1

        try:
            y_lora = [float(r_aw_l[col]), float(r_mu_l[col])]
            y_full = [float(r_aw_f[col]), float(r_mu_f[col])]
        except (KeyError, ValueError) as e:
            print(f"Invalid norm column {col!r}: {e}", file=sys.stderr)
            return 1

        c0 = to_rgb("C0")  # blue (Muon)
        c1 = to_rgb("C1")  # orange (AdamW)
        colors_lora = [(*c1, 0.55), (*c0, 0.55)]
        colors_full = [(*c1, 1.0), (*c0, 1.0)]

        fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="tight")
        width = 0.36
        x0, x1 = 0, 1
        x_lora = [x0 - width / 2, x1 - width / 2]
        x_full = [x0 + width / 2, x1 + width / 2]

        bars_l = ax.bar(x_lora, y_lora, width, color=colors_lora,
                        edgecolor="black", linewidth=0.6, zorder=2)
        bars_f = ax.bar(x_full, y_full, width, color=colors_full,
                        edgecolor="black", linewidth=0.6, zorder=2)

        ax.set_xticks([x0, x1])
        ax.set_xticklabels(["AdamW\n(finetune)", "Muon\n(finetune)"], fontsize=10)

        # Y-axis label — always present
        ax.set_ylabel(NORM_YLABEL[norm_key], fontsize=9, labelpad=6)

        if args.log_y:
            ax.set_yscale("log")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.55, color="gray", zorder=1)

        from matplotlib.transforms import blended_transform_factory
        blended = blended_transform_factory(ax.transData, ax.transAxes)

        bar_row_pairs = [
            (bars_l[0], r_aw_l, "LoRA"),
            (bars_l[1], r_mu_l, "LoRA"),
            (bars_f[0], r_aw_f, "Full FT"),
            (bars_f[1], r_mu_f, "Full FT"),
        ]
        for bar, r, kind in bar_row_pairs:
            cx = bar.get_x() + bar.get_width() / 2
            ax.annotate(
                kind,
                xy=(cx, 0.0), xycoords=blended,
                xytext=(0, -6), textcoords="offset points",
                ha="center", va="top", fontsize=8, clip_on=False,
            )
            step = r.get("checkpoint_step", "")
            ax.annotate(
                f"step {step}",
                xy=(cx, 0.0), xycoords=blended,
                xytext=(0, -20), textcoords="offset points",
                ha="center", va="top", fontsize=6, color="gray", clip_on=False,
            )

        lf_note = _layer_filter_note(r_aw_f)
        title = (
            f"{norm_label.replace(chr(10), ' ')} — {args.pretrain} pretrain, "
            f"LoRA vs full FT{pair_note}  [{lf_note}]"
        )
        ax.set_title(title, fontsize=10, pad=6)

        for bars, ys, rows_pair in (
            (bars_l, y_lora, (r_aw_l, r_mu_l)),
            (bars_f, y_full, (r_aw_f, r_mu_f)),
        ):
            for bar, v, r in zip(bars, ys, rows_pair):
                ax.annotate(
                    f"{v:.4g}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                )

        fig.subplots_adjust(bottom=0.22)

        if not args.no_caption:
            fig.text(
                0.5, -0.01,
                _METHODOLOGY_CAPTION,
                ha="center", va="top",
                fontsize=7, color="#555555", style="italic",
            )

        out = args.output
        if out is None:
            suf_tag = suffix_key(r_aw_f) or "default"
            safe_suf = "".join(c if c.isalnum() or c in "-_" else "_" for c in suf_tag)
            safe_norm = norm_key.replace("_", "")
            out = (
                csv_path.parent
                / f"compare_{args.pretrain}_lora_full_{safe_norm}_{safe_suf}.png"
            )
        else:
            out = out.resolve()

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(str(out))
        return 0

    # ========== Single finetune type ==========
    try:
        pairs = find_pairs(rows, args.pretrain, args.finetune_type)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    if not pairs:
        print(
            f"No rows with both AdamW and Muon for pretrain={args.pretrain!r}, "
            f"finetune_type={args.finetune_type!r}.",
            file=sys.stderr,
        )
        return 1

    if want_suffix is not None:
        if want_suffix not in pairs:
            have = ", ".join(
                repr(s) if s else "(blank)"
                for s in sorted(pairs.keys(), key=lambda s: (s == "", s))
            )
            print(
                f"No pair for name_suffix={want_suffix!r}. Available: {have}.",
                file=sys.stderr,
            )
            return 1
        r_adamw, r_muon = pairs[want_suffix]
    else:
        if len(pairs) > 1:
            have = ", ".join(
                repr(s) if s else "(blank)"
                for s in sorted(pairs.keys(), key=lambda s: (s == "", s))
            )
            print(
                "Multiple (AdamW, Muon) pairs match; choose one with --name-suffix. "
                f"Available: {have}.",
                file=sys.stderr,
            )
            return 1
        only_suffix = next(iter(pairs.keys()))
        r_adamw, r_muon = pairs[only_suffix]

    try:
        y_adamw = float(r_adamw[col])
        y_muon = float(r_muon[col])
    except (KeyError, ValueError) as e:
        print(f"Invalid norm column {col!r}: {e}", file=sys.stderr)
        return 1

    ft_label = "LoRA" if normalize_finetune_type(args.finetune_type) == "lora" else "Full FT"
    bar_labels = ["AdamW\n(finetune)", "Muon\n(finetune)"]
    values = [y_adamw, y_muon]
    colors = ["C1", "C0"]  # orange for AdamW, blue for Muon

    fig, ax = plt.subplots(figsize=(6.5, 5.0), layout="constrained")
    x = (0, 1)
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6, width=0.55, zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(bar_labels, fontsize=10)

    # Y-axis label — always present
    ax.set_ylabel(NORM_YLABEL[norm_key], fontsize=9, labelpad=6)

    if args.log_y:
        ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.55, color="gray", zorder=1)

    suf_disp = (r_adamw.get("name_suffix") or "").strip()
    suf_part = f", suffix={suf_disp!r}" if suf_disp else ""
    lf_note = _layer_filter_note(r_adamw)
    title = (
        f"{norm_label.replace(chr(10), ' ')} — {args.pretrain} pretrain, "
        f"{ft_label}{suf_part}  [{lf_note}]"
    )
    ax.set_title(title, fontsize=10, pad=6)

    for bar, v, r in zip(bars, values, (r_adamw, r_muon)):
        ax.annotate(
            f"{v:.4g}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )
        step = r.get("checkpoint_step", "")
        ax.annotate(
            f"step {step}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_y()),
            xytext=(0, -14), textcoords="offset points",
            ha="center", va="top", fontsize=7, color="gray",
        )

    if not args.no_caption:
        fig.text(
            0.5, -0.01,
            _METHODOLOGY_CAPTION,
            ha="center", va="top",
            fontsize=7, color="#555555", style="italic",
        )

    out = args.output
    if out is None:
        suf_tag = suffix_key(r_adamw) or "default"
        safe_suf = "".join(c if c.isalnum() or c in "-_" else "_" for c in suf_tag)
        safe_norm = norm_key.replace("_", "")
        out = (
            csv_path.parent
            / f"compare_{args.pretrain}_{args.finetune_type}_{safe_norm}_{safe_suf}.png"
        )
    else:
        out = out.resolve()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())