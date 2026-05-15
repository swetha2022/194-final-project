#!/usr/bin/env python3
"""
Scatter plot of per-tensor weight-change norms across one or more runs.

Each point is one tensor from one per-tensor CSV produced by
``run_analysis.py --per-tensor-norms``. Pick any two norm columns for the
x- and y-axes; each CSV gets its own **color** so points from different
finetuned models are easy to tell apart.

By default each point's **marker shape** indicates what kind of weight the
tensor is (embedding / Q / KV / O / MLP / router / layernorm / other) so the
plot encodes two attributes at once. Use ``--mark-by csv`` to revert to the
old behavior where the marker shape selects the CSV instead.

Examples:

  python3 weight_change_analysis/plot_per_tensor_scatter.py \\
      --csvs weight_change_analysis_output/per_tensor_norms/*.csv \\
      --x-norm l2 --y-norm l_inf --log-x --log-y

  python3 weight_change_analysis/plot_per_tensor_scatter.py \\
      --csvs \\
        weight_change_analysis_output/per_tensor_norms/adam_ckpt_driving_adamw.csv \\
        weight_change_analysis_output/per_tensor_norms/adam_ckpt_driving_muon.csv \\
      --labels "AdamW LoRA" "Muon LoRA" \\
      --x-norm l2 --y-norm rms
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Iterable

# Canonical norm key -> per-tensor CSV column name (as written by
# run_analysis.py --per-tensor-norms).
PER_TENSOR_COLUMN: dict[str, str] = {
    "l_inf": "l_inf",
    "l2": "l2",
    "rms": "rms_rms_induced",
}

# Pretty axis label per canonical key (mathtext).
NORM_AXIS_LABEL: dict[str, str] = {
    "l_inf": r"$\|\Delta W\|_{\infty}$  (per tensor)",
    "l2": r"$\|\Delta W\|_{F}$  (per tensor)",
    "rms": r"$\|\Delta W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}$  (per 2D tensor)",
}

# User-facing aliases -> canonical key.
NORM_ALIASES: dict[str, str] = {
    "l_inf": "l_inf",
    "linf": "l_inf",
    "l-infinity": "l_inf",
    "inf": "l_inf",
    "infty": "l_inf",
    "l2": "l2",
    "l_2": "l2",
    "frob": "l2",
    "frobenius": "l2",
    "rms": "rms",
    "rms_rms": "rms",
    "rms_rms_induced": "rms",
}


# Ordered tensor categories: key -> (display label, matplotlib marker).
# Order controls legend order and disambiguation priority in categorize_tensor().
TENSOR_CATEGORIES: list[tuple[str, str, str]] = [
    ("embed", "Embedding / LM head", "*"),
    ("q", "Attention Q proj", "^"),
    ("kv", "Attention KV proj", "v"),
    ("o", "Attention O proj", "D"),
    ("mlp", "MLP / MoE experts", "o"),
    ("router", "MoE router gate", "P"),
    ("norm", "LayerNorm / RMSNorm", "s"),
    ("other", "Other", "X"),
]
CATEGORY_MARKER: dict[str, str] = {k: m for k, _, m in TENSOR_CATEGORIES}
CATEGORY_LABEL: dict[str, str] = {k: l for k, l, _ in TENSOR_CATEGORIES}
CATEGORY_ORDER: list[str] = [k for k, _, _ in TENSOR_CATEGORIES]

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("embed", re.compile(r"(^|\.)embed_tokens($|\.)|(^|\.)lm_head($|\.)")),
    ("q", re.compile(r"self_attn\.q_proj")),
    ("kv", re.compile(r"self_attn\.(kv_a_proj_with_mqa|kv_b_proj|k_proj|v_proj)")),
    ("o", re.compile(r"self_attn\.o_proj")),
    # Router gate weight / bias (e.g. mlp.gate.weight, mlp.gate.e_score_correction_bias).
    # Matched BEFORE the broader mlp.* rule.
    ("router", re.compile(r"mlp\.gate(\.|$)")),
    # Any MLP / MoE projection: mlp.{up,gate,down}_proj, mlp.experts.N.*_proj,
    # mlp.shared_experts.*_proj.
    ("mlp", re.compile(r"mlp\.(.*_proj)")),
    # LayerNorm / RMSNorm — 1D weights; covers input_layernorm, post_attention_layernorm,
    # self_attn.kv_a_layernorm, model.norm, etc.
    ("norm", re.compile(r"(layernorm|(^|\.)norm)(\.|$)", re.IGNORECASE)),
]


def categorize_tensor(name: str) -> str:
    """Return a short category key for a safetensors tensor name."""
    for key, pat in _CATEGORY_PATTERNS:
        if pat.search(name):
            return key
    return "other"


def canonical_norm(token: str) -> str:
    t = token.strip().lower().replace(" ", "")
    if t not in NORM_ALIASES:
        allowed = ", ".join(sorted(set(NORM_ALIASES.keys())))
        raise SystemExit(f"Unknown norm {token!r}. Use one of: {allowed}")
    return NORM_ALIASES[t]


def load_norm_pairs(
    csv_path: Path, x_col: str, y_col: str
) -> tuple[list[float], list[float], list[str], dict[str, str]]:
    """Read (x, y, category) triples from a per-tensor CSV; skip rows where either is NaN."""
    xs: list[float] = []
    ys: list[float] = []
    cats: list[str] = []
    meta: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or x_col not in reader.fieldnames or y_col not in reader.fieldnames:
            raise SystemExit(
                f"{csv_path}: missing required column(s). "
                f"Need {x_col!r} and {y_col!r}; found {reader.fieldnames!r}."
            )
        for row in reader:
            if not meta:
                meta = {
                    "run_folder": row.get("run_folder", "") or csv_path.stem,
                    "pretrain_optimizer": row.get("pretrain_optimizer", ""),
                    "finetune_type": row.get("finetune_type", ""),
                    "finetune_optimizer": row.get("finetune_optimizer", ""),
                    "name_suffix": row.get("name_suffix", ""),
                }
            try:
                xv = float(row[x_col])
                yv = float(row[y_col])
            except ValueError:
                continue
            if math.isnan(xv) or math.isnan(yv):
                continue
            xs.append(xv)
            ys.append(yv)
            cats.append(categorize_tensor(row.get("tensor_name", "")))
    return xs, ys, cats, meta


def default_label(meta: dict[str, str], csv_path: Path) -> str:
    return meta.get("run_folder") or csv_path.stem


def color_cycle(n: int) -> list[tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt

    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i % 10) for i in range(n)]
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i % 20) for i in range(n)]
    cmap = plt.get_cmap("turbo")
    return [cmap(i / max(1, n - 1)) for i in range(n)]


_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]


def safe_positive(values: Iterable[float], floor: float | None) -> list[float]:
    """Replace non-positive values with `floor` (for log scale); pass through if floor is None."""
    if floor is None:
        return list(values)
    out: list[float] = []
    for v in values:
        out.append(v if v > 0 else floor)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Scatter plot of per-tensor weight-change norms across one or more "
            "per-tensor CSVs produced by run_analysis.py --per-tensor-norms."
        )
    )
    p.add_argument(
        "--csvs",
        type=Path,
        nargs="+",
        required=True,
        metavar="CSV",
        help="One or more per-tensor CSV files (one per model).",
    )
    p.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Optional display labels (one per CSV, same order). "
        "Default: each CSV's run_folder column (or filename stem).",
    )
    p.add_argument(
        "--x-norm",
        type=str,
        default="l2",
        help="Norm for x-axis. Keys: l_inf, l2, rms (aliases also accepted).",
    )
    p.add_argument(
        "--y-norm",
        type=str,
        default="l_inf",
        help="Norm for y-axis. Keys: l_inf, l2, rms (aliases also accepted).",
    )
    p.add_argument(
        "--mark-by",
        choices=("type", "csv"),
        default="type",
        help="What the marker SHAPE encodes. 'type' (default): tensor category "
        "(embedding/Q/KV/O/MLP/router/norm/other); 'csv': which CSV the point came from. "
        "Color always encodes the CSV/model. Ignored when --facet type is set.",
    )
    p.add_argument(
        "--facet",
        choices=("none", "type"),
        default="none",
        help="If 'type', draw one subplot per tensor category (embedding/Q/KV/O/MLP/"
        "router/norm/other) in a shared figure with shared axes. Default 'none' keeps "
        "all categories on one panel.",
    )
    p.add_argument(
        "--facet-cols",
        type=int,
        default=0,
        metavar="N",
        help="Number of columns in the subplot grid when --facet type. "
        "Default 0 = auto (roughly square: e.g. 4 categories -> 2x2, "
        "6 -> 3x2, 7-9 -> 3x3).",
    )
    p.add_argument(
        "--facet-share-axes",
        choices=("none", "x", "y", "all"),
        default="none",
        help="Axis sharing when --facet type. Default 'none' lets every subplot "
        "autoscale around its own tensor type's data (so small categories like "
        "Q proj are no longer cramped by the MLP cloud). Use 'all' for the old "
        "behavior of a single shared axis range across the grid.",
    )
    p.add_argument(
        "--exclude-types",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated tensor categories to drop before plotting. "
        f"Valid keys: {', '.join(CATEGORY_ORDER)}. "
        "Example: --exclude-types embed,router",
    )
    p.add_argument(
        "--log-x",
        action="store_true",
        help="Use log scale on the x-axis.",
    )
    p.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale on the y-axis.",
    )
    p.add_argument(
        "--log-floor",
        type=float,
        default=1e-12,
        help="When using log scale, replace zero / non-positive values with this floor "
        "so they remain visible. Set to a non-positive number to drop those points instead.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Point alpha (transparency).",
    )
    p.add_argument(
        "--marker-size",
        type=float,
        default=22.0,
        help="Point area (matplotlib s= argument).",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override plot title.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (.png or .pdf). Default: "
        "weight_change_analysis_output/per_tensor_scatter_<ynorm>_vs_<xnorm>.png",
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(8.0, 6.0),
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    args = p.parse_args(argv)

    x_key = canonical_norm(args.x_norm)
    y_key = canonical_norm(args.y_norm)
    x_col = PER_TENSOR_COLUMN[x_key]
    y_col = PER_TENSOR_COLUMN[y_key]

    csv_paths = [c.resolve() for c in args.csvs]
    missing = [p for p in csv_paths if not p.is_file()]
    if missing:
        for m in missing:
            print(f"CSV not found: {m}", file=sys.stderr)
        return 1

    if args.labels is not None and len(args.labels) != len(csv_paths):
        print(
            f"--labels has {len(args.labels)} entries but --csvs has {len(csv_paths)}.",
            file=sys.stderr,
        )
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
        return 1

    colors = color_cycle(len(csv_paths))

    from matplotlib.lines import Line2D

    use_log_floor = args.log_floor if args.log_floor > 0 else None

    excluded_types: set[str] = set()
    if args.exclude_types:
        for raw in args.exclude_types.split(","):
            tok = raw.strip().lower()
            if not tok:
                continue
            if tok not in CATEGORY_LABEL:
                print(
                    f"Unknown tensor category {raw!r}. Valid keys: "
                    f"{', '.join(CATEGORY_ORDER)}.",
                    file=sys.stderr,
                )
                return 1
            excluded_types.add(tok)

    # ---- Load + log-filter every CSV once, store the processed series ----
    per_csv_data: list[tuple[list[float], list[float], list[str]]] = []
    csv_labels: list[str] = []
    csv_point_counts: list[int] = []
    category_point_counts: dict[str, int] = {k: 0 for k in CATEGORY_ORDER}

    for i, path in enumerate(csv_paths):
        xs, ys, cats, meta = load_norm_pairs(path, x_col, y_col)
        label = args.labels[i] if args.labels is not None else default_label(meta, path)
        csv_labels.append(label)

        if not xs:
            print(f"[warn] {path}: no valid (x, y) rows for {x_col!r}, {y_col!r}", file=sys.stderr)
            per_csv_data.append(([], [], []))
            csv_point_counts.append(0)
            continue

        if excluded_types:
            kept = [(xv, yv, c) for xv, yv, c in zip(xs, ys, cats) if c not in excluded_types]
            xs = [t[0] for t in kept]
            ys = [t[1] for t in kept]
            cats = [t[2] for t in kept]

        if args.log_x and use_log_floor is not None:
            xs = [v if v > 0 else use_log_floor for v in xs]
        if args.log_y and use_log_floor is not None:
            ys = [v if v > 0 else use_log_floor for v in ys]
        if (args.log_x or args.log_y) and use_log_floor is None:
            paired = [
                (xv, yv, c)
                for xv, yv, c in zip(xs, ys, cats)
                if (not args.log_x or xv > 0) and (not args.log_y or yv > 0)
            ]
            xs = [p[0] for p in paired]
            ys = [p[1] for p in paired]
            cats = [p[2] for p in paired]

        if not xs:
            print(f"[warn] {path}: all points removed by log filtering", file=sys.stderr)
            per_csv_data.append(([], [], []))
            csv_point_counts.append(0)
            continue

        per_csv_data.append((xs, ys, cats))
        csv_point_counts.append(len(xs))
        for c in cats:
            if c in category_point_counts:
                category_point_counts[c] += 1

    total_points = sum(csv_point_counts)
    if total_points == 0:
        print("No data points to plot.", file=sys.stderr)
        return 1

    default_title = f"Per-tensor weight-change norms: {y_key} vs {x_key}"
    title = args.title or default_title

    if args.facet == "type":
        # ---- One subplot per tensor category present in the data ----
        present_cats = [k for k in CATEGORY_ORDER if category_point_counts[k] > 0]
        n = len(present_cats)
        if int(args.facet_cols) > 0:
            ncols = int(args.facet_cols)
        else:
            ncols = max(1, math.ceil(math.sqrt(n))) if n > 0 else 1
        nrows = (n + ncols - 1) // ncols if n > 0 else 1
        # Scale default figsize sensibly if the user kept the (8, 6) single-panel default.
        if args.figsize == (8.0, 6.0):
            facet_size = (4.0 * ncols + 1.5, 3.4 * nrows + 1.0)
        else:
            facet_size = tuple(args.figsize)
        share_map = {"none": (False, False), "x": (True, False), "y": (False, True), "all": (True, True)}
        sharex, sharey = share_map[args.facet_share_axes]
        fig, axes_grid = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=facet_size,
            sharex=sharex,
            sharey=sharey,
            squeeze=False,
        )
        axes_flat = [ax for row in axes_grid for ax in row]

        for idx, cat in enumerate(present_cats):
            ax = axes_flat[idx]
            for i, (xs, ys, cats) in enumerate(per_csv_data):
                gx = [x for x, c in zip(xs, cats) if c == cat]
                gy = [y for y, c in zip(ys, cats) if c == cat]
                if not gx:
                    continue
                ax.scatter(
                    gx,
                    gy,
                    s=args.marker_size,
                    alpha=args.alpha,
                    color=colors[i],
                    marker=CATEGORY_MARKER[cat],
                    edgecolors="none",
                )
            ax.set_title(
                f"{CATEGORY_LABEL[cat]}  (n={category_point_counts[cat]})",
                fontsize=10,
            )
            if args.log_x:
                ax.set_xscale("log")
            if args.log_y:
                ax.set_yscale("log")
            ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

        # Hide unused axes.
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        # When axes are shared, only the outer subplots need labels (matplotlib
        # auto-hides the inner ones). When axes are independent, every subplot
        # has its own tick numbers, so put the axis labels on every panel.
        for j, ax in enumerate(axes_flat[:n]):
            row, col = divmod(j, ncols)
            if sharex:
                if row == nrows - 1 or j + ncols >= n:
                    ax.set_xlabel(NORM_AXIS_LABEL[x_key])
            else:
                ax.set_xlabel(NORM_AXIS_LABEL[x_key])
            if sharey:
                if col == 0:
                    ax.set_ylabel(NORM_AXIS_LABEL[y_key])
            else:
                ax.set_ylabel(NORM_AXIS_LABEL[y_key])
            # A bit of margin so the data clusters don't sit on the spines.
            ax.margins(x=0.08, y=0.10)

        # Reserve room at the top for the title + horizontal legend bar.
        # Independent axes need a bit more vertical/horizontal spacing for tick
        # labels on every subplot, so widen wspace/hspace when not sharing.
        title_top = 0.985
        legend_top = 0.94
        axes_top = 0.84
        ws = 0.10 if sharey else 0.28
        hs = 0.36 if sharex else 0.45
        fig.subplots_adjust(top=axes_top, bottom=0.10, left=0.07, right=0.985, hspace=hs, wspace=ws)
        fig.suptitle(title, fontsize=12, y=title_top)

        model_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=colors[i],
                markeredgecolor="none",
                markersize=7,
                label=f"{csv_labels[i]}  (n={csv_point_counts[i]})",
            )
            for i in range(len(csv_paths))
        ]
        fig.legend(
            handles=model_handles,
            title="Model",
            loc="upper center",
            bbox_to_anchor=(0.5, legend_top),
            ncols=min(len(model_handles), 4),
            fontsize=9,
            title_fontsize=10,
            framealpha=0.85,
        )

        out = args.output
        if out is None:
            default_dir = Path("weight_change_analysis_output")
            default_dir.mkdir(parents=True, exist_ok=True)
            out = default_dir / f"per_tensor_scatter_{y_key}_vs_{x_key}_by_type.png"
        else:
            out = out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(str(out))
        return 0

    # ---- Single-panel path (existing behavior) -----------------------------
    fig, ax = plt.subplots(figsize=tuple(args.figsize), layout="constrained")

    for i, (xs, ys, cats) in enumerate(per_csv_data):
        if not xs:
            continue
        color = colors[i]
        if args.mark_by == "type":
            grouped: dict[str, tuple[list[float], list[float]]] = {}
            for xv, yv, c in zip(xs, ys, cats):
                gx, gy = grouped.setdefault(c, ([], []))
                gx.append(xv)
                gy.append(yv)
            for cat in CATEGORY_ORDER:
                if cat not in grouped:
                    continue
                gx, gy = grouped[cat]
                ax.scatter(
                    gx,
                    gy,
                    s=args.marker_size,
                    alpha=args.alpha,
                    color=color,
                    marker=CATEGORY_MARKER[cat],
                    edgecolors="none",
                )
        else:
            ax.scatter(
                xs,
                ys,
                s=args.marker_size,
                alpha=args.alpha,
                color=color,
                marker=_MARKERS[i % len(_MARKERS)],
                edgecolors="none",
            )

    ax.set_xlabel(NORM_AXIS_LABEL[x_key])
    ax.set_ylabel(NORM_AXIS_LABEL[y_key])
    if args.log_x:
        ax.set_xscale("log")
    if args.log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)

    ax.set_title(title, fontsize=11)

    # Dual legend: models (by color) + tensor types (by marker shape).
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o" if args.mark_by == "type" else _MARKERS[i % len(_MARKERS)],
            linestyle="",
            color=colors[i],
            markeredgecolor="none",
            markersize=7,
            label=f"{csv_labels[i]}  (n={csv_point_counts[i]})",
        )
        for i in range(len(csv_paths))
    ]
    model_legend = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.85,
    )

    if args.mark_by == "type":
        type_handles = [
            Line2D(
                [0],
                [0],
                marker=CATEGORY_MARKER[k],
                linestyle="",
                color="#444444",
                markeredgecolor="none",
                markersize=7,
                label=f"{CATEGORY_LABEL[k]}  (n={category_point_counts[k]})",
            )
            for k in CATEGORY_ORDER
            if category_point_counts[k] > 0
        ]
        if type_handles:
            ax.add_artist(model_legend)
            ax.legend(
                handles=type_handles,
                title="Tensor type",
                loc="lower right",
                fontsize=8,
                title_fontsize=9,
                framealpha=0.85,
            )

    out = args.output
    if out is None:
        default_dir = Path("weight_change_analysis_output")
        default_dir.mkdir(parents=True, exist_ok=True)
        out = default_dir / f"per_tensor_scatter_{y_key}_vs_{x_key}.png"
    else:
        out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
