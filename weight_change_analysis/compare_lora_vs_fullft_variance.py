#!/usr/bin/env python3
"""
Pivot the per-(run, tensor-type) variance CSV into Markdown comparison tables.

Reads the CSV produced by ``compute_per_tensor_norm_variance.py`` and, for each
pretraining optimizer (e.g. adam / muon), emits one Markdown table per norm
(``l_inf``, ``l2``, ``rms_rms_induced``) where:

  * rows are tensor categories (Embedding / Q proj / KV proj / O proj / MLP /
    router / LayerNorm / Other), and
  * columns are the four finetune variants ``(finetune_type, finetune_optimizer)``:
    AdamW LoRA, Muon LoRA, AdamW Full FT, Muon Full FT.

Each cell is the variance of that norm across all tensors in that category for
that run. Missing variants (e.g. when a run is absent from the CSV) are shown
as ``-``; NaN variances (e.g. ``rms_rms_induced`` for 1D LayerNorm weights) are
shown as ``NaN``.

Example:

  python3 weight_change_analysis/compare_lora_vs_fullft_variance.py \\
      --variance-csv weight_change_analysis_output/per_tensor_norm_variance.csv \\
      -o weight_change_analysis_output/per_tensor_norm_variance_tables.md
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from pathlib import Path


CATEGORY_ORDER: list[tuple[str, str]] = [
    ("embed", "Embedding / LM head"),
    ("q", "Attention Q proj"),
    ("kv", "Attention KV proj"),
    ("o", "Attention O proj"),
    ("mlp", "MLP / MoE experts"),
    ("router", "MoE router gate"),
    ("norm", "LayerNorm / RMSNorm"),
    ("other", "Other"),
]
CATEGORY_LABEL: dict[str, str] = dict(CATEGORY_ORDER)

NORM_INFO: list[tuple[str, str, str, str]] = [
    (
        "l_inf",
        "var_l_inf",
        r"$\mathrm{Var}\bigl(\|\Delta W\|_\infty\bigr)$",
        r"$\|\Delta W\|_\infty$",
    ),
    (
        "l2",
        "var_l2",
        r"$\mathrm{Var}\bigl(\|\Delta W\|_F\bigr)$",
        r"$\|\Delta W\|_F$",
    ),
    (
        "rms_rms_induced",
        "var_rms_rms_induced",
        r"$\mathrm{Var}\bigl(\|\Delta W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}\bigr)$",
        r"$\|\Delta W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}$",
    ),
]

PRETRAIN_LABEL = {"adam": "Adam", "adamw": "AdamW", "muon": "Muon"}

# Column key -> (header label, finetune_type, finetune_optimizer).
COL_VARIANTS: list[tuple[str, str, str, str]] = [
    ("adamw_lora", "AdamW LoRA", "lora", "adamw"),
    ("muon_lora", "Muon LoRA", "lora", "muon"),
    ("adamw_full", "AdamW Full FT", "full", "adamw"),
    ("muon_full", "Muon Full FT", "full", "muon"),
]

# Compare AdamW vs Muon *within* each finetune type. Indices refer to
# positions inside ``COL_VARIANTS``.
COL_GROUPS: list[tuple[int, int]] = [(0, 1), (2, 3)]


def format_value(v: float, bold: bool = False) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    if v == 0:
        s = "0"
    else:
        av = abs(v)
        if av >= 1e3 or av < 1e-3:
            s = f"{v:.3e}"
        else:
            s = f"{v:.4g}"
    return f"**{s}**" if bold else s


def format_value_latex(v: float | None, bold: bool = False) -> str:
    """LaTeX-friendly cell formatter with proper scientific notation."""
    if v is None:
        return "--"
    if isinstance(v, float) and math.isnan(v):
        return r"\textrm{NaN}"
    if v == 0:
        inner = "0"
    else:
        av = abs(v)
        if av >= 1e3 or av < 1e-3:
            mant_str = f"{v:.3e}"
            mant, exp_str = mant_str.split("e")
            exp_int = int(exp_str)
            inner = f"{mant} \\times 10^{{{exp_int}}}"
        else:
            inner = f"{v:.4g}"
    if bold:
        return f"$\\mathbf{{{inner}}}$"
    return f"${inner}$"


def _group_min_indices(
    values: list[float | None],
    groups: list[tuple[int, ...]],
) -> set[int]:
    """For each ``group`` of column indices, mark the group's minimum cell(s).

    Ignores ``None`` and ``NaN``. Within a group, nothing is highlighted when
    there is only one numeric value or every numeric value is equal (so we
    don't bold a sub-group where there is no real "lowest").
    """
    keep: set[int] = set()
    for group in groups:
        numeric = [
            (i, values[i])
            for i in group
            if isinstance(values[i], (int, float))
            and not (isinstance(values[i], float) and math.isnan(values[i]))
        ]
        if len(numeric) < 2:
            continue
        mn = min(v for _, v in numeric)
        if mn == max(v for _, v in numeric):
            continue
        for i, v in numeric:
            if v <= mn:
                keep.add(i)
    return keep


def read_variance_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: empty CSV")
    return rows


def index_rows(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    """Map (pretrain_optimizer, finetune_type, finetune_optimizer, tensor_category) -> row."""
    idx: dict[tuple[str, str, str, str], dict] = {}
    for r in rows:
        key = (
            (r.get("pretrain_optimizer") or "").lower(),
            (r.get("finetune_type") or "").lower(),
            (r.get("finetune_optimizer") or "").lower(),
            (r.get("tensor_category") or "").lower(),
        )
        idx[key] = r
    return idx


def _expand_norms(norms: list[str]) -> list[tuple[str, str, str, str]]:
    return [
        (key, col, latex_var, latex_bare)
        for key, col, latex_var, latex_bare in NORM_INFO
        if key in norms
    ]


def _list_pretrains(rows: list[dict]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for r in rows:
        po = (r.get("pretrain_optimizer") or "").lower()
        if po and po not in seen:
            seen.add(po)
            out.append(po)
    return out


def build_latex(
    rows: list[dict],
    excluded: set[str],
    norms: list[str],
) -> str:
    """Emit booktabs-style LaTeX tables (one per pretrain optimizer x norm)."""
    idx = index_rows(rows)
    pretrains = _list_pretrains(rows)
    cats = [(k, label) for k, label in CATEGORY_ORDER if k not in excluded]
    selected_norms = _expand_norms(norms)

    buf = io.StringIO()
    buf.write(
        "% Per-tensor weight-change norm variance: LoRA vs Full FT, AdamW vs Muon.\n"
        "% Requires: \\usepackage{booktabs}\n"
        "% Each cell is the sample variance (ddof=1) of the per-tensor norm\n"
        "% across all tensors in that category for one finetuned run.\n"
        "% '--' = no row present for that variant; 'NaN' = norm undefined.\n\n"
    )

    ncols = len(COL_VARIANTS)

    def _col_align() -> str:
        boundaries = {g[0] for g in COL_GROUPS if g[0] > 0}
        parts = ["l"]
        for i in range(ncols):
            if i in boundaries:
                parts.append("|")
            parts.append("r")
        return "".join(parts)

    col_align = _col_align()

    for po in pretrains:
        po_label = PRETRAIN_LABEL.get(po, po.capitalize())
        for norm_key, col, _latex_var, latex_bare in selected_norms:
            caption = (
                f"{po_label} pretraining: variance of {latex_bare} "
                "across tensors in each category, by finetune variant. "
                "Bold cells mark the lower AdamW-vs-Muon variance within "
                "each finetune type (LoRA on the left of the divider, "
                "Full FT on the right)."
            )
            label_key = norm_key.replace("_", "-")
            buf.write("\\begin{table}[h]\n")
            buf.write("\\centering\n")
            buf.write(f"\\begin{{tabular}}{{{col_align}}}\n")
            buf.write("\\toprule\n")
            header_cells = ["Tensor type"] + [label for _, label, _, _ in COL_VARIANTS]
            buf.write(" & ".join(header_cells) + " \\\\\n")
            buf.write("\\midrule\n")
            for cat_key, cat_label in cats:
                row_values: list[float | None] = []
                for _, _, ft, opt in COL_VARIANTS:
                    r = idx.get((po, ft, opt, cat_key))
                    if r is None:
                        row_values.append(None)
                        continue
                    raw = r.get(col, "")
                    try:
                        row_values.append(float(raw))
                    except (TypeError, ValueError):
                        row_values.append(float("nan"))
                if all(v is None for v in row_values):
                    continue
                bold_idx = _group_min_indices(row_values, COL_GROUPS)
                cells: list[str] = [cat_label]
                for i, v in enumerate(row_values):
                    cells.append(format_value_latex(v, bold=i in bold_idx))
                buf.write(" & ".join(cells) + " \\\\\n")
            buf.write("\\bottomrule\n")
            buf.write("\\end{tabular}\n")
            buf.write(f"\\caption{{{caption}}}\n")
            buf.write(f"\\label{{tab:var-{po}-{label_key}}}\n")
            buf.write("\\end{table}\n\n")
    return buf.getvalue()


def build_markdown(
    rows: list[dict],
    excluded: set[str],
    norms: list[str],
) -> str:
    idx = index_rows(rows)

    pretrains: list[str] = []
    seen: set[str] = set()
    for r in rows:
        po = (r.get("pretrain_optimizer") or "").lower()
        if po and po not in seen:
            seen.add(po)
            pretrains.append(po)

    cats: list[tuple[str, str]] = [
        (k, label) for k, label in CATEGORY_ORDER if k not in excluded
    ]

    selected_norms = _expand_norms(norms)

    buf = io.StringIO()
    buf.write(
        "# Per-tensor weight-change norm variance: LoRA vs Full FT, AdamW vs Muon\n\n"
    )
    buf.write(
        "Each cell is the **sample variance** (ddof=1) of the per-tensor norm across "
        "all tensors in that category for one finetuned run. Bold cells mark the "
        "lower AdamW-vs-Muon variance within each finetune type (LoRA columns "
        "compared to each other, Full FT columns compared to each other).  \n"
        "`-` = no row present for that (pretraining optimizer, finetune variant). "
        "`NaN` = norm undefined for every tensor in the group "
        "(e.g. `rms->rms` for 1-D LayerNorm weights).\n\n"
    )

    for po in pretrains:
        po_label = PRETRAIN_LABEL.get(po, po.capitalize())
        for norm_key, col, latex_var, _latex_bare in selected_norms:
            buf.write(f"## {po_label} pretraining — {latex_var}\n\n")
            buf.write(
                "| Tensor type | "
                + " | ".join(label for _, label, _, _ in COL_VARIANTS)
                + " |\n"
            )
            buf.write(
                "|---|" + "|".join(["---:"] * len(COL_VARIANTS)) + "|\n"
            )
            for cat_key, cat_label in cats:
                row_values: list[float | None] = []
                for _, _, ft, opt in COL_VARIANTS:
                    r = idx.get((po, ft, opt, cat_key))
                    if r is None:
                        row_values.append(None)
                        continue
                    raw = r.get(col, "")
                    try:
                        row_values.append(float(raw))
                    except (TypeError, ValueError):
                        row_values.append(float("nan"))
                if all(v is None for v in row_values):
                    continue
                bold_idx = _group_min_indices(row_values, COL_GROUPS)
                cells: list[str] = [cat_label]
                for i, v in enumerate(row_values):
                    cells.append(format_value(v, bold=i in bold_idx))
                buf.write("| " + " | ".join(cells) + " |\n")
            buf.write("\n")
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Pivot the per-(run, tensor-type) variance CSV into Markdown "
            "comparison tables (LoRA vs Full FT, AdamW vs Muon) per "
            "pretraining optimizer."
        )
    )
    p.add_argument(
        "--variance-csv",
        type=Path,
        default=Path("weight_change_analysis_output/per_tensor_norm_variance.csv"),
        help="Path to the variance CSV from compute_per_tensor_norm_variance.py.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write Markdown here. If omitted, prints to stdout.",
    )
    p.add_argument(
        "--exclude-types",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated tensor categories to drop from the table rows. "
            f"Valid keys: {', '.join(k for k, _ in CATEGORY_ORDER)}."
        ),
    )
    p.add_argument(
        "--norms",
        type=str,
        default="l_inf,rms_rms_induced",
        metavar="LIST",
        help=(
            "Comma-separated list of norm columns to emit one table for. "
            "Valid keys: l_inf, l2, rms_rms_induced. "
            "Default: 'l_inf,rms_rms_induced'."
        ),
    )
    p.add_argument(
        "--format",
        choices=("markdown", "latex"),
        default="markdown",
        help="Output format. 'latex' produces booktabs-style tables.",
    )
    args = p.parse_args(argv)

    excluded: set[str] = set()
    if args.exclude_types:
        for raw in args.exclude_types.split(","):
            tok = raw.strip().lower()
            if not tok:
                continue
            if tok not in CATEGORY_LABEL:
                print(
                    f"Unknown tensor category {raw!r}. Valid keys: "
                    f"{', '.join(k for k, _ in CATEGORY_ORDER)}.",
                    file=sys.stderr,
                )
                return 1
            excluded.add(tok)

    valid_norm_keys = {k for k, *_ in NORM_INFO}
    norms = [t.strip().lower() for t in args.norms.split(",") if t.strip()]
    bad = [n for n in norms if n not in valid_norm_keys]
    if bad:
        print(
            f"Unknown norm key(s) {bad}. Valid: {sorted(valid_norm_keys)}.",
            file=sys.stderr,
        )
        return 1
    if not norms:
        print("No norms requested.", file=sys.stderr)
        return 1

    rows = read_variance_csv(args.variance_csv)
    if args.format == "latex":
        out = build_latex(rows, excluded, norms)
    else:
        out = build_markdown(rows, excluded, norms)

    if args.output is None:
        sys.stdout.write(out)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
