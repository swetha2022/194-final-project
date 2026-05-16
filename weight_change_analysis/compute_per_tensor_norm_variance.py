#!/usr/bin/env python3
"""
Aggregate per-tensor weight-change norms into per-(run, tensor-type) statistics.

Reads one or more per-tensor CSVs produced by
``run_analysis.py --per-tensor-norms`` and writes a single CSV with one row per
``(finetuned run, tensor category)``. For each cell we report count, mean,
sample variance (``ddof=1``) and sample standard deviation for each norm
column we know about (``l_inf``, ``l2``, ``rms_rms_induced``).

Tensor categories use the same regex-based classifier as
``plot_per_tensor_scatter.py`` (embedding / Q / KV / O / MLP / router / norm /
other) so the grouping here is consistent with the scatter plots.

The ``rms_rms_induced`` norm is NaN for non-2D tensors (e.g. LayerNorm weights);
NaN values are dropped before computing the stats for that norm only, so the
``count_rms_rms_induced`` column may be smaller than ``count`` for that group.

Example:

  python3 weight_change_analysis/compute_per_tensor_norm_variance.py \\
      --csvs weight_change_analysis_output/per_tensor_norms/*.csv \\
      -o weight_change_analysis_output/per_tensor_norm_variance.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path


TENSOR_CATEGORIES: list[tuple[str, str]] = [
    ("embed", "Embedding / LM head"),
    ("q", "Attention Q proj"),
    ("kv", "Attention KV proj"),
    ("o", "Attention O proj"),
    ("mlp", "MLP / MoE experts"),
    ("router", "MoE router gate"),
    ("norm", "LayerNorm / RMSNorm"),
    ("other", "Other"),
]
CATEGORY_LABEL: dict[str, str] = {k: l for k, l in TENSOR_CATEGORIES}
CATEGORY_ORDER: list[str] = [k for k, _ in TENSOR_CATEGORIES]

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("embed", re.compile(r"(^|\.)embed_tokens($|\.)|(^|\.)lm_head($|\.)")),
    ("q", re.compile(r"self_attn\.q_proj")),
    ("kv", re.compile(r"self_attn\.(kv_a_proj_with_mqa|kv_b_proj|k_proj|v_proj)")),
    ("o", re.compile(r"self_attn\.o_proj")),
    ("router", re.compile(r"mlp\.gate(\.|$)")),
    ("mlp", re.compile(r"mlp\.(.*_proj)")),
    ("norm", re.compile(r"(layernorm|(^|\.)norm)(\.|$)", re.IGNORECASE)),
]


def categorize_tensor(name: str) -> str:
    for key, pat in _CATEGORY_PATTERNS:
        if pat.search(name):
            return key
    return "other"


NORM_COLS: list[str] = ["l_inf", "l2", "rms_rms_induced"]

META_COLS: list[str] = [
    "pretrain_optimizer",
    "run_folder",
    "run_path",
    "pretrain_base_path",
    "finetune_type",
    "finetune_optimizer",
    "name_suffix",
    "checkpoint_step",
]


def _parse_float(v: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _stats(vals: list[float]) -> tuple[int, float, float, float, float, float]:
    """Return (n, mean, variance_ddof1, std_ddof1, vmin, vmax) ignoring NaN."""
    clean = [v for v in vals if not math.isnan(v)]
    n = len(clean)
    if n == 0:
        nan = float("nan")
        return 0, nan, nan, nan, nan, nan
    mean = sum(clean) / n
    if n >= 2:
        var = sum((v - mean) ** 2 for v in clean) / (n - 1)
        std = math.sqrt(var)
    else:
        var = float("nan")
        std = float("nan")
    return n, mean, var, std, min(clean), max(clean)


def build_output_fields() -> list[str]:
    fields = list(META_COLS) + ["tensor_category", "tensor_category_label", "count"]
    for col in NORM_COLS:
        fields += [
            f"count_{col}",
            f"mean_{col}",
            f"var_{col}",
            f"std_{col}",
            f"min_{col}",
            f"max_{col}",
        ]
    return fields


def aggregate_csv(csv_path: Path, excluded: set[str]) -> list[dict]:
    """Aggregate one per-tensor CSV into per-category rows."""
    buckets: dict[str, dict] = {}
    run_meta: dict[str, str] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"{csv_path}: empty CSV")
        missing = [c for c in NORM_COLS if c not in reader.fieldnames]
        if missing:
            raise SystemExit(
                f"{csv_path}: missing required column(s): {missing}. "
                f"Found: {reader.fieldnames!r}."
            )
        for row in reader:
            if not run_meta:
                run_meta = {k: row.get(k, "") for k in META_COLS}
            cat = categorize_tensor(row.get("tensor_name", ""))
            if cat in excluded:
                continue
            bucket = buckets.setdefault(
                cat,
                {col: [] for col in NORM_COLS} | {"_tensor_count": 0},
            )
            bucket["_tensor_count"] += 1
            for col in NORM_COLS:
                bucket[col].append(_parse_float(row.get(col, "")))

    if not run_meta:
        run_meta = {k: "" for k in META_COLS}

    out_rows: list[dict] = []
    for cat in CATEGORY_ORDER:
        if cat not in buckets:
            continue
        bucket = buckets[cat]
        row: dict = {k: run_meta.get(k, "") for k in META_COLS}
        row["tensor_category"] = cat
        row["tensor_category_label"] = CATEGORY_LABEL[cat]
        row["count"] = bucket["_tensor_count"]
        for col in NORM_COLS:
            n, mean, var, std, vmin, vmax = _stats(bucket[col])
            row[f"count_{col}"] = n
            row[f"mean_{col}"] = mean
            row[f"var_{col}"] = var
            row[f"std_{col}"] = std
            row[f"min_{col}"] = vmin
            row[f"max_{col}"] = vmax
        out_rows.append(row)
    return out_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate per-tensor weight-change norms into per-(run, tensor-type) "
            "summary statistics (count, mean, variance, std, min, max)."
        )
    )
    p.add_argument(
        "--csvs",
        type=Path,
        nargs="+",
        required=True,
        metavar="CSV",
        help="One or more per-tensor CSVs (e.g. weight_change_analysis_output/per_tensor_norms/*.csv).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        metavar="OUT_CSV",
        help="Path to write the aggregated CSV.",
    )
    p.add_argument(
        "--exclude-types",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated tensor categories to drop before aggregating. "
            f"Valid keys: {', '.join(CATEGORY_ORDER)}. Example: --exclude-types embed,router"
        ),
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
                    f"{', '.join(CATEGORY_ORDER)}.",
                    file=sys.stderr,
                )
                return 1
            excluded.add(tok)

    all_rows: list[dict] = []
    for csv_path in args.csvs:
        if not csv_path.is_file():
            print(f"[warn] not a file, skipping: {csv_path}", file=sys.stderr)
            continue
        all_rows.extend(aggregate_csv(csv_path, excluded))

    if not all_rows:
        print("No rows aggregated; nothing to write.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = build_output_fields()
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
