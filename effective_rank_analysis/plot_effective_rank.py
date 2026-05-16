#!/usr/bin/env python3
"""Plot effective rank vs training step from compute_effective_ranks.py CSVs."""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


AggKey = Tuple[str, int, str]  # run_name, step, matrix_group


def read_csv_rows(paths: List[Path]) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    return rows


def aggregate_mean(
    rows: List[dict],
    group_field: str,
) -> Dict[Tuple[str, int, str], Tuple[float, int]]:
    """
    Return map (run_name, step, group) -> (sum_effective_rank, count)
    for averaging.
    """
    sums_counts: DefaultDict[Tuple[str, int, str], List[float]] = defaultdict(
        lambda: [0.0, 0]
    )
    for row in rows:
        run = row["run_name"]
        step = int(row["step"])
        grp = row[group_field]
        v = float(row["effective_rank"])
        k = (run, step, grp)
        sums_counts[k][0] += v
        sums_counts[k][1] += 1
    out: Dict[Tuple[str, int, str], Tuple[float, int]] = {}
    for k, (s, c) in sums_counts.items():
        out[k] = (s, c)
    return out


def aggregate_median_by_list(
    rows: List[dict],
    group_field: str,
) -> Dict[Tuple[str, int, str], List[float]]:
    buckets: DefaultDict[Tuple[str, int, str], List[float]] = defaultdict(list)
    for row in rows:
        run = row["run_name"]
        step = int(row["step"])
        grp = row[group_field]
        buckets[(run, step, grp)].append(float(row["effective_rank"]))
    return dict(buckets)


def plot_lines(
    agg: Dict[Tuple[str, int, str], float],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    # series: (run, group) -> sorted list of (step, value)
    series: DefaultDict[Tuple[str, str], List[Tuple[int, float]]] = defaultdict(list)
    for (run, step, grp), val in agg.items():
        series[(run, grp)].append((step, val))
    for k in series:
        series[k].sort(key=lambda t: t[0])

    plt.figure(figsize=(11, 6))
    for (run, grp), pts in sorted(series.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = f"{run} | {grp}"
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=label)

    plt.xlabel("Training step")
    plt.ylabel("Mean effective rank")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_facets_by_group(
    agg: Dict[Tuple[str, int, str], float],
    title: str,
    out_path: Path,
) -> None:
    """One subplot per matrix_group; lines colored by run_name."""
    import matplotlib.pyplot as plt

    groups = sorted({g for (_, _, g) in agg})
    runs = sorted({r for (r, _, _) in agg})
    n = len(groups)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    cmap = plt.get_cmap("tab10")
    run_color = {run: cmap(i % 10) for i, run in enumerate(runs)}

    for idx, grp in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        for run in runs:
            pts = sorted(
                [(s, agg[(run, s, grp)]) for (r, s, g) in agg if r == run and g == grp],
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=2.5,
                linewidth=1.1,
                color=run_color[run],
                label=run,
            )
        ax.set_title(grp, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("mean eff. rank")
        ax.grid(True, alpha=0.25)
    for j in range(len(groups), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    from matplotlib.lines import Line2D

    leg_handles = [Line2D([0], [0], color=run_color[r], linewidth=2, label=r) for r in runs]
    fig.legend(handles=leg_handles, loc="upper center", ncol=min(6, len(runs)), fontsize=8)
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "csvs",
        nargs="*",
        help="CSV paths or globs (default: effective_rank_analysis_output/*_effective_rank.csv).",
    )
    ap.add_argument(
        "--group-by",
        choices=("matrix_group", "matrix_type"),
        default="matrix_group",
        help="Column to aggregate (mean) over tensors before plotting.",
    )
    ap.add_argument(
        "--stat",
        choices=("mean", "median"),
        default="mean",
        help="How to aggregate tensors within a group at each step.",
    )
    ap.add_argument(
        "--facet",
        action="store_true",
        help="If set, write a faceted PNG (one panel per group) instead of one crowded line chart.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "effective_rank_analysis_output" / "effective_rank_mean_by_group.png",
        help="Output PNG path.",
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        default=None,
        metavar="RUN_NAME",
        help=(
            "Only include these run_name values (folder names). "
            "Useful with a broad CSV glob to overlay a subset of runs on one figure."
        ),
    )
    args = ap.parse_args()

    if args.csvs:
        paths: List[Path] = []
        for pat in args.csvs:
            if any(c in pat for c in "*?["):
                paths.extend(Path(p) for p in glob.glob(pat))
            else:
                paths.append(Path(pat))
    else:
        paths = sorted(
            (_PROJECT_ROOT / "effective_rank_analysis_output").glob("*_effective_rank.csv")
        )

    paths = [p for p in paths if p.is_file()]
    if not paths:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    rows = read_csv_rows(paths)
    if not rows:
        print("CSV files were empty.", file=sys.stderr)
        sys.exit(1)

    if args.runs is not None:
        want = set(args.runs)
        rows = [r for r in rows if r["run_name"] in want]
        if not rows:
            print(f"No rows left after --runs filter {sorted(want)!r}.", file=sys.stderr)
            sys.exit(1)
        missing = want - {r["run_name"] for r in rows}
        if missing:
            print(f"warning: no data for runs {sorted(missing)!r}", file=sys.stderr)

    group_field = args.group_by
    if args.stat == "mean":
        sums = aggregate_mean(rows, group_field)
        agg: Dict[Tuple[str, int, str], float] = {
            k: s / c for k, (s, c) in sums.items() if c > 0
        }
    else:
        med_buckets = aggregate_median_by_list(rows, group_field)
        agg = {}
        for k, vals in med_buckets.items():
            vals_sorted = sorted(vals)
            mid = len(vals_sorted) // 2
            if len(vals_sorted) % 2 == 1:
                med = vals_sorted[mid]
            else:
                med = 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
            agg[k] = med

    title = f"Effective rank vs step ({args.stat} over tensors, by {group_field})"
    if args.runs is not None:
        title += f" — runs: {', '.join(sorted(set(args.runs)))}"
    if args.facet:
        plot_facets_by_group(agg, title, args.output)
    else:
        plot_lines(agg, title, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
