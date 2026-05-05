#!/usr/bin/env python3
"""Draw a bar chart of selected norm columns for one row of weight_change_norms.csv."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Canonical key -> (CSV column, x-tick label for bar, bar color)
NORM_SPECS: dict[str, tuple[str, str, str]] = {
    "l_inf": ("l_inf", r"$L_\infty$", "#2c5282"),
    "l2": ("l2", r"$L_2$", "#2f855a"),
    "rms": ("rms_rms_induced_max_over_layers", "RMS→RMS\n(max over layers)", "#9b2c2c"),
}

# Y-axis label (mathtext) for the quantity plotted — one bar group or single-metric figures
NORM_YLABEL: dict[str, str] = {
    "l_inf": r"$\|\Delta W\|_{\infty}$",
    "l2": r"$\|\Delta W\|_{\ell_2}$",
    "rms": r"$\|\Delta W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}$" + "\n(max over 2D layers)",
}

# User-facing aliases -> canonical key
NORM_ALIASES: dict[str, str] = {
    "l_inf": "l_inf",
    "linf": "l_inf",
    "l-infinity": "l_inf",
    "infty": "l_inf",
    "inf": "l_inf",
    "l2": "l2",
    "l_2": "l2",
    "rms": "rms",
    "rms_rms": "rms",
    "rms_rms_induced": "rms",
    "rms_rms_induced_max_over_layers": "rms",
}

DEFAULT_NORM_ORDER = ("l_inf", "l2", "rms")


def ylabel_for_norm_selection(norm_keys: list[str]) -> str:
    """Y-axis label matching the norm(s) shown; multi-norm charts get a generic caption."""
    if len(norm_keys) == 1:
        return NORM_YLABEL[norm_keys[0]]
    return r"$\|\Delta W\|$" + "\n(several norms; see $x$-axis labels)"


def parse_norm_list(s: str) -> list[str]:
    """Return ordered canonical norm keys from a comma-separated list."""
    keys: list[str] = []
    seen: set[str] = set()
    for raw in s.split(","):
        token = raw.strip().lower().replace(" ", "")
        if not token:
            continue
        canon = NORM_ALIASES.get(token)
        if canon is None:
            allowed = ", ".join(sorted(set(NORM_ALIASES.keys())))
            raise SystemExit(f"Unknown norm {raw!r}. Use one of: {allowed}")
        if canon in seen:
            continue
        seen.add(canon)
        keys.append(canon)
    if not keys:
        raise SystemExit("At least one norm is required (see --norms).")
    return keys


def norms_from_exclude(exclude_s: str) -> list[str]:
    ex: set[str] = set()
    for raw in exclude_s.split(","):
        token = raw.strip().lower().replace(" ", "")
        if not token:
            continue
        canon = NORM_ALIASES.get(token)
        if canon is None:
            raise SystemExit(f"Unknown norm in --exclude-norms: {raw!r}")
        ex.add(canon)
    out = [k for k in DEFAULT_NORM_ORDER if k not in ex]
    if not out:
        raise SystemExit("--exclude-norms removed every norm; leave at least one.")
    return out


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


def pick_row_by_spec(
    rows: list[dict[str, str]],
    pretrain: str,
    finetune_optimizer: str,
    finetune_type: str,
    name_suffix: str | None,
) -> tuple[int, dict[str, str]]:
    pre = pretrain.strip().lower()
    if pre not in ("adam", "muon"):
        raise SystemExit(f"--pretrain must be adam or muon, got {pretrain!r}")

    ft_opt = finetune_optimizer.strip().lower()
    ft_type = normalize_finetune_type(finetune_type)

    matches: list[tuple[int, dict[str, str]]] = []
    for i, r in enumerate(rows):
        if (r.get("pretrain_optimizer", "").lower() != pre
                or r.get("finetune_optimizer", "").lower() != ft_opt
                or r.get("finetune_type", "").lower() != ft_type):
            continue
        suf = (r.get("name_suffix") or "").strip()
        if name_suffix is not None:
            want = name_suffix.strip()
            if suf != want:
                continue
        matches.append((i, r))

    if not matches:
        hint = f" (with name_suffix={name_suffix!r})" if name_suffix is not None else ""
        raise SystemExit(
            f"No row matches pretrain={pre!r}, finetune_optimizer={ft_opt!r}, "
            f"finetune_type={ft_type!r}{hint}.\n"
            "Use --list to see available rows, or try --name-suffix if several variants exist."
        )
    if len(matches) > 1:
        lines = [
            f"  run_folder={r.get('run_folder')!r}  name_suffix={r.get('name_suffix')!r}"
            for _, r in matches
        ]
        raise SystemExit(
            "Multiple rows match that combination; disambiguate with --name-suffix "
            "(see name_suffix column) or use --run-folder.\n" + "\n".join(lines)
        )
    return matches[0]


def pick_row_legacy(
    rows: list[dict[str, str]], row: int | None, run_folder: str | None
) -> tuple[int, dict[str, str]]:
    if run_folder is not None:
        matches = [(i, r) for i, r in enumerate(rows) if r.get("run_folder") == run_folder]
        if not matches:
            candidates = [r.get("run_folder", "") for r in rows]
            raise SystemExit(
                f"No row with run_folder={run_folder!r}. Known run_folder values:\n"
                + "\n".join(f"  {c!r}" for c in candidates)
            )
        if len(matches) > 1:
            raise SystemExit(f"Multiple rows match run_folder={run_folder!r}; use --row instead.")
        return matches[0]
    if row is None:
        raise SystemExit("Internal: row is None without run_folder")
    if row < 0 or row >= len(rows):
        raise SystemExit(f"--row must be in [0, {len(rows) - 1}] for this file ({len(rows)} data rows).")
    return row, rows[row]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Bar plot of selected norm columns (l_inf, l2, RMS→RMS max) for one CSV row."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("weight_change_analysis_output/weight_change_norms.csv"),
        help="Path to weight_change_norms.csv",
    )
    p.add_argument(
        "--pretrain",
        choices=("adam", "muon"),
        default=None,
        help="Pretraining optimizer (with --finetune-optimizer and --finetune-type)",
    )
    p.add_argument(
        "--finetune-optimizer",
        dest="finetune_optimizer",
        choices=("adamw", "muon"),
        default=None,
        help="Fine-tuning optimizer (with --pretrain and --finetune-type)",
    )
    p.add_argument(
        "--finetune-type",
        choices=("lora", "full"),
        default=None,
        help="lora or full fine-tuning (with --pretrain and --finetune-optimizer)",
    )
    p.add_argument(
        "--name-suffix",
        default=None,
        metavar="SUFFIX",
        help="Optional exact name_suffix when several runs share the same triple above "
        '(e.g. "4-28" or "swa_4-28"; empty string matches rows with blank suffix)',
    )
    p.add_argument(
        "--row",
        type=int,
        default=None,
        metavar="N",
        help="0-based data row index (alternative to selection by run metadata)",
    )
    p.add_argument(
        "--run-folder",
        type=str,
        default=None,
        metavar="NAME",
        help="Select row by exact run_folder column (alternative selector)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (.png or .pdf). Default: norms_bar_<run_folder>.png next to CSV",
    )
    p.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale on the y-axis (useful when L2 dwarfs L∞)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override plot title (default: run_folder and key metadata)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print pretrain, finetune type/opt, suffix, and run_folder; then exit",
    )
    p.add_argument(
        "--norms",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated norms to plot, in bar order. "
        "Keys: l_inf (aliases: linf, inf), l2, rms (aliases: rms_rms, …). "
        "Default: l_inf,l2,rms. Example: --norms l2,rms",
    )
    p.add_argument(
        "--exclude-norms",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated norms to omit (same keys as --norms). "
        "Cannot be combined with --norms.",
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

    if args.list:
        for r in rows:
            suf = r.get("name_suffix") or ""
            print(
                f"{r.get('pretrain_optimizer')}\t{r.get('finetune_type')}\t"
                f"{r.get('finetune_optimizer')}\t{suf!r}\t{r.get('run_folder')}"
            )
        return 0

    spec = (args.pretrain, args.finetune_optimizer, args.finetune_type)
    spec_count = sum(x is not None for x in spec)
    if args.name_suffix is not None and spec_count != 3:
        print(
            "--name-suffix only applies with --pretrain, --finetune-optimizer, and --finetune-type.",
            file=sys.stderr,
        )
        return 1
    if spec_count not in (0, 3):
        print(
            "Use all three of --pretrain, --finetune-optimizer, and --finetune-type together, "
            "or use --row / --run-folder instead.",
            file=sys.stderr,
        )
        return 1

    if args.row is not None and (args.run_folder or spec_count):
        print("Do not combine --row with --run-folder or the --pretrain triple.", file=sys.stderr)
        return 1
    if args.run_folder and spec_count:
        print("Do not combine --run-folder with the --pretrain triple.", file=sys.stderr)
        return 1

    if spec_count == 3:
        name_suffix = args.name_suffix
        row_idx, r = pick_row_by_spec(
            rows,
            args.pretrain,
            args.finetune_optimizer,
            args.finetune_type,
            name_suffix=name_suffix,
        )
    elif args.run_folder is not None:
        row_idx, r = pick_row_legacy(rows, None, args.run_folder)
    elif args.row is not None:
        row_idx, r = pick_row_legacy(rows, args.row, None)
    else:
        print(
            "Specify selection: --pretrain, --finetune-optimizer, and --finetune-type; "
            "or --run-folder; or --row; or --list.",
            file=sys.stderr,
        )
        return 1

    if args.norms is not None and args.exclude_norms is not None:
        print("Use either --norms or --exclude-norms, not both.", file=sys.stderr)
        return 1

    if args.exclude_norms is not None:
        norm_keys = norms_from_exclude(args.exclude_norms)
    elif args.norms is not None:
        norm_keys = parse_norm_list(args.norms)
    else:
        norm_keys = list(DEFAULT_NORM_ORDER)

    try:
        values: list[float] = []
        labels: list[str] = []
        colors: list[str] = []
        for k in norm_keys:
            col, lab, colr = NORM_SPECS[k]
            values.append(float(r[col]))
            labels.append(lab)
            colors.append(colr)
    except (KeyError, ValueError) as e:
        print(f"Missing or invalid norm columns: {e}", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    x = range(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel_for_norm_selection(norm_keys))
    if args.log_y:
        ax.set_yscale("log")

    title = args.title
    if not title:
        parts = [
            r.get("run_folder", ""),
            f"pretrain={r.get('pretrain_optimizer', '')}",
            f"{r.get('finetune_type', '')} / {r.get('finetune_optimizer', '')}",
        ]
        if r.get("name_suffix"):
            parts.append(f"suffix={r['name_suffix']}")
        title = "  |  ".join(p for p in parts if p)
    ax.set_title(title, fontsize=11)

    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v:.4g}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    out = args.output
    if out is None:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in r.get("run_folder", f"row{row_idx}"))
        out = csv_path.parent / f"norms_bar_{safe_name}.png"
    else:
        out = out.resolve()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
