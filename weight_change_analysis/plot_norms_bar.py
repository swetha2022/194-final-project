"""
Draw a bar chart of selected norm columns for one row of weight_change_norms.csv.
 
How the norms are computed
--------------------------
Each bar represents a norm of the weight-update vector Δ = θ_finetuned − θ_pretrained,
accumulated over all shared parameter tensors.
 
  L∞   : max |Δ[j]|  over all parameters j.
  L2   : √(Σ Δ[j]²)  (Euclidean length of the flattened parameter delta).
  RMS→RMS : for each 2-D weight matrix W (shape m×n),
             ‖ΔW‖_{rms→rms} = √(n/m) · σ_max(ΔW),
             where σ_max is the largest singular value computed via power iteration.
             The bar reports the maximum of this quantity over all 2-D layers.
             Use --rms-layers-log to see which layers dominate.
"""
 
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
 
# Y-axis label (mathtext) for the quantity plotted
NORM_YLABEL: dict[str, str] = {
    "l_inf": r"$\|\Delta W\|_{\infty}$  (max absolute weight change)",
    "l2": r"$\|\Delta W\|_{\ell_2}$  (Euclidean norm of weight delta)",
    "rms": (
        r"$\|\Delta W\|_{\mathrm{RMS}\rightarrow\mathrm{RMS}}$"
        "\n(max over 2-D layers; "
        r"$= \sqrt{n/m}\,\sigma_{\max}(\Delta W)$)"
    ),
}
 
# Generic y-axis label for multi-norm charts
_MULTI_NORM_YLABEL = r"$\|\Delta W\|$" + "  (see x-axis for norm type)"
 
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
 
# Caption appended to the figure as a subtitle / footnote
_METHODOLOGY_CAPTION = (
    "Δ = finetuned − pretrained weights, streamed over all shared tensors.\n"
    r"L∞ = max|Δ|; L2 = ‖Δ‖; RMS→RMS = max_{2-D layers} √(n/m)·σ_max(ΔW)."
)
 
 
def ylabel_for_norm_selection(norm_keys: list[str]) -> str:
    if len(norm_keys) == 1:
        return NORM_YLABEL[norm_keys[0]]
    return _MULTI_NORM_YLABEL
 
 
def parse_norm_list(s: str) -> list[str]:
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
            "Use --list to see available rows."
        )
    if len(matches) > 1:
        lines = [
            f"  run_folder={r.get('run_folder')!r}  name_suffix={r.get('name_suffix')!r}"
            for _, r in matches
        ]
        raise SystemExit(
            "Multiple rows match; disambiguate with --name-suffix or --run-folder.\n"
            + "\n".join(lines)
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
                f"No row with run_folder={run_folder!r}. Known values:\n"
                + "\n".join(f"  {c!r}" for c in candidates)
            )
        if len(matches) > 1:
            raise SystemExit(f"Multiple rows match run_folder={run_folder!r}; use --row.")
        return matches[0]
    if row is None:
        raise SystemExit("Internal: row is None without run_folder")
    if row < 0 or row >= len(rows):
        raise SystemExit(
            f"--row must be in [0, {len(rows) - 1}] ({len(rows)} data rows)."
        )
    return row, rows[row]
 
 
def _layer_filter_note(r: dict[str, str]) -> str:
    lf = (r.get("layer_filter") or "all").strip()
    if lf and lf != "all":
        return f"layers: {lf}"
    return "all layers"
 
 
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
    p.add_argument("--pretrain", choices=("adam", "muon"), default=None)
    p.add_argument("--finetune-optimizer", dest="finetune_optimizer",
                   choices=("adamw", "muon"), default=None)
    p.add_argument("--finetune-type", choices=("lora", "full"), default=None)
    p.add_argument("--name-suffix", default=None, metavar="SUFFIX")
    p.add_argument("--row", type=int, default=None, metavar="N")
    p.add_argument("--run-folder", type=str, default=None, metavar="NAME")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--log-y", action="store_true")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--list", action="store_true",
                   help="Print available rows and exit")
    p.add_argument("--norms", type=str, default=None, metavar="LIST",
                   help="Comma-separated norms to plot (l_inf, l2, rms). Default: all three.")
    p.add_argument("--exclude-norms", type=str, default=None, metavar="LIST")
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
 
    if args.list:
        for r in rows:
            suf = r.get("name_suffix") or ""
            lf = r.get("layer_filter") or "all"
            print(
                f"{r.get('pretrain_optimizer')}\t{r.get('finetune_type')}\t"
                f"{r.get('finetune_optimizer')}\t{suf!r}\t{lf}\t{r.get('run_folder')}"
            )
        return 0
 
    spec = (args.pretrain, args.finetune_optimizer, args.finetune_type)
    spec_count = sum(x is not None for x in spec)
    if args.name_suffix is not None and spec_count != 3:
        print("--name-suffix only applies with the full --pretrain triple.", file=sys.stderr)
        return 1
    if spec_count not in (0, 3):
        print(
            "Use all three of --pretrain, --finetune-optimizer, --finetune-type together.",
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
        row_idx, r = pick_row_by_spec(
            rows, args.pretrain, args.finetune_optimizer,
            args.finetune_type, name_suffix=args.name_suffix,
        )
    elif args.run_folder is not None:
        row_idx, r = pick_row_legacy(rows, None, args.run_folder)
    elif args.row is not None:
        row_idx, r = pick_row_legacy(rows, args.row, None)
    else:
        print(
            "Specify selection: --pretrain/--finetune-optimizer/--finetune-type, "
            "or --run-folder, or --row, or --list.",
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
 
    fig, ax = plt.subplots(figsize=(7, 5.2), layout="constrained")
    x = range(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
 
    # Y-axis label — always present, uses mathtext formula
    ax.set_ylabel(ylabel_for_norm_selection(norm_keys), fontsize=9, labelpad=6)
 
    if args.log_y:
        ax.set_yscale("log")
 
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.55, color="gray", zorder=1)
 
    title = args.title
    if not title:
        parts = [
            r.get("run_folder", ""),
            f"pretrain={r.get('pretrain_optimizer', '')}",
            f"{r.get('finetune_type', '')} / {r.get('finetune_optimizer', '')}",
        ]
        if r.get("name_suffix"):
            parts.append(f"suffix={r['name_suffix']}")
        parts.append(_layer_filter_note(r))
        title = "  |  ".join(p for p in parts if p)
    ax.set_title(title, fontsize=10, pad=6)
 
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
 
    # Methodology caption below the plot
    if not args.no_caption:
        fig.text(
            0.5, -0.02,
            _METHODOLOGY_CAPTION,
            ha="center",
            va="top",
            fontsize=7,
            color="#555555",
            style="italic",
        )
 
    out = args.output
    if out is None:
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in r.get("run_folder", f"row{row_idx}")
        )
        out = csv_path.parent / f"norms_bar_{safe_name}.png"
    else:
        out = out.resolve()
 
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(str(out))
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())