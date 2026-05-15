#!/usr/bin/env python3
"""Compute effective rank (entropy of singular values) per matrix per checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from effective_rank_analysis.rank_metrics import ranks_for_matrix_2d
from effective_rank_analysis.tensor_meta import classify_tensor
from weight_change_analysis.loading import (
    ShardedSafetensors,
    keys_sorted_by_shard,
    read_shard_layout,
    resolve_weights_dir,
)
from weight_change_analysis.parsing import parse_run_folder

CSV_FIELDS = [
    "run_name",
    "pretrain_optimizer",
    "finetune_type",
    "finetune_optimizer",
    "name_suffix",
    "step",
    "tensor_name",
    "matrix_type",
    "matrix_group",
    "tensor_subtype",
    "layer",
    "expert_index",
    "shape_0",
    "shape_1",
    "min_dim",
    "effective_rank",
    "stable_rank",
]


def discover_step_dirs(run_root: Path) -> List[Path]:
    steps = sorted(
        [p for p in run_root.glob("step_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[1]),
    )
    return steps


def parse_run_meta_safe(run_name: str) -> dict[str, str]:
    try:
        m = parse_run_folder(run_name)
        return {
            "pretrain_optimizer": m.pretrain_optimizer,
            "finetune_type": m.finetune_type,
            "finetune_optimizer": m.finetune_optimizer,
            "name_suffix": m.name_suffix or "",
        }
    except ValueError:
        return {
            "pretrain_optimizer": "",
            "finetune_type": "",
            "finetune_optimizer": "",
            "name_suffix": "",
        }


def iter_run_dirs(ft_root: Path) -> List[Path]:
    return sorted([p for p in ft_root.iterdir() if p.is_dir()])


def process_checkpoint(
    run_name: str,
    meta: dict[str, str],
    step: int,
    weights_dir: Path,
    device: torch.device,
    writer: csv.DictWriter,
    keys_progress: bool,
    max_tensors: int | None,
) -> int:
    layout = read_shard_layout(weights_dir)
    store = ShardedSafetensors(layout)
    keys = keys_sorted_by_shard(layout)
    rows = 0

    iterator: Iterable[str]
    if keys_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(keys, desc=f"{run_name} step_{step}")
        except ImportError:
            iterator = keys
    else:
        iterator = keys

    for key in iterator:
        if max_tensors is not None and rows >= max_tensors:
            break
        cls = classify_tensor(key)
        if cls is None:
            continue
        t = store.get_tensor(key)
        if t.ndim != 2 or not t.dtype.is_floating_point:
            continue
        eff, stab, _k = ranks_for_matrix_2d(t, device)
        m0, m1 = int(t.shape[0]), int(t.shape[1])
        writer.writerow(
            {
                "run_name": run_name,
                "pretrain_optimizer": meta["pretrain_optimizer"],
                "finetune_type": meta["finetune_type"],
                "finetune_optimizer": meta["finetune_optimizer"],
                "name_suffix": meta["name_suffix"],
                "step": step,
                "tensor_name": key,
                "matrix_type": cls.matrix_type,
                "matrix_group": cls.matrix_group,
                "tensor_subtype": cls.tensor_subtype,
                "layer": "" if cls.layer is None else cls.layer,
                "expert_index": "" if cls.expert_index is None else cls.expert_index,
                "shape_0": m0,
                "shape_1": m1,
                "min_dim": min(m0, m1),
                "effective_rank": f"{eff:.8g}",
                "stable_rank": f"{stab:.8g}",
            }
        )
        rows += 1
    store.drop_cache()
    return rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ft-root",
        type=Path,
        default=Path("/scratch/celine/ft_out"),
        help="Root directory containing one folder per finetuning run.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "effective_rank_analysis_output",
        help="Directory to write <run_name>_effective_rank.csv files.",
    )
    ap.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Specific run folder names under ft-root (default: all).",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device for SVD (default: cuda if available).",
    )
    ap.add_argument(
        "--keys-progress",
        action="store_true",
        help="Show a tqdm progress bar per checkpoint (requires tqdm).",
    )
    ap.add_argument(
        "--max-tensors",
        type=int,
        default=None,
        help="Stop after this many classified 2D tensors (debug).",
    )
    args = ap.parse_args()
    device = torch.device(args.device)

    run_dirs = iter_run_dirs(args.ft_root)
    if args.runs:
        want = set(args.runs)
        run_dirs = [p for p in run_dirs if p.name in want]

    for run_root in run_dirs:
        run_name = run_root.name
        steps = discover_step_dirs(run_root)
        if not steps:
            print(f"skip {run_name}: no step_* checkpoints", flush=True)
            continue
        out_csv = args.output_dir / f"{run_name}_effective_rank.csv"
        meta = parse_run_meta_safe(run_name)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for sd in steps:
                step = int(sd.name.split("_")[1])
                try:
                    wdir = resolve_weights_dir(sd)
                except FileNotFoundError as e:
                    print(f"skip {run_name} {sd.name}: {e}", flush=True)
                    continue
                n = process_checkpoint(
                    run_name,
                    meta,
                    step,
                    wdir,
                    device,
                    writer,
                    keys_progress=args.keys_progress,
                    max_tensors=args.max_tensors,
                )
                f.flush()
                print(f"{run_name} {sd.name}: wrote {n} tensor rows", flush=True)


if __name__ == "__main__":
    main()
