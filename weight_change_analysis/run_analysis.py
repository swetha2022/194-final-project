#!/usr/bin/env python3
"""Compute weight-change norms and pairwise angles; write CSV summaries."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Running as `python weight_change_analysis/run_analysis.py` sets __package__ = None;
# ensure project root is on sys.path before importing this package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from weight_change_analysis.config import DEFAULT_FT_ROOT, PRETRAINED_BASE_BY_OPT
from weight_change_analysis.loading import (
    ShardedSafetensors,
    find_latest_step_dir,
    iter_shared_keys,
    read_shard_layout,
    resolve_weights_dir,
)
from weight_change_analysis.metrics import (
    accumulate_multi_dot_and_norms,
    global_max_rms_induced_for_tensor,
    per_tensor_norms,
)
from weight_change_analysis.parsing import RunMeta, parse_run_folder


def discover_run_dirs(ft_root: Path) -> List[Path]:
    return sorted([p for p in ft_root.iterdir() if p.is_dir()])


def resolve_run_paths(run_root: Path) -> Tuple[int, Path]:
    step_dir = find_latest_step_dir(run_root)
    step_num = int(step_dir.name.split("_")[1])
    weights_dir = resolve_weights_dir(step_dir)
    return step_num, weights_dir


def count_parameters(keys: Sequence[str], stores: Sequence[ShardedSafetensors]) -> int:
    total = 0
    base = stores[0]
    for k in keys:
        total += int(base.get_tensor(k).numel())
    return total


def analyze_pretrain_group(
    pretrain_optimizer: str,
    entries: List[Tuple[RunMeta, Path, int, Path]],
    chunk_elems: int,
    spectral_device: torch.device,
    keys_progress: bool,
    max_tensors: int | None,
    pca_k: int | None = None,
) -> Tuple[List[dict], List[dict]]:
    """entries: (meta, run_root, step_num, weights_dir). Returns (norm_rows, angle_rows)."""
    if not entries:
        return [], []

    base_dir = PRETRAINED_BASE_BY_OPT[pretrain_optimizer]
    layouts = [read_shard_layout(base_dir)] + [read_shard_layout(w) for _, _, _, w in entries]
    stores = [ShardedSafetensors(layouts[0])] + [
        ShardedSafetensors(layouts[i + 1]) for i in range(len(entries))
    ]

    common_keys = list(iter_shared_keys([ly for ly in layouts]))
    if max_tensors is not None:
        common_keys = common_keys[: max_tensors]
    param_count = count_parameters(common_keys, stores)

    r = len(entries)
    norm_sq = [0.0] * r
    linf = [0.0] * r
    max_rms = [0.0] * r
    dots_upper = [[0.0] * r for _ in range(r)]

    iterator: Iterable[str]
    if keys_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(common_keys, desc=f"{pretrain_optimizer} tensors")
        except ImportError:
            iterator = common_keys
    else:
        iterator = common_keys

    for key in iterator:
        base_t = stores[0].get_tensor(key)
        fts = [stores[i + 1].get_tensor(key) for i in range(r)]

        ns_part, dots_part, linf_part = accumulate_multi_dot_and_norms(
            base_t, fts, chunk_elems=chunk_elems
        )
        for i in range(r):
            norm_sq[i] += ns_part[i]
            linf[i] = max(linf[i], linf_part[i])
            for j in range(i, r):
                dots_upper[i][j] += dots_part[i][j]

        if base_t.ndim == 2:
            for i in range(r):
                max_rms[i] = max(
                    max_rms[i],
                    global_max_rms_induced_for_tensor(base_t, fts[i], spectral_device),
                )

    z_pca = None
    k_eff_global = 0
    if pca_k is not None and pca_k >= 1 and r >= 2:
        from weight_change_analysis.pca_angles import (
            cosine_and_angle_degrees,
            double_center_gram,
            gram_from_upper_triangle,
            pca_scores_from_centered_gram,
        )

        g = gram_from_upper_triangle(dots_upper, r)
        gc = double_center_gram(g)
        z_pca, k_eff_global = pca_scores_from_centered_gram(gc, pca_k)
    else:
        cosine_and_angle_degrees = None  # type: ignore[assignment,misc]

    norm_rows: List[dict] = []
    for idx, (meta, run_root, step_num, _) in enumerate(entries):
        l2 = math.sqrt(norm_sq[idx]) if norm_sq[idx] > 0 else 0.0
        norm_rows.append(
            {
                "pretrain_optimizer": pretrain_optimizer,
                "run_folder": meta.folder_name,
                "run_path": str(run_root),
                "finetune_type": meta.finetune_type,
                "finetune_optimizer": meta.finetune_optimizer,
                "name_suffix": meta.name_suffix,
                "checkpoint_step": step_num,
                "shared_tensor_keys": len(common_keys),
                "shared_parameter_count": param_count,
                "l_inf": linf[idx],
                "l2": l2,
                "rms_rms_induced_max_over_layers": max_rms[idx],
            }
        )

    angle_rows: List[dict] = []
    for i in range(r):
        for j in range(i + 1, r):
            ni = math.sqrt(norm_sq[i])
            nj = math.sqrt(norm_sq[j])
            dot_ij = dots_upper[i][j]
            if ni == 0.0 or nj == 0.0:
                cos_t = float("nan")
                ang = float("nan")
            else:
                cos_t = dot_ij / (ni * nj)
                cos_t = max(-1.0, min(1.0, cos_t))
                ang = math.degrees(math.acos(cos_t))

            mi, mj = entries[i][0], entries[j][0]
            row = {
                "pretrain_optimizer": pretrain_optimizer,
                "run_i": mi.folder_name,
                "run_j": mj.folder_name,
                "path_run_i": str(entries[i][1]),
                "path_run_j": str(entries[j][1]),
                "cosine_similarity": cos_t,
                "angle_degrees": ang,
                "dot_product": dot_ij,
                "l2_delta_run_i": ni,
                "l2_delta_run_j": nj,
                "shared_parameter_count": param_count,
            }
            if pca_k is not None and z_pca is not None and z_pca.shape[1] > 0:
                assert cosine_and_angle_degrees is not None
                cos_p, ang_p = cosine_and_angle_degrees(z_pca[i], z_pca[j])
                row["pca_target_k"] = pca_k
                row["pca_components_used"] = k_eff_global
                row["cosine_similarity_pca"] = cos_p
                row["angle_degrees_pca"] = ang_p
            elif pca_k is not None:
                row["pca_target_k"] = pca_k
                row["pca_components_used"] = k_eff_global
                row["cosine_similarity_pca"] = float("nan")
                row["angle_degrees_pca"] = float("nan")
            angle_rows.append(row)

    return norm_rows, angle_rows


PER_TENSOR_FIELDS = [
    "pretrain_optimizer",
    "run_folder",
    "run_path",
    "pretrain_base_path",
    "finetune_type",
    "finetune_optimizer",
    "name_suffix",
    "checkpoint_step",
    "tensor_name",
    "tensor_shape",
    "numel",
    "l_inf",
    "l2",
    "rms_rms_induced",
]


def analyze_single_run_per_tensor(
    meta: RunMeta,
    run_root: Path,
    step_num: int,
    weights_dir: Path,
    chunk_elems: int,
    spectral_device: torch.device,
    max_tensors: int | None,
    keys_progress: bool,
) -> List[dict]:
    """Compute per-tensor delta norms for one finetune run vs its pretrained base."""
    base_dir = PRETRAINED_BASE_BY_OPT[meta.pretrain_optimizer]
    base_layout = read_shard_layout(base_dir)
    ft_layout = read_shard_layout(weights_dir)
    base_store = ShardedSafetensors(base_layout)
    ft_store = ShardedSafetensors(ft_layout)

    common_keys = list(iter_shared_keys([base_layout, ft_layout]))
    if max_tensors is not None:
        common_keys = common_keys[: max_tensors]

    iterator: Iterable[str]
    if keys_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(common_keys, desc=f"{meta.folder_name} tensors")
        except ImportError:
            iterator = common_keys
    else:
        iterator = common_keys

    rows: List[dict] = []
    for key in iterator:
        base_t = base_store.get_tensor(key)
        ft_t = ft_store.get_tensor(key)
        linf, l2, rms = per_tensor_norms(
            base_t, ft_t, chunk_elems=chunk_elems, spectral_device=spectral_device
        )
        rows.append(
            {
                "pretrain_optimizer": meta.pretrain_optimizer,
                "run_folder": meta.folder_name,
                "run_path": str(run_root),
                "pretrain_base_path": str(base_dir),
                "finetune_type": meta.finetune_type,
                "finetune_optimizer": meta.finetune_optimizer,
                "name_suffix": meta.name_suffix,
                "checkpoint_step": step_num,
                "tensor_name": key,
                "tensor_shape": ",".join(str(d) for d in tuple(base_t.shape)),
                "numel": int(base_t.numel()),
                "l_inf": linf,
                "l2": l2,
                "rms_rms_induced": rms,
            }
        )
    return rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compare finetuned checkpoints to pretrained Moonlight weights."
    )
    p.add_argument(
        "--ft-root",
        type=Path,
        default=DEFAULT_FT_ROOT,
        help="Directory containing {adam,muon}_ckpt_driving_* run folders",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weight_change_analysis_output"),
        help="Where to write CSV files",
    )
    p.add_argument(
        "--chunk-elements",
        type=int,
        default=4_000_000,
        help="Chunk size (elements) for flat-vector accumulation",
    )
    p.add_argument(
        "--cuda-spectral",
        action="store_true",
        help="Compute RMS→RMS spectral factors on CUDA when available",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show per-tensor progress bar (requires tqdm)",
    )
    p.add_argument(
        "--only-pretrain",
        choices=("adam", "muon", "both"),
        default="both",
        help="Restrict which pretrained base family to analyze",
    )
    p.add_argument(
        "--max-runs-per-group",
        type=int,
        default=None,
        help="Limit runs per pretrain family (useful for smoke tests; default: all)",
    )
    p.add_argument(
        "--max-tensors",
        type=int,
        default=None,
        help="Process only the first N shared tensors (debug / smoke test)",
    )
    p.add_argument(
        "--per-tensor-norms",
        action="store_true",
        help="Also write one CSV per finetune run with per-tensor delta norms "
        "(l_inf, l2, rms_rms_induced) to <output-dir>/<per-tensor-output-subdir>/.",
    )
    p.add_argument(
        "--per-tensor-output-subdir",
        type=str,
        default="per_tensor_norms",
        help="Subdirectory under --output-dir for per-tensor CSVs.",
    )
    p.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip the cross-run aggregate norms + pairwise angles pass "
        "(only meaningful with --per-tensor-norms).",
    )
    p.add_argument(
        "--pca-k",
        type=int,
        default=None,
        metavar="K",
        help="If set (K>=1), also compute pairwise angles in R^K using dual PCA on the "
        "centered Gram matrix of weight-update inner products within each pretrain group. "
        "Requires numpy. Adds columns cosine_similarity_pca, angle_degrees_pca, etc.",
    )
    args = p.parse_args(argv)

    if args.pca_k is not None and args.pca_k < 1:
        print("--pca-k must be >= 1 when set.", file=sys.stderr)
        return 1

    spectral_device = torch.device(
        "cuda" if args.cuda_spectral and torch.cuda.is_available() else "cpu"
    )

    discovered = discover_run_dirs(args.ft_root)
    grouped: Dict[str, List[Tuple[RunMeta, Path, int, Path]]] = defaultdict(list)

    for run_root in discovered:
        name = run_root.name
        try:
            meta = parse_run_folder(name)
        except ValueError as e:
            print(f"[skip] {name}: {e}", file=sys.stderr)
            continue
        if args.only_pretrain == "adam" and meta.pretrain_optimizer != "adam":
            continue
        if args.only_pretrain == "muon" and meta.pretrain_optimizer != "muon":
            continue
        try:
            step_num, weights_dir = resolve_run_paths(run_root)
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}", file=sys.stderr)
            continue
        grouped[meta.pretrain_optimizer].append((meta, run_root, step_num, weights_dir))

    if args.per_tensor_norms:
        per_tensor_dir = args.output_dir / args.per_tensor_output_subdir
        per_tensor_dir.mkdir(parents=True, exist_ok=True)
        total_written = 0
        for pre in ("adam", "muon"):
            if args.only_pretrain == "adam" and pre != "adam":
                continue
            if args.only_pretrain == "muon" and pre != "muon":
                continue
            entries = grouped.get(pre, [])
            if args.max_runs_per_group is not None:
                entries = entries[: args.max_runs_per_group]
            for meta, run_root, step_num, weights_dir in entries:
                print(
                    f"[per-tensor] {meta.folder_name} "
                    f"(spectral device: {spectral_device})...",
                    file=sys.stderr,
                )
                rows = analyze_single_run_per_tensor(
                    meta,
                    run_root,
                    step_num,
                    weights_dir,
                    chunk_elems=args.chunk_elements,
                    spectral_device=spectral_device,
                    max_tensors=args.max_tensors,
                    keys_progress=args.progress,
                )
                out_path = per_tensor_dir / f"{meta.folder_name}.csv"
                write_csv(out_path, PER_TENSOR_FIELDS, rows)
                total_written += 1
                print(
                    f"  wrote {out_path} ({len(rows)} rows)", file=sys.stderr
                )
        print(
            f"Wrote {total_written} per-tensor CSV(s) under {per_tensor_dir}",
            file=sys.stderr,
        )

    if args.skip_aggregate:
        return 0

    all_norm: List[dict] = []
    all_angle: List[dict] = []

    for pre in ("adam", "muon"):
        if args.only_pretrain == "adam" and pre != "adam":
            continue
        if args.only_pretrain == "muon" and pre != "muon":
            continue
        entries = grouped.get(pre, [])
        if args.max_runs_per_group is not None:
            entries = entries[: args.max_runs_per_group]
        if not entries:
            continue
        print(
            f"Analyzing {len(entries)} runs on {pre} pretrain "
            f"(spectral device: {spectral_device})...",
            file=sys.stderr,
        )
        n_rows, a_rows = analyze_pretrain_group(
            pre,
            entries,
            chunk_elems=args.chunk_elements,
            spectral_device=spectral_device,
            keys_progress=args.progress,
            max_tensors=args.max_tensors,
            pca_k=args.pca_k,
        )
        all_norm.extend(n_rows)
        all_angle.extend(a_rows)

    norms_path = args.output_dir / "weight_change_norms.csv"
    angles_path = args.output_dir / "weight_change_pairwise_angles.csv"

    norm_fields = [
        "pretrain_optimizer",
        "run_folder",
        "run_path",
        "finetune_type",
        "finetune_optimizer",
        "name_suffix",
        "checkpoint_step",
        "shared_tensor_keys",
        "shared_parameter_count",
        "l_inf",
        "l2",
        "rms_rms_induced_max_over_layers",
    ]
    angle_fields = [
        "pretrain_optimizer",
        "run_i",
        "run_j",
        "path_run_i",
        "path_run_j",
        "cosine_similarity",
        "angle_degrees",
        "dot_product",
        "l2_delta_run_i",
        "l2_delta_run_j",
        "shared_parameter_count",
    ]
    if args.pca_k is not None:
        angle_fields.extend(
            [
                "pca_target_k",
                "pca_components_used",
                "cosine_similarity_pca",
                "angle_degrees_pca",
            ]
        )

    write_csv(norms_path, norm_fields, all_norm)
    write_csv(angles_path, angle_fields, all_angle)

    print(f"Wrote {norms_path} ({len(all_norm)} rows)", file=sys.stderr)
    print(f"Wrote {angles_path} ({len(all_angle)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
