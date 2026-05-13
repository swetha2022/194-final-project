"""
Compute weight-change norms and pairwise angles; write CSV summaries.
 
Methodology overview
--------------------
For each fine-tuning run we compute the *weight delta*:
 
    Δᵢ = θ_finetuned_i  −  θ_pretrained
 
where θ is the full concatenation of all shared parameter tensors (flattened),
streamed in chunks so that the full vectors are never materialised in RAM.
 
Norms reported per run
~~~~~~~~~~~~~~~~~~~~~~
- L∞ (l_inf)   : max |Δᵢ[j]| over all parameters j.
- L2 (l2)      : √(Σ Δᵢ[j]²).
- RMS→RMS induced (rms_rms_induced_max_over_layers)
              : for each 2-D weight matrix W with shape (m, n),
                ‖ΔW‖_{rms→rms} = √(n/m) · σ_max(ΔW),
                where σ_max is the largest singular value computed via
                power iteration.  We report the *max over all 2-D layers*.
 
Pairwise angles
~~~~~~~~~~~~~~~
The cosine similarity and angle between two weight-update vectors Δᵢ and Δⱼ
require only their dot product and norms — no explicit storage of the
billion-dimensional vectors.  All three quantities are accumulated during the
same streaming pass.
 
PCA-space angles (--pca-k K)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Direct inner-product angles in the full parameter space can be noisy when many
runs are compared.  The dual (kernel) PCA trick projects each run's Δᵢ into a
K-dimensional score space that captures the dominant directions of variation:
 
  1. Gram matrix G[i,j] = <Δᵢ, Δⱼ>  (m × m, already in memory).
  2. Double-centre G to get H·G·H (removes the mean update direction).
  3. Eigendecompose H·G·H; scores Z = V·√Λ  (m × k_eff rows in R^k_eff).
  4. Report cos∠(zᵢ, zⱼ) as the PCA-space angle.
 
This is equivalent to PCA on the (m × d) weight-delta matrix without ever
forming it.  We log cumulative variance explained so the choice of K is
transparent — see stderr output during the run.
 
Layer filtering (--layer-categories)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pass a comma-separated list of layer categories to restrict the analysis to
specific layer types (e.g. attention_qkv,mlp).  Available categories:
  attention_qkv, attention_out, mlp, embed, lm_head, norm, other
"""
 
from __future__ import annotations
 
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
 
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
)
from weight_change_analysis.parsing import RunMeta, parse_run_folder
 
# Diagnostics module (lives alongside this file)
try:
    from weight_change_analysis.pca_diagnostics import (
        filter_keys_by_category,
        log_rms_contributions,
        pca_scores_with_diagnostics,
        recommend_k,
        summarize_key_categories,
    )
    _DIAG_AVAILABLE = True
except ImportError:
    _DIAG_AVAILABLE = False
 
 
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
    max_tensors: Optional[int],
    pca_k: Optional[int] = None,
    layer_categories: Optional[List[str]] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Compute per-run norms and pairwise angles for one pretrain-optimizer group.
 
    Parameters
    ----------
    entries          : (meta, run_root, step_num, weights_dir) per finetuned run.
    chunk_elems      : number of scalar elements per streaming chunk (RAM control).
    spectral_device  : device for power-iteration spectral norm (cpu or cuda).
    pca_k            : if set, compute angles in a K-dim PCA score space and log
                       cumulative variance explained so the choice of K is justified.
    layer_categories : if set, restrict analysis to these layer categories
                       (e.g. ["attention_qkv", "mlp"]).  None = all layers.
 
    Returns
    -------
    norm_rows, angle_rows : lists of dicts ready for CSV writing.
    """
    if not entries:
        return [], []
 
    base_dir = PRETRAINED_BASE_BY_OPT[pretrain_optimizer]
    layouts = [read_shard_layout(base_dir)] + [read_shard_layout(w) for _, _, _, w in entries]
    stores = [ShardedSafetensors(layouts[0])] + [
        ShardedSafetensors(layouts[i + 1]) for i in range(len(entries))
    ]
 
    common_keys = list(iter_shared_keys([ly for ly in layouts]))
 
    # --- Layer filtering ---------------------------------------------------
    if layer_categories is not None and _DIAG_AVAILABLE:
        all_cats = summarize_key_categories(common_keys)
        print(
            f"[layers] Before filtering: {len(common_keys)} keys — "
            + ", ".join(f"{cat}={n}" for cat, n in sorted(all_cats.items())),
            file=sys.stderr,
        )
        common_keys = filter_keys_by_category(common_keys, layer_categories)
        print(
            f"[layers] After filtering to {layer_categories}: {len(common_keys)} keys.",
            file=sys.stderr,
        )
        if not common_keys:
            print(
                "[layers] WARNING: no keys remain after filtering — check category names.",
                file=sys.stderr,
            )
            return [], []
 
    if max_tensors is not None:
        common_keys = common_keys[:max_tensors]
 
    param_count = count_parameters(common_keys, stores)
 
    r = len(entries)
    norm_sq = [0.0] * r
    linf = [0.0] * r
    max_rms = [0.0] * r
    dots_upper = [[0.0] * r for _ in range(r)]
 
    # Per-layer RMS values (for top-layer reporting, reviewer request)
    # key -> max RMS over runs (for inventory log)
    key_rms_max: Dict[str, float] = {}
 
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
            layer_rms_vals = []
            for i in range(r):
                rms_val = global_max_rms_induced_for_tensor(base_t, fts[i], spectral_device)
                max_rms[i] = max(max_rms[i], rms_val)
                layer_rms_vals.append(rms_val)
            # Track maximum RMS across runs for this key (for reviewer inventory)
            key_rms_max[key] = max(layer_rms_vals)
 
    # Log which layers dominate the RMS norm (addresses "RMS has lots of matrices" critique)
    if _DIAG_AVAILABLE and key_rms_max:
        log_rms_contributions(key_rms_max, top_n=15)
 
    # --- PCA for pairwise angles -------------------------------------------
    # We use the dual PCA trick: the m×m Gram matrix G[i,j] = <Δᵢ,Δⱼ> has already
    # been accumulated above.  Double-centring and eigendecomposing it gives score
    # vectors z_i ∈ R^k whose inner products reproduce those of the (unobservable)
    # centered weight-delta vectors in the leading K-dimensional subspace.
    # Cumulative variance explained is logged to stderr to justify the choice of K.
    z_pca = None
    k_eff_global = 0
    cumvar_global = None
    if pca_k is not None and pca_k >= 1 and r >= 2:
        from weight_change_analysis.pca_angles import (
            cosine_and_angle_degrees,
            double_center_gram,
            gram_from_upper_triangle,
        )
 
        g = gram_from_upper_triangle(dots_upper, r)
        gc = double_center_gram(g)
 
        if _DIAG_AVAILABLE:
            # Use diagnostic version that logs variance explained
            z_pca, k_eff_global, cumvar_global = pca_scores_with_diagnostics(
                gc, pca_k, verbose=True
            )
            k90 = recommend_k(cumvar_global, threshold=0.90)
            k95 = recommend_k(cumvar_global, threshold=0.95)
            print(
                f"[PCA] Components needed for 90% variance: k={k90}; "
                f"for 95%: k={k95}.  "
                f"Requested k={pca_k}.",
                file=sys.stderr,
            )
        else:
            from weight_change_analysis.pca_angles import pca_scores_from_centered_gram
            z_pca, k_eff_global = pca_scores_from_centered_gram(gc, pca_k)
            cumvar_global = None
    else:
        cosine_and_angle_degrees = None  # type: ignore[assignment,misc]
 
    # If we used the diagnostic path, cosine_and_angle_degrees may not be bound
    if pca_k is not None and r >= 2:
        from weight_change_analysis.pca_angles import cosine_and_angle_degrees  # noqa: F811
 
    # --- Norm rows --------------------------------------------------------
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
                "layer_filter": ",".join(layer_categories) if layer_categories else "all",
                "l_inf": linf[idx],
                "l2": l2,
                "rms_rms_induced_max_over_layers": max_rms[idx],
            }
        )
 
    # --- Angle rows -------------------------------------------------------
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
                "layer_filter": ",".join(layer_categories) if layer_categories else "all",
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
                # Variance captured at k_eff_global
                if cumvar_global is not None and len(cumvar_global) >= k_eff_global > 0:
                    row["pca_variance_explained"] = round(
                        float(cumvar_global[k_eff_global - 1]), 6
                    )
                else:
                    row["pca_variance_explained"] = float("nan")
                row["cosine_similarity_pca"] = cos_p
                row["angle_degrees_pca"] = ang_p
            elif pca_k is not None:
                row["pca_target_k"] = pca_k
                row["pca_components_used"] = k_eff_global
                row["pca_variance_explained"] = float("nan")
                row["cosine_similarity_pca"] = float("nan")
                row["angle_degrees_pca"] = float("nan")
            angle_rows.append(row)
 
    return norm_rows, angle_rows
 
 
def write_csv(path: Path, fieldnames: Sequence[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow(row)
 
 
def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Compare finetuned checkpoints to pretrained Moonlight weights.\n\n"
            "Angles are computed as follows:\n"
            "  1. For every shared tensor key, stream Δ = finetuned − pretrained in\n"
            "     chunks and accumulate ‖Δᵢ‖², ‖Δᵢ‖∞, and <Δᵢ,Δⱼ> for all pairs.\n"
            "  2. Full-space angle: cos⁻¹(<Δᵢ,Δⱼ> / (‖Δᵢ‖·‖Δⱼ‖)).\n"
            "  3. PCA-space angle (--pca-k K): dual PCA on the double-centred Gram\n"
            "     matrix; angle between K-dim score vectors.  Variance explained is\n"
            "     logged to stderr to justify the choice of K."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--pca-k",
        type=int,
        default=None,
        metavar="K",
        help=(
            "If set (K>=1), compute pairwise angles in R^K using dual PCA on the "
            "centered Gram matrix of weight-update inner products within each pretrain "
            "group.  Cumulative variance explained is printed to stderr so the choice "
            "of K is transparent.  Requires numpy.  Adds columns cosine_similarity_pca, "
            "angle_degrees_pca, pca_variance_explained."
        ),
    )
    p.add_argument(
        "--layer-categories",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated layer categories to include (default: all layers).  "
            "Available: attention_qkv, attention_out, mlp, embed, lm_head, norm, other.  "
            "Example: --layer-categories attention_qkv,mlp"
        ),
    )
    args = p.parse_args(argv)
 
    if args.pca_k is not None and args.pca_k < 1:
        print("--pca-k must be >= 1 when set.", file=sys.stderr)
        return 1
 
    # Parse layer category filter
    layer_categories: Optional[List[str]] = None
    if args.layer_categories is not None:
        layer_categories = [c.strip() for c in args.layer_categories.split(",") if c.strip()]
        print(f"[config] Layer filter: {layer_categories}", file=sys.stderr)
 
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
            f"(spectral device: {spectral_device}, "
            f"layer filter: {layer_categories or 'all'})...",
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
            layer_categories=layer_categories,
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
        "layer_filter",
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
        "layer_filter",
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
                "pca_variance_explained",
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