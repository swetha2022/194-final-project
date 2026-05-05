"""Load Hugging Face-style sharded safetensors without loading the full model at once."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Set

import torch
from safetensors.torch import load_file


@dataclass
class ShardLayout:
    model_dir: Path
    weight_map: Dict[str, str]  # tensor_name -> shard_filename


def read_shard_layout(model_dir: os.PathLike) -> ShardLayout:
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)
        return ShardLayout(model_dir=model_dir, weight_map=idx["weight_map"])
    # Single-file fallback
    singles = sorted(model_dir.glob("model*.safetensors"))
    singles = [p for p in singles if "index" not in p.name.lower()]
    if len(singles) == 1:
        from safetensors import safe_open

        with safe_open(singles[0], framework="pt", device="cpu") as sf:
            keys = list(sf.keys())
        return ShardLayout(
            model_dir=model_dir,
            weight_map={k: singles[0].name for k in keys},
        )
    raise FileNotFoundError(f"No model.safetensors.index.json or single model*.safetensors in {model_dir}")


def tensor_keys(layout: ShardLayout) -> Set[str]:
    return set(layout.weight_map.keys())


def keys_sorted_by_shard(layout: ShardLayout) -> List[str]:
    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    for k, shard in layout.weight_map.items():
        shard_to_keys[shard].append(k)
    out: List[str] = []
    for shard in sorted(shard_to_keys.keys()):
        out.extend(sorted(shard_to_keys[shard]))
    return out


class ShardedSafetensors:
    """Lazy per-shard cache for sequential access."""

    def __init__(self, layout: ShardLayout):
        self.layout = layout
        self._cache_shard: str | None = None
        self._cache: Dict[str, torch.Tensor] | None = None

    def _ensure_shard(self, shard: str) -> None:
        if self._cache_shard == shard and self._cache is not None:
            return
        path = self.layout.model_dir / shard
        self._cache = load_file(str(path))
        self._cache_shard = shard

    def get_tensor(self, name: str) -> torch.Tensor:
        shard = self.layout.weight_map[name]
        self._ensure_shard(shard)
        assert self._cache is not None
        return self._cache[name]

    def drop_cache(self) -> None:
        self._cache = None
        self._cache_shard = None


def iter_shared_keys(layouts: List[ShardLayout]) -> Iterator[str]:
    if not layouts:
        return
    common: Set[str] = tensor_keys(layouts[0])
    for ly in layouts[1:]:
        common &= tensor_keys(ly)
    # Exclude non-float buffers if any (should be rare)
    for k in sorted(common):
        yield k


def find_latest_step_dir(run_root: Path) -> Path:
    steps = sorted(run_root.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if not steps:
        raise FileNotFoundError(f"No step_* directory under {run_root}")
    return steps[-1]


def resolve_weights_dir(step_dir: Path) -> Path:
    """LoRA runs store merged full weights; full finetune uses consolidated shards."""
    model_root = step_dir / "policy" / "weights" / "model"
    consolidated = model_root / "consolidated"
    merged = model_root / "merged"
    if consolidated.is_dir():
        return consolidated
    if merged.is_dir():
        return merged
    raise FileNotFoundError(
        f"Expected policy/weights/model/{{consolidated|merged}} under {step_dir}"
    )
