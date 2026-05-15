"""Map safetensor weight keys to coarse groups and optional layer / expert indices."""

from __future__ import annotations

import re
from dataclasses import dataclass

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")


@dataclass(frozen=True)
class TensorClassification:
    """Human-readable buckets for plotting and filtering."""

    matrix_type: str
    matrix_group: str
    tensor_subtype: str
    layer: int | None
    expert_index: int | None


def parse_layer_expert(key: str) -> tuple[int | None, int | None]:
    layer = int(m.group(1)) if (m := _LAYER_RE.search(key)) else None
    expert = int(m.group(1)) if (m := _EXPERT_RE.search(key)) else None
    return layer, expert


def classify_tensor(key: str) -> TensorClassification | None:
    """
    Return classification for weight keys, or None if the key should be skipped
    for matrix-level analysis (unknown layout).
    """
    layer, expert = parse_layer_expert(key)

    if "model.embed_tokens.weight" == key or key.endswith("embed_tokens.weight"):
        return TensorClassification(
            matrix_type="embedding",
            matrix_group="embedding",
            tensor_subtype="",
            layer=None,
            expert_index=None,
        )

    if key.endswith("lm_head.weight"):
        return TensorClassification(
            matrix_type="lm_head",
            matrix_group="lm_head",
            tensor_subtype="",
            layer=None,
            expert_index=None,
        )

    if ".self_attn." in key:
        if "q_proj" in key and key.endswith("q_proj.weight"):
            return TensorClassification(
                matrix_type="q_proj",
                matrix_group="q_proj",
                tensor_subtype="",
                layer=layer,
                expert_index=None,
            )
        if "kv_a_proj" in key and key.endswith(".weight"):
            return TensorClassification(
                matrix_type="kv_a_proj",
                matrix_group="kv_proj",
                tensor_subtype="kv_a",
                layer=layer,
                expert_index=None,
            )
        if "kv_b_proj" in key and key.endswith("kv_b_proj.weight"):
            return TensorClassification(
                matrix_type="kv_b_proj",
                matrix_group="kv_proj",
                tensor_subtype="kv_b",
                layer=layer,
                expert_index=None,
            )
        if "o_proj" in key and key.endswith("o_proj.weight"):
            return TensorClassification(
                matrix_type="o_proj",
                matrix_group="o_proj",
                tensor_subtype="",
                layer=layer,
                expert_index=None,
            )
        return None

    if ".mlp." in key:
        if key.endswith("mlp.gate.weight"):
            return TensorClassification(
                matrix_type="moe_router",
                matrix_group="moe_router",
                tensor_subtype="",
                layer=layer,
                expert_index=None,
            )
        if "experts" in key:
            if "gate_proj" in key and key.endswith("gate_proj.weight"):
                return TensorClassification(
                    matrix_type="moe_gate_proj",
                    matrix_group="moe_ffn",
                    tensor_subtype="gate_proj",
                    layer=layer,
                    expert_index=expert,
                )
            if "up_proj" in key and key.endswith("up_proj.weight"):
                return TensorClassification(
                    matrix_type="moe_up_proj",
                    matrix_group="moe_ffn",
                    tensor_subtype="up_proj",
                    layer=layer,
                    expert_index=expert,
                )
            if "down_proj" in key and key.endswith("down_proj.weight"):
                return TensorClassification(
                    matrix_type="moe_down_proj",
                    matrix_group="moe_ffn",
                    tensor_subtype="down_proj",
                    layer=layer,
                    expert_index=expert,
                )
            return None
        if "gate_proj" in key and key.endswith("gate_proj.weight"):
            return TensorClassification(
                matrix_type="shared_mlp_gate_proj",
                matrix_group="mlp_shared",
                tensor_subtype="gate_proj",
                layer=layer,
                expert_index=None,
            )
        if "up_proj" in key and key.endswith("up_proj.weight"):
            return TensorClassification(
                matrix_type="shared_mlp_up_proj",
                matrix_group="mlp_shared",
                tensor_subtype="up_proj",
                layer=layer,
                expert_index=None,
            )
        if "down_proj" in key and key.endswith("down_proj.weight"):
            return TensorClassification(
                matrix_type="shared_mlp_down_proj",
                matrix_group="mlp_shared",
                tensor_subtype="down_proj",
                layer=layer,
                expert_index=None,
            )
        return None

    return None
