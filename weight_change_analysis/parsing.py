"""Parse finetune run folder names into structured metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RunMeta:
    folder_name: str
    pretrain_optimizer: str  # "adam" | "muon"
    finetune_type: str  # "lora" | "full"
    # Primary fine-tune optimizer label (e.g. adamw, muon), before _swa / _4-28
    finetune_optimizer: str
    # Remainder of the name after the optimizer (e.g. "4-28", "swa_4-28", "")
    name_suffix: str


def parse_run_folder(name: str) -> RunMeta:
    """
    Examples:
        adam_ckpt_driving_adamw -> pre adam, lora, ft adamw
        adam_ckpt_driving_fullft_adamw_4-28 -> pre adam, full, ft adamw, suffix 4-28
        muon_ckpt_driving_fullft_muon_swa_4-28 -> pre muon, full, ft muon, suffix swa_4-28
    """
    m = re.match(r"^(adam|muon)_ckpt_driving_(.+)$", name)
    if not m:
        raise ValueError(f"Unrecognized run folder name: {name!r}")
    pre, rest = m.group(1), m.group(2)
    if rest.startswith("fullft_"):
        ft_type = "full"
        body = rest[len("fullft_") :]
    else:
        ft_type = "lora"
        body = rest
    # body: "adamw", "adamw_4-28", "muon", "muon_swa_4-28"
    if body.startswith("adamw"):
        ft_opt = "adamw"
        suffix = body[len("adamw") :].lstrip("_")
    elif body.startswith("muon"):
        ft_opt = "muon"
        suffix = body[len("muon") :].lstrip("_")
    else:
        parts = body.split("_")
        ft_opt = parts[0]
        suffix = "_".join(parts[1:]) if len(parts) > 1 else ""
    return RunMeta(
        folder_name=name,
        pretrain_optimizer=pre,
        finetune_type=ft_type,
        finetune_optimizer=ft_opt,
        name_suffix=suffix,
    )
