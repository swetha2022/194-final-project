"""Default filesystem paths for pretrained bases and finetune roots."""

from pathlib import Path

# Override via CLI flags when analyzing runs stored elsewhere.
DEFAULT_FT_ROOT = Path("/scratch/celine/ft_out")
DEFAULT_WEIGHT_ROOT = Path("/scratch/celine/moonlight_weights")

PRETRAINED_BASE_BY_OPT = {
    "adam": DEFAULT_WEIGHT_ROOT / "Moonlight_adam_hf_step_42000",
    "muon": DEFAULT_WEIGHT_ROOT / "Moonlight_hf_step_42000",
}
