"""Base orchestrator for native ComfyUI (core) DualKSampler."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import comfy.model_management
import nodes
import torch

from dual_ksampler.shared import alignment as core_alignment
from dual_ksampler.shared import logging as core_logging
from dual_ksampler.shared import notifications as core_notifications
from dual_ksampler.shared import models as core_models

logger = logging.getLogger("dual_ksampler.ksampler.base")
bare_logger = logging.getLogger("dual_ksampler.separator")


class DualKSamplerBase:
    """Shared helpers for all DualKSampler (KSampler) node variants."""

    # ComfyUI required attributes
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    RETURN_TYPES = ("LATENT", "SIGMAS")
    RETURN_NAMES = ("LATENT", "sigmas")
    FUNCTION = "sample"
    CATEGORY = "DualKSampler/sampling"

    @classmethod
    def _get_base_input_types(cls) -> Dict[str, Any]:
        return {
            "base_model": ("MODEL", {"tooltip": "Stage 1 model (base / high-noise)."}),
            "lightning_model": (
                "MODEL",
                {"tooltip": "Stage 2 model (Lightning / low-noise)."},
            ),
            "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning."}),
            "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning."}),
            "latent_image": ("LATENT", {"tooltip": "Latent image to denoise."}),
            "seed": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "The random seed used for creating the noise.",
                },
            ),
            "sigma_shift": (
@@ -83,56 +83,61 @@ class DualKSamplerBase:
                {
                    "default": 8,
                    "min": 2,
                    "max": 100,
                    "tooltip": "Total steps for lightning schedule.",
                },
            ),
        }

    @classmethod
    def _calculate_perfect_alignment(
        cls, base_quality_threshold: int, lightning_start: int, lightning_steps: int
    ) -> Tuple[int, int, str]:
        return core_alignment.calculate_perfect_alignment(
            base_quality_threshold, lightning_start, lightning_steps
        )

    def _format_stage_range(self, start: int, end: int, total: int) -> str:
        return core_logging.format_stage_range(start, end, total)

    def _send_dry_run_notification(
        self,
        stage1_info: str,
        stage2_info: str,
        base_calculation_info: str = "",
        switch_info: str = "",
    ) -> None:
        lines = []
        if base_calculation_info:
            lines.append("Calculations:")
            lines.append(f"• {core_logging.format_base_calculation_compact(base_calculation_info)}")
            lines.append("")
        if switch_info:
            lines.append("Switching:")
            lines.append(f"• {switch_info}")
            lines.append("")
        lines.extend(["Stage Configuration:", f"• {stage1_info}", f"• {stage2_info}"])
        core_notifications.send_dry_run_notification("\n".join(lines))

    def _run_sampling_stage(
        self,
        model: Any,
        positive: Any,
        negative: Any,
        latent: Dict[str, torch.Tensor],
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        start_at_step: int,
        end_at_step: int,
        add_noise: bool,
        return_with_leftover_noise: bool,
        dry_run: bool = False,
        stage_name: str = "Sampler",
        stage_info: str | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        if start_at_step >= end_at_step:
            raise ValueError(
                f"{stage_name}: start_at_step ({start_at_step}) >= end_at_step ({end_at_step})."