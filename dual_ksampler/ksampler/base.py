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
                "FLOAT",
                {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Sigma adjustment applied via ModelSamplingSD3 for model sampling.",
                },
            ),
            "base_cfg": (
                "FLOAT",
                {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "CFG scale for Stage 1.",
                },
            ),
            "lightning_cfg": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "CFG scale for Stage 2.",
                },
            ),
            "lightning_steps": (
                "INT",
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
    ) -> None:
        lines = []
        if base_calculation_info:
            lines.append("Calculations:")
            lines.append(f"• {core_logging.format_base_calculation_compact(base_calculation_info)}")
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
            )

        bare_logger.info("")
        if stage_info:
            logger.info("%s: %s", stage_name, stage_info)

        if dry_run:
            return (latent,)

        advanced_sampler = nodes.KSamplerAdvanced()
        add_noise_mode = "enable" if add_noise else "disable"
        return_noise_mode = "enable" if return_with_leftover_noise else "disable"

        try:
            result = advanced_sampler.sample(
                model=model,
                add_noise=add_noise_mode,
                noise_seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent,
                start_at_step=start_at_step,
                end_at_step=end_at_step,
                return_with_leftover_noise=return_noise_mode,
                denoise=1.0,
            )
        except comfy.model_management.InterruptProcessingException:
            raise
        except Exception as exc:
            msg = str(exc).strip()
            if msg:
                raise RuntimeError(f"{stage_name}: sampling failed - {type(exc).__name__}: {msg}") from exc
            raise RuntimeError(f"{stage_name}: sampling failed - {type(exc).__name__}") from exc

        return result

    def _patch_models_for_sampling(
        self, base_model: Any, lightning_model: Any, sigma_shift: float
    ) -> Tuple[Any, Any]:
        return core_models.patch_models_with_sigma_shift(base_model, lightning_model, sigma_shift)

    # -------------------------
    # Minimal validation helpers
    # -------------------------
    def _validate_basic_parameters(
        self, lightning_steps: int, lightning_start: int, base_steps: int, base_quality_threshold: int
    ) -> None:
        if base_quality_threshold < 1:
            raise ValueError("base_quality_threshold must be >= 1")
        if lightning_steps < 2:
            raise ValueError("lightning_steps must be >= 2")
        if not (0 <= lightning_start < lightning_steps):
            raise ValueError("lightning_start must be between 0 and lightning_steps-1")
        if base_steps < -1:
            raise ValueError("base_steps must be -1 (auto) or >= 0")
