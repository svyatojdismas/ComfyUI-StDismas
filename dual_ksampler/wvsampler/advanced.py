"""Advanced dual-stage WanVideo sampler (AdvancedAlt / static UI).

Two-stage pipeline:
  Stage 1: base model denoising (high-noise / quality)
  Stage 2: lightning model continuation (typically low-noise Lightning/LightX2V)

Derived from ComfyUI-TripleKSampler's WanVideo implementation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict

from dual_ksampler.shared import alignment as core_alignment
from dual_ksampler.shared import logging as core_logging
from dual_ksampler.shared import notifications as core_notifications
from .utils import get_wanvideo_components

try:
    from comfy.model_management import InterruptProcessingException
except Exception:  # pragma: no cover
    class InterruptProcessingException(Exception):
        pass

logger = logging.getLogger("dual_ksampler.wvsampler")

_DEFAULT_BASE_QUALITY_THRESHOLD = 20
_DEFAULT_BOUNDARY_T2V = 0.875
_DEFAULT_BOUNDARY_I2V = 0.900

START_MODES = [
    "Manual step",
    "T2V boundary",
    "I2V boundary",
    "Manual boundary",
]


class DualWVSamplerAdvancedAlt:
    """Dual-stage WanVideo sampler (static UI / AdvancedAlt)."""

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("samples", "denoised_samples")
    FUNCTION = "sample"
    CATEGORY = "DualKSampler/wanvideo"
    DESCRIPTION = (
        "Dual-stage WanVideo sampler: Stage 1 uses a base model, then Stage 2 continues "
        "with a Lightning model. Designed for setups without separate high/low-noise "
        "Lightning experts (e.g., Wan Animate)."
    )

    def __init__(self):
        self.wanvideo_sampler = None
        self.get_scheduler = None

    def _ensure_wanvideo(self) -> None:
        """Lazy-load WanVideoSampler and get_scheduler."""
        if self.wanvideo_sampler is not None and self.get_scheduler is not None:
            return
        sampler_class, _scheduler_list, get_scheduler_func = get_wanvideo_components()
        self.wanvideo_sampler = sampler_class()
        self.get_scheduler = get_scheduler_func

    def _compute_wanvideo_boundary_step(
        self,
        model: Any,
        scheduler: str,
        steps: int,
        shift: float,
        boundary: float,
    ) -> int:
        """Compute the first step index where WanVideo timestep drops below boundary."""
        import torch

        if self.get_scheduler is None:
            return max(0, min(steps - 1, math.ceil(steps / 2)))

        try:
            transformer_dim = (
                model.model.diffusion_model.dim
                if hasattr(model, "model") and hasattr(model.model, "diffusion_model")
                else 5120
            )
        except Exception:
            transformer_dim = 5120

        device = torch.device("cpu")
        try:
            sample_scheduler, _timesteps, _, _ = self.get_scheduler(
                scheduler,
                steps,
                start_step=0,
                end_step=-1,
                shift=shift,
                device=device,
                transformer_dim=transformer_dim,
            )
            sigmas = sample_scheduler.sigmas
        except Exception as e:
            logger.warning("Failed to compute boundary step via WanVideo scheduler '%s': %s", scheduler, e)
            return max(0, min(steps - 1, math.ceil(steps / 2)))

        timesteps_normalized = []
        for sigma in sigmas[:-1]:
            try:
                timesteps_normalized.append(float(sigma.item()))
            except Exception:
                timesteps_normalized.append(float(sigma))

        switching_step = steps - 1
        for i, t in enumerate(timesteps_normalized):
            if t < float(boundary):
                switching_step = i
                break

        return int(max(0, min(steps - 1, switching_step)))

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Static UI inputs.

        We reuse WanVideoSampler's input specs when available (via NODE_CLASS_MAPPINGS),
        otherwise fall back to safe defaults so ComfyUI can still start.
        """
        from nodes import NODE_CLASS_MAPPINGS

        if "WanVideoSampler" in NODE_CLASS_MAPPINGS:
            original_inputs = NODE_CLASS_MAPPINGS["WanVideoSampler"].INPUT_TYPES()
            scheduler_spec = original_inputs.get("required", {}).get("scheduler", [])
            scheduler_list = scheduler_spec[0] if isinstance(scheduler_spec, tuple) else scheduler_spec
        else:
            scheduler_list = [
                "unipc",
                "unipc/beta",
                "dpm++",
                "dpm++/beta",
                "dpm++_sde",
                "dpm++_sde/beta",
                "euler",
                "euler/beta",
                "longcat_distill_euler",
                "deis",
                "lcm",
                "lcm/beta",
                "res_multistep",
                "flowmatch_causvid",
                "flowmatch_distill",
                "flowmatch_pusa",
                "multitalk",
                "sa_ode_stable",
                "rcm",
            ]
            original_inputs = {
                "required": {
                    "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                    "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                    "scheduler": (scheduler_list, {"default": "unipc"}),
                    "force_offload": ("BOOLEAN", {"default": True}),
                    "riflex_freq_index": ("INT", {"default": 0}),
                    "batched_cfg": ("BOOLEAN", {"default": False}),
                    "rope_function": (["default", "comfy", "comfy_chunked"],),
                },
                "optional": {},
            }

        required: Dict[str, Any] = {}
        optional: Dict[str, Any] = {}

        required["base_model"] = (
            "WANVIDEOMODEL",
            {"tooltip": "Stage 1 model (base/high-noise)."},
        )
        required["lightning_model"] = (
            "WANVIDEOMODEL",
            {"tooltip": "Stage 2 model (Lightning/LightX2V)."},
        )

        # WanVideo core inputs
        required["image_embeds"] = original_inputs["required"]["image_embeds"]

        # seed
        seed_spec = original_inputs["required"].get("seed", ("INT", {"default": 0}))
        if isinstance(seed_spec, tuple) and len(seed_spec) == 2:
            seed_spec = (
                seed_spec[0],
                {
                    **seed_spec[1],
                    "control_after_generate": True,
                    "tooltip": "The random seed used for creating the noise.",
                },
            )
        required["seed"] = seed_spec

        # sigma_shift (renamed from WanVideo 'shift')
        shift_spec = original_inputs["required"].get(
            "shift",
            ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01}),
        )
        if isinstance(shift_spec, tuple) and len(shift_spec) == 2:
            shift_spec = (
                shift_spec[0],
                {
                    **shift_spec[1],
                    "tooltip": "Sigma adjustment applied to all models for WanVideo sampling.",
                },
            )
        required["sigma_shift"] = shift_spec

        # Base stage params
        required["base_quality_threshold"] = (
            "INT",
            {
                "default": _DEFAULT_BASE_QUALITY_THRESHOLD,
                "min": 1,
                "max": 100,
                "step": 1,
                "tooltip": "Minimum total steps for base_steps auto-calculation (only if base_steps=-1).",
            },
        )
        required["base_steps"] = (
            "INT",
            {
                "default": -1,
                "min": -1,
                "max": 100,
                "tooltip": "Stage 1 end step (base model). Use -1 for auto-calculation.",
            },
        )
        required["base_cfg"] = (
            "FLOAT",
            {
                "default": 3.5,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "tooltip": "CFG scale for Stage 1 (base model).",
            },
        )
        required["base_scheduler"] = (
            scheduler_list if scheduler_list else ["unipc"],
            {"default": "unipc", "tooltip": "Scheduler for Stage 1 (base model)."},
        )

        # Lightning stage params
        required["lightning_start_mode"] = (
            START_MODES,
            {"default": "Manual step", "tooltip": "How to determine lightning_start."},
        )
        required["lightning_start"] = (
            "INT",
            {
                "default": 1,
                "min": 0,
                "max": 99,
                "tooltip": "Stage 2 start step within lightning schedule (used for Manual step mode).",
            },
        )
        required["start_boundary"] = (
            "FLOAT",
            {
                "default": _DEFAULT_BOUNDARY_T2V,
                "min": 0.0,
                "max": 1.0,
                "step": 0.001,
                "tooltip": "Boundary (0..1) used for boundary modes. Use 0.875 (T2V) / 0.9 (I2V) as typical defaults.",
            },
        )
        required["lightning_steps"] = (
            "INT",
            {"default": 8, "min": 2, "max": 100, "tooltip": "Total steps for Stage 2."},
        )
        required["lightning_cfg"] = (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "tooltip": "CFG scale for Stage 2 (Lightning). 1.0 is typical for 4-step Lightning LoRA.",
            },
        )
        required["lightning_scheduler"] = (
            scheduler_list if scheduler_list else ["unipc"],
            {"default": "unipc", "tooltip": "Scheduler for Stage 2 (Lightning model)."},
        )

        # WanVideo shared required params
        for key in ["force_offload", "riflex_freq_index", "batched_cfg", "rope_function"]:
            if key in original_inputs.get("required", {}):
                required[key] = original_inputs["required"][key]

        # Copy WanVideo optional params through (except 'model' and 'samples' if present)
        for k, v in original_inputs.get("optional", {}).items():
            if k in {"model", "samples"}:
                continue
            optional[k] = v

        # DualKSampler extra optional
        optional["dry_run"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "If enabled, validates step math and shows a toast with stage ranges; does not sample.",
            },
        )

        return {"required": required, "optional": optional}

    def sample(
        self,
        base_model: Any,
        lightning_model: Any,
        image_embeds: Any,
        seed: int,
        sigma_shift: float,
        base_quality_threshold: int,
        base_steps: int,
        base_cfg: float,
        base_scheduler: str,
        start_mode: str,
        lightning_start: int,
        start_boundary: float,
        lightning_steps: int,
        lightning_cfg: float,
        lightning_scheduler: str,
        force_offload: bool,
        riflex_freq_index: int,
        batched_cfg: bool,
        rope_function: str,
        dry_run: bool = False,
        **kwargs,
    ):
        """Run dual-stage WanVideo sampling."""
        # Lazy-load WanVideo components at runtime (ComfyUI load-order safe)
        self._ensure_wanvideo()

        # ---- validate basics ----
        lightning_steps = int(max(2, lightning_steps))
        lightning_start = int(max(0, min(lightning_steps - 1, lightning_start)))

        # ---- resolve lightning_start from strategy (optional) ----
        effective_start = lightning_start
        if start_mode == "T2V boundary":
            effective_start = self._compute_wanvideo_boundary_step(
                model=lightning_model,
                scheduler=lightning_scheduler,
                steps=lightning_steps,
                shift=float(sigma_shift),
                boundary=_DEFAULT_BOUNDARY_T2V,
            )
        elif start_mode == "I2V boundary":
            effective_start = self._compute_wanvideo_boundary_step(
                model=lightning_model,
                scheduler=lightning_scheduler,
                steps=lightning_steps,
                shift=float(sigma_shift),
                boundary=_DEFAULT_BOUNDARY_I2V,
            )
        elif start_mode == "Manual boundary":
            boundary = float(max(0.0, min(1.0, start_boundary)))
            effective_start = self._compute_wanvideo_boundary_step(
                model=lightning_model,
                scheduler=lightning_scheduler,
                steps=lightning_steps,
                shift=float(sigma_shift),
                boundary=boundary,
            )

        effective_start = int(max(0, min(lightning_steps - 1, effective_start)))

        # ---- compute base schedule alignment ----
        if effective_start == 0:
            base_steps_resolved = 0
            total_base_steps = 0
            base_calc_info = "Stage 1 skipped (lightning_start=0)"
        elif base_steps == -1:
            base_steps_resolved, total_base_steps, base_calc_info = (
                core_alignment.calculate_perfect_alignment(
                    base_quality_threshold, effective_start, lightning_steps
                )
            )
        else:
            base_steps_resolved = int(max(0, base_steps))
            total_base_steps, base_calc_info = core_alignment.calculate_manual_base_steps_alignment(
                base_steps_resolved, effective_start, lightning_steps
            )

        # ---- overlap warning (manual base_steps can overshoot) ----
        overlap_pct = 0.0
        if total_base_steps > 0:
            stage1_end = base_steps_resolved / float(total_base_steps)
            stage2_start = effective_start / float(lightning_steps)
            if stage1_end > stage2_start:
                overlap_pct = (stage1_end - stage2_start) * 100.0
                if base_steps != -1:
                    core_notifications.send_overlap_warning(overlap_pct)

        # ---- dry run ----
        if dry_run:
            lines = []
            lines.append(f"Lightning start mode: {start_mode} → start_step={effective_start} of {lightning_steps}")
            if start_mode == "Manual boundary":
                lines.append(f"  boundary={float(start_boundary):.4f}")
            elif start_mode == "T2V boundary":
                lines.append(f"  boundary={_DEFAULT_BOUNDARY_T2V:.4f}")
            elif start_mode == "I2V boundary":
                lines.append(f"  boundary={_DEFAULT_BOUNDARY_I2V:.4f}")
            lines.append(core_logging.format_base_calculation_compact(base_calc_info))
            if total_base_steps > 0:
                lines.append(
                    "Stage 1 (base): "
                    + core_logging.format_stage_range(0, base_steps_resolved, total_base_steps)
                )
            else:
                lines.append("Stage 1 (base): skipped")
            lines.append(
                "Stage 2 (lightning): "
                + core_logging.format_stage_range(effective_start, lightning_steps, lightning_steps)
            )
            lines.append(f"Stage 1 scheduler: {base_scheduler}")
            lines.append(f"Stage 2 scheduler: {lightning_scheduler}")
            if overlap_pct > 0:
                lines.append(f"Overlap: {overlap_pct:.1f}%")

            core_notifications.send_dry_run_notification("\n".join(lines))
            raise InterruptProcessingException("DualKSampler dry_run")

        # ---- shared params (WanVideoSampler.process) ----
        shared_params: Dict[str, Any] = {
            "image_embeds": image_embeds,
            "seed": seed,
            "riflex_freq_index": riflex_freq_index,
            "batched_cfg": batched_cfg,
            "rope_function": rope_function,
            **kwargs,
        }

        # Stage 1
        stage1_samples = None
        if base_steps_resolved > 0 and total_base_steps > 0:
            logger.info(
                "DualKSampler Stage 1 (base): %s",
                core_logging.format_stage_range(0, base_steps_resolved, total_base_steps),
            )
            stage1_samples, _ = self.wanvideo_sampler.process(
                model=base_model,
                steps=int(total_base_steps),
                cfg=float(base_cfg),
                shift=float(sigma_shift),
                scheduler=base_scheduler,
                force_offload=force_offload,
                start_step=0,
                end_step=int(base_steps_resolved),
                add_noise_to_samples=True,
                **shared_params,
            )

        # Stage 2
        logger.info(
            "DualKSampler Stage 2 (lightning): %s",
            core_logging.format_stage_range(effective_start, lightning_steps, lightning_steps),
        )
        stage2_add_noise = stage1_samples is None
        stage2_samples, stage2_denoised = self.wanvideo_sampler.process(
            model=lightning_model,
            steps=int(lightning_steps),
            cfg=float(lightning_cfg),
            shift=float(sigma_shift),
            scheduler=lightning_scheduler,
            force_offload=force_offload,
            samples=stage1_samples,
            start_step=int(effective_start),
            end_step=-1,
            add_noise_to_samples=stage2_add_noise,
            **shared_params,
        )

        return (stage2_samples, stage2_denoised)
