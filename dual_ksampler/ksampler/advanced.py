"""Advanced native (ComfyUI core) DualKSampler nodes.

Implements a two-stage cascade using ComfyUI's ``KSamplerAdvanced``.
Derived from the native KSampler implementation in ComfyUI-TripleKSampler.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import comfy.model_management
import comfy.samplers

from dual_ksampler.shared import alignment as core_alignment
from dual_ksampler.shared import logging as core_logging
from dual_ksampler.shared import models as core_models
from dual_ksampler.shared import notifications as core_notifications

from .base import DualKSamplerBase

logger = logging.getLogger("dual_ksampler.ksampler.advanced")

DEFAULT_BOUNDARY_T2V = 0.875
DEFAULT_BOUNDARY_I2V = 0.9
SWITCH_STRATEGIES = [
    "50% of steps",
    "Manual switch step",
    "T2V boundary",
    "I2V boundary",
    "Manual boundary",
    "T2V boundary (refined)",
    "I2V boundary (refined)",
    "Manual boundary (refined)",
]


class DualKSamplerAdvancedAlt(DualKSamplerBase):
    """Dual-stage sampler with fully exposed controls (static UI)."""

    DESCRIPTION = (
        "Dual-stage core KSampler: Stage 1 uses a base model, then Stage 2 continues "
        "with a Lightning model. Intended for Wan Animate / setups without separate "
        "Lightning high/low experts."
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # IMPORTANT: ComfyUI renders widgets in the insertion order of the
        # ``required`` dict. We match TripleKSampler's layout:
        # sigma_shift → base_quality_threshold → base_steps → base_cfg → base sampler/scheduler →
        # lightning_start → lightning_steps → lightning_cfg → lightning sampler/scheduler → dry_run.
        base = cls._get_base_input_types()

        required: Dict[str, Any] = {
            # models + conditioning + latent
            "base_model": base["base_model"],
            "lightning_model": base["lightning_model"],
            "positive": base["positive"],
            "negative": base["negative"],
            "latent_image": base["latent_image"],
            # seed
            "seed": base["seed"],
            # sigma
            "sigma_shift": base["sigma_shift"],
            # base stage (alignment then cfg)
            "base_quality_threshold": (
                "INT",
                {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Minimum total steps used when base_steps is auto (-1).",
                },
            ),
            "base_steps": (
                "INT",
                {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "tooltip": "Stage 1 end step (base). -1 = auto (perfect alignment).",
                },
            ),
            "base_cfg": base["base_cfg"],
            "base_sampler": (
                comfy.samplers.KSampler.SAMPLERS,
                {"default": "euler", "tooltip": "Sampler for Stage 1."},
            ),
            "base_scheduler": (
                comfy.samplers.KSampler.SCHEDULERS,
                {"default": "simple", "tooltip": "Scheduler for Stage 1."},
            ),
            # lightning stage
            "lightning_start": (
                "INT",
                {
                    "default": 1,
                    "min": 0,
                    "max": 99,
                    "tooltip": "Stage 2 start step inside the lightning schedule (0 skips Stage 1).",
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
            "lightning_cfg": base["lightning_cfg"],
            "lightning_sampler": (
                comfy.samplers.KSampler.SAMPLERS,
                {"default": "euler", "tooltip": "Sampler for Stage 2."},
            ),
            "lightning_scheduler": (
                comfy.samplers.KSampler.SCHEDULERS,
                {"default": "simple", "tooltip": "Scheduler for Stage 2."},
            ),
            "switch_strategy": (
                SWITCH_STRATEGIES,
                {
                    "default": "Manual switch step",
                    "tooltip": "How to pick the model switch point. Refined modes auto-tune sigma_shift for boundary alignment.",
                },
            ),
            "switch_step": (
                "INT",
                {
                    "default": -1,
                    "min": -1,
                    "max": 99,
                    "tooltip": "Manual step to switch models. Only used when switch_strategy is 'Manual switch step'. "
                    "Use -1 to fall back to lightning_start.",
                },
            ),
            "switch_boundary": (
                "FLOAT",
                {
                    "default": DEFAULT_BOUNDARY_T2V,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Boundary used for boundary-based strategies.",
                },
            ),
            "dry_run": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Validate & show resolved stage ranges (toast). Does not sample.",
                },
            ),
        }

        return {"required": required}

    def sample(
        self,
        base_model: Any,
        lightning_model: Any,
        positive: Any,
        negative: Any,
        latent_image: Dict[str, Any],
        seed: int,
        sigma_shift: float,
        base_cfg: float,
        lightning_cfg: float,
        lightning_steps: int,
        base_quality_threshold: int,
        base_steps: int,
        base_sampler: str,
        base_scheduler: str,
        lightning_start: int,
        lightning_sampler: str,
        lightning_scheduler: str,
        switch_strategy: str,
        switch_step: int,
        switch_boundary: float,
        dry_run: bool = False,
    ):
        # Patch models with sigma shift (needed for boundary strategies)
        patched_base, patched_lightning = self._patch_models_for_sampling(
            base_model, lightning_model, sigma_shift
        )

        # Resolve switch step from strategy (used as effective lightning_start)
        switch_step_resolved, switch_info = self._resolve_switch_strategy(
            switch_strategy=switch_strategy,
            switch_step=switch_step,
            switch_boundary=switch_boundary,
            lightning_steps=lightning_steps,
            lightning_start=lightning_start,
            lightning_model=patched_lightning,
            lightning_scheduler=lightning_scheduler,
        )

        sigma_shift_final = sigma_shift
        if self._is_refined_strategy(switch_strategy):
            target_boundary = self._get_boundary_for_strategy(
                switch_strategy, switch_boundary
            )
            refined_shift, refine_msg = core_models.calculate_perfect_shift_for_step(
                model=lightning_model,
                scheduler=lightning_scheduler,
                total_steps=lightning_steps,
                target_step=switch_step_resolved,
                target_sigma=target_boundary,
                initial_shift=sigma_shift,
            )
            if refined_shift != sigma_shift:
                sigma_shift_final = refined_shift
                switch_info += f" [Refined shift: {sigma_shift:.2f}→{refined_shift:.2f}]"
                logger.info("Sigma shift refinement: %s", refine_msg)
            elif refine_msg != "already aligned":
                logger.info("Sigma shift refinement: %s", refine_msg)

        if sigma_shift_final != sigma_shift:
            patched_base, patched_lightning = self._patch_models_for_sampling(
                base_model, lightning_model, sigma_shift_final
            )

        sigmas_output = core_models.get_sigmas_tensor(
            patched_lightning, lightning_scheduler, lightning_steps
        )

        effective_start = int(max(0, min(lightning_steps - 1, switch_step_resolved)))

        # Validate user inputs
        self._validate_basic_parameters(
            lightning_steps=lightning_steps,
            lightning_start=effective_start,
            base_steps=base_steps,
            base_quality_threshold=base_quality_threshold,
        )

        # Resolve base_steps / total_base_steps
        base_calc_info = ""
        if effective_start == 0:
            resolved_base_steps = 0
            total_base_steps = 0
            base_calc_info = "Lightning-only mode (base stage skipped)"
        else:
            if base_steps == -1:
                resolved_base_steps, total_base_steps, method = self._calculate_perfect_alignment(
                    base_quality_threshold, effective_start, lightning_steps
                )
                base_calc_info = (
                    f"Auto-calculated base_steps = {resolved_base_steps}, "
                    f"total_base_steps = {total_base_steps} ({method})"
                )
            else:
                total_base_steps = core_alignment.calculate_manual_base_steps_alignment(
                    base_steps, effective_start, lightning_steps
                )
                resolved_base_steps = base_steps
                base_calc_info = (
                    f"Auto-calculated total_base_steps = {total_base_steps} "
                    f"for manual base_steps = {resolved_base_steps}"
                )

            # Overlap warning: Stage 1 end > Stage 2 start
            if total_base_steps > 0:
                s1_end = float(resolved_base_steps) / float(total_base_steps)
                s2_start = float(effective_start) / float(lightning_steps)
                if s1_end > s2_start:
                    overlap_pct = (s1_end - s2_start) * 100.0
                    core_notifications.send_overlap_warning(overlap_pct)

        # Stage 1
        if effective_start == 0:
            stage1_toast = "Stage 1 (Base): skipped"
            stage1_latent = latent_image
        else:
            stage1_toast = (
                "Stage 1 (Base): "
                + self._format_stage_range(0, resolved_base_steps, total_base_steps)
            )
            stage1_latent = self._run_sampling_stage(
                model=patched_base,
                positive=positive,
                negative=negative,
                latent=latent_image,
                seed=seed,
                steps=total_base_steps,
                cfg=base_cfg,
                sampler_name=base_sampler,
                scheduler=base_scheduler,
                start_at_step=0,
                end_at_step=resolved_base_steps,
                add_noise=True,
                return_with_leftover_noise=True,
                dry_run=dry_run,
                stage_name="Stage 1",
                stage_info=stage1_toast,
            )[0]

        # Stage 2
        if effective_start >= lightning_steps:
            # Should never happen due to validation, but keep it safe.
            raise ValueError("lightning_start must be < lightning_steps")

        stage2_toast = (
            "Stage 2 (Lightning): "
            + self._format_stage_range(effective_start, lightning_steps, lightning_steps)
        )

        stage2_latent = self._run_sampling_stage(
            model=patched_lightning,
            positive=positive,
            negative=negative,
            latent=stage1_latent,
            seed=seed,
            steps=lightning_steps,
            cfg=lightning_cfg,
            sampler_name=lightning_sampler,
            scheduler=lightning_scheduler,
            start_at_step=effective_start,
            end_at_step=lightning_steps,
            add_noise=(effective_start == 0),
            return_with_leftover_noise=False,
            dry_run=dry_run,
            stage_name="Stage 2",
            stage_info=stage2_toast,
        )[0]

        if dry_run:
            self._send_dry_run_notification(
                stage1_toast,
                stage2_toast,
                base_calc_info,
                switch_info=core_logging.format_switch_info_compact(switch_info),
            )
            # Stop graph execution without treating as error.
            raise comfy.model_management.InterruptProcessingException()

        return (stage2_latent, sigmas_output)

    def _resolve_switch_strategy(
        self,
        switch_strategy: str,
        switch_step: int,
        switch_boundary: float,
        lightning_steps: int,
        lightning_start: int,
        lightning_model: Any,
        lightning_scheduler: str,
    ) -> tuple[int, str]:
        switch_step_resolved = int(max(0, min(lightning_steps - 1, lightning_start)))
        boundary = None
        fallback_step = max(0, min(lightning_steps - 1, lightning_steps // 2))

        if switch_strategy == "50% of steps":
            switch_step_resolved = fallback_step
        elif switch_strategy == "Manual switch step":
            if switch_step != -1:
                switch_step_resolved = int(max(0, min(lightning_steps - 1, switch_step)))
            else:
                switch_step_resolved = int(max(0, min(lightning_steps - 1, lightning_start)))
        else:
            boundary = self._get_boundary_for_strategy(switch_strategy, switch_boundary)
            switch_step_resolved = core_models.calculate_boundary_step(
                model=lightning_model,
                scheduler=lightning_scheduler,
                steps=lightning_steps,
                boundary=boundary,
                fallback_step=fallback_step,
            )

        if boundary is not None:
            switch_info = (
                f"Model switching: {switch_strategy.replace(' (refined)', '')} "
                f"(boundary = {boundary:.3f}) → switch at step {switch_step_resolved} of {lightning_steps}"
            )
        else:
            switch_info = (
                f"Model switching: {switch_strategy} → switch at step {switch_step_resolved} of {lightning_steps}"
            )

        logger.info(switch_info)
        return switch_step_resolved, switch_info

    @staticmethod
    def _get_boundary_for_strategy(strategy: str, switch_boundary: float) -> float:
        if "T2V boundary" in strategy:
            return DEFAULT_BOUNDARY_T2V
        if "I2V boundary" in strategy:
            return DEFAULT_BOUNDARY_I2V
        return float(max(0.0, min(1.0, switch_boundary)))

    @staticmethod
    def _is_refined_strategy(strategy: str) -> bool:
        return "(refined)" in strategy


class DualKSamplerAdvanced(DualKSamplerAdvancedAlt):
    """Alias node with identical behavior (kept for parity with TripleKSampler)."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        base_inputs = super().INPUT_TYPES()
        required = dict(base_inputs["required"])

        for key in ("switch_step", "switch_boundary", "dry_run"):
            required.pop(key, None)

        optional = {
            "switch_step": (
                "INT",
                {
                    "default": -1,
                    "min": -1,
                    "max": 99,
                    "tooltip": "Manual step to switch models. Only used when switch_strategy is 'Manual switch step'. "
                    "Use -1 to fall back to lightning_start.",
                },
            ),
            "switch_boundary": (
                "FLOAT",
                {
                    "default": DEFAULT_BOUNDARY_T2V,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Boundary used for boundary-based strategies.",
                },
            ),
            "dry_run": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Validate & show resolved stage ranges (toast). Does not sample.",
                },
            ),
        }

        return {"required": required, "optional": optional}
