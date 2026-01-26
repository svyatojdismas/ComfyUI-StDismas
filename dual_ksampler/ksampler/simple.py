"""Simplified native (ComfyUI core) DualKSampler node."""

from __future__ import annotations

from typing import Any, Dict

import comfy.samplers

from .advanced import DualKSamplerAdvancedAlt


class DualKSampler(DualKSamplerAdvancedAlt):
    """Simple two-stage sampler with fewer exposed controls.

    - Single sampler & scheduler shared by both stages
    - Lightning CFG fixed to 1.0 by default
    - Base steps can be manual or auto (-1)
    """

    DESCRIPTION = (
        "Dual-stage core KSampler (simple): base + lightning. "
        "Auto base alignment available via base_steps=-1."
    )

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        base = cls._get_base_input_types()

        required: Dict[str, Any] = {
            # models + conditioning + latent + seed + sigma_shift + cfgs
            "base_model": base["base_model"],
            "lightning_model": base["lightning_model"],
            "positive": base["positive"],
            "negative": base["negative"],
            "latent_image": base["latent_image"],
            "seed": base["seed"],
            "sigma_shift": base["sigma_shift"],
            "base_cfg": base["base_cfg"],

            # alignment controls
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

            # shared sampler/scheduler
            "sampler_name": (
                comfy.samplers.KSampler.SAMPLERS,
                {"default": "euler", "tooltip": "Sampler used for both stages."},
            ),
            "scheduler": (
                comfy.samplers.KSampler.SCHEDULERS,
                {"default": "simple", "tooltip": "Scheduler used for both stages."},
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
        base_quality_threshold: int,
        base_steps: int,
        lightning_start: int,
        lightning_steps: int,
        sampler_name: str,
        scheduler: str,
        dry_run: bool = False,
    ):
        # Delegate to AdvancedAlt with shared sampler/scheduler.
        return (super().sample(
            base_model=base_model,
            lightning_model=lightning_model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            seed=seed,
            sigma_shift=sigma_shift,
            base_cfg=base_cfg,
            lightning_cfg=1.0,
            lightning_steps=lightning_steps,
            base_quality_threshold=base_quality_threshold,
            base_steps=base_steps,
            base_sampler=sampler_name,
            base_scheduler=scheduler,
            lightning_start=lightning_start,
            lightning_sampler=sampler_name,
            lightning_scheduler=scheduler,
            switch_strategy="50% of steps",
            switch_boundary=0.875,
            dry_run=dry_run,
        )[0],)
