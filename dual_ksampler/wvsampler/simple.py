"""Simple dual-stage WanVideo sampler node.

Keeps the core benefit: set base_steps (or auto) and lightning_steps,
then run base → lightning sequentially.

Internally delegates to DualWVSamplerAdvancedAlt with shared scheduler.
"""

from __future__ import annotations

from typing import Any, Dict

from .advanced import DualWVSamplerAdvancedAlt, START_MODES


class DualWVSampler:
    """Simplified DualKSampler node for WanVideo."""

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("samples", "denoised_samples")
    FUNCTION = "sample"
    CATEGORY = "DualKSampler/wanvideo"
    DESCRIPTION = "Dual-stage WanVideo sampler (simple UI): base model then Lightning model."

    def __init__(self):
        self._impl = DualWVSamplerAdvancedAlt()

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        # Build from AdvancedAlt and keep only a common subset
        full = DualWVSamplerAdvancedAlt.INPUT_TYPES()
        r = full.get("required", {})
        o = full.get("optional", {})

        required = {
            "base_model": r["base_model"],
            "lightning_model": r["lightning_model"],
            "image_embeds": r["image_embeds"],
            "seed": r["seed"],
            "sigma_shift": r["sigma_shift"],
            "base_quality_threshold": r["base_quality_threshold"],
            "base_steps": r["base_steps"],
            "base_cfg": r["base_cfg"],
            "scheduler": (r["base_scheduler"][0], {"default": r["base_scheduler"][1].get("default", "unipc")}),
            "lightning_steps": r["lightning_steps"],
            "lightning_cfg": r["lightning_cfg"],
            "lightning_start": r["lightning_start"],
        }

        # Keep WanVideo shared required params if available
        for key in ["force_offload", "riflex_freq_index", "batched_cfg", "rope_function"]:
            if key in r:
                required[key] = r[key]

        # Keep all WanVideo optional params + dry_run
        optional = dict(o)
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
        scheduler: str,
        lightning_steps: int,
        lightning_cfg: float,
        lightning_start: int,
        force_offload: bool,
        riflex_freq_index: int,
        batched_cfg: bool,
        rope_function: str,
        dry_run: bool = False,
        **kwargs,
    ):
        return self._impl.sample(
            base_model=base_model,
            lightning_model=lightning_model,
            image_embeds=image_embeds,
            seed=seed,
            sigma_shift=sigma_shift,
            base_quality_threshold=base_quality_threshold,
            base_steps=base_steps,
            base_cfg=base_cfg,
            base_scheduler=scheduler,
            start_mode=START_MODES[0],  # Manual step
            lightning_start=lightning_start,
            start_boundary=0.9,
            lightning_steps=lightning_steps,
            lightning_cfg=lightning_cfg,
            lightning_scheduler=scheduler,
            force_offload=force_offload,
            riflex_freq_index=riflex_freq_index,
            batched_cfg=batched_cfg,
            rope_function=rope_function,
            dry_run=dry_run,
            **kwargs,
        )
