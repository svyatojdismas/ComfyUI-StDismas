"""Model utilities for DualKSampler.

This module mirrors the *sigma shift* patching approach used by
ComfyUI-TripleKSampler for the *native ComfyUI* KSampler path.

For ComfyUI samplers, sigma schedules are derived from each model's
``model_sampling`` object. Wan/Wan2.2 style models often require a
``sigma_shift`` adjustment (ModelSamplingSD3) to match the expected
noise schedule.

All patching operations are **non-mutating**: ``ModelSamplingSD3.patch``
returns a cloned/modified model instance.
"""

from __future__ import annotations

from typing import Any, Tuple


try:
    # ComfyUI extra nodes (present in normal installs)
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3
except Exception:  # pragma: no cover
    ModelSamplingSD3 = None  # type: ignore


def patch_models_with_sigma_shift(
    base_model: Any, lightning_model: Any, sigma_shift: float
) -> Tuple[Any, Any]:
    """Patch both models with a sigma shift using ModelSamplingSD3.

    Args:
        base_model: ComfyUI MODEL object for Stage 1.
        lightning_model: ComfyUI MODEL object for Stage 2.
        sigma_shift: Shift value (float).

    Returns:
        (patched_base_model, patched_lightning_model)

    Raises:
        ImportError: If ModelSamplingSD3 isn't available.
    """

    if ModelSamplingSD3 is None:
        raise ImportError(
            "ModelSamplingSD3 is not available. "
            "Install/enable comfy_extras (standard ComfyUI distribution)."
        )

    patcher = ModelSamplingSD3()
    shift_value = float(sigma_shift)

    patched_base = patcher.patch(base_model, shift_value)[0]
    patched_lightning = patcher.patch(lightning_model, shift_value)[0]
    return patched_base, patched_lightning
