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

from typing import Any, List, Optional, Tuple

import torch


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


def _resolve_sigmas(model: Any, scheduler: str, steps: int) -> Optional[List[float]]:
    """Resolve the sigma schedule for a model/scheduler pair."""
    import comfy.model_management
    import comfy.samplers

    sigmas = None
    if hasattr(comfy.samplers, "calculate_sigmas"):
        try:
            sigmas = comfy.samplers.calculate_sigmas(model, scheduler, steps)
        except TypeError:
            sigmas = comfy.samplers.calculate_sigmas(model.model_sampling, scheduler, steps)
        except Exception:
            sigmas = None

    if sigmas is None:
        try:
            device = comfy.model_management.get_torch_device()
            sampler = comfy.samplers.KSampler(
                model=model,
                steps=steps,
                device=device,
                sampler_name="euler",
                scheduler=scheduler,
                denoise=1.0,
            )
            sigmas = getattr(sampler, "sigmas", None)
        except Exception:
            sigmas = None

    if sigmas is None:
        return None

    sigmas_list = []
    for sigma in sigmas:
        try:
            sigmas_list.append(float(sigma.item()))
        except Exception:
            sigmas_list.append(float(sigma))

    return sigmas_list


def get_sigmas_tensor(model: Any, scheduler: str, steps: int) -> torch.Tensor:
    """Return sigma schedule as a torch tensor for ComfyUI SIGMAS output."""
    sigmas = _resolve_sigmas(model, scheduler, steps)
    if sigmas is None:
        return torch.zeros(0, dtype=torch.float32)
    return torch.tensor(sigmas, dtype=torch.float32)


def calculate_boundary_step(
    model: Any, scheduler: str, steps: int, boundary: float, fallback_step: int
) -> int:
    """Calculate the step index where sigma drops below the boundary."""
    sigmas = _resolve_sigmas(model, scheduler, steps)
    if not sigmas:
        return fallback_step

    sigma_list = sigmas[:-1] if len(sigmas) > steps else sigmas
    switch_step = steps - 1
    for idx, sigma in enumerate(sigma_list):
        if float(sigma) < float(boundary):
            switch_step = idx
            break

    return int(max(0, min(steps - 1, switch_step)))


def calculate_perfect_shift_for_step(
    model: Any,
    scheduler: str,
    total_steps: int,
    target_step: int,
    target_sigma: float,
    initial_shift: float,
    max_iters: int = 12,
    tolerance: float = 1e-4,
) -> Tuple[float, str]:
    """Attempt to refine sigma_shift so sigma at target_step matches target_sigma."""
    if ModelSamplingSD3 is None:
        return initial_shift, "ModelSamplingSD3 unavailable"

    target_step = int(max(0, min(total_steps - 1, target_step)))

    def sigma_at_shift(shift: float) -> float:
        patched = ModelSamplingSD3().patch(model, float(shift))[0]
        sigmas = _resolve_sigmas(patched, scheduler, total_steps)
        if not sigmas:
            raise RuntimeError("sigma schedule unavailable")
        sigma_list = sigmas[:-1] if len(sigmas) > total_steps else sigmas
        return float(sigma_list[target_step])

    try:
        initial_sigma = sigma_at_shift(initial_shift)
    except Exception as exc:
        return initial_shift, f"refinement unavailable: {exc}"

    if abs(initial_sigma - target_sigma) <= tolerance:
        return initial_shift, "already aligned"

    shift_low = 0.0
    shift_high = max(10.0, float(initial_shift) * 2.0 + 1.0)
    sigma_low = sigma_at_shift(shift_low)
    sigma_high = sigma_at_shift(shift_high)

    for _ in range(4):
        if (sigma_low - target_sigma) * (sigma_high - target_sigma) <= 0:
            break
        shift_high = min(100.0, shift_high * 2.0)
        sigma_high = sigma_at_shift(shift_high)

    if (sigma_low - target_sigma) * (sigma_high - target_sigma) > 0:
        return initial_shift, "unable to bracket target"

    refined = float(initial_shift)
    for _ in range(max_iters):
        refined = (shift_low + shift_high) / 2.0
        sigma_mid = sigma_at_shift(refined)
        if abs(sigma_mid - target_sigma) <= tolerance:
            return refined, "converged"
        if (sigma_low - target_sigma) * (sigma_mid - target_sigma) <= 0:
            shift_high = refined
            sigma_high = sigma_mid
        else:
            shift_low = refined
            sigma_low = sigma_mid

    return refined, "max iterations reached"
