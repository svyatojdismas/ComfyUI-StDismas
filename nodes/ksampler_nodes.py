"""Node registration for native ComfyUI (core) DualKSampler nodes."""

from __future__ import annotations

from dual_ksampler.ksampler import (
    DualKSampler,
    DualKSamplerAdvanced,
    DualKSamplerAdvancedAlt,
)

NODE_CLASS_MAPPINGS = {
    "DualKSamplerCore": DualKSampler,
    "DualKSamplerCoreAdvanced": DualKSamplerAdvanced,
    "DualKSamplerCoreAdvancedAlt": DualKSamplerAdvancedAlt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualKSamplerCore": "DualKSampler (Core KSampler) (StDismas)",
    "DualKSamplerCoreAdvanced": "DualKSampler (Core KSampler Advanced) (StDismas)",
    "DualKSamplerCoreAdvancedAlt": "DualKSampler (Core KSampler Advanced Alt) (StDismas)",
}
