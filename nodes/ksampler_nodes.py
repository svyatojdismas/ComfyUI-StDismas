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
    "DualKSamplerCore": "DualKsampler Simple (StDismas)",
    "DualKSamplerCoreAdvanced": "DualKsampler Advanced (StDismas)",
    "DualKSamplerCoreAdvancedAlt": "DualKsampler Advanced (Static UI) (StDismas)",
}
