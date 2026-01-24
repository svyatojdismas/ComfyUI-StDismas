"""Node registration for WanVideo DualKSampler nodes."""

from __future__ import annotations

from dual_ksampler.wvsampler.advanced import DualWVSamplerAdvancedAlt
from dual_ksampler.wvsampler.simple import DualWVSampler

NODE_CLASS_MAPPINGS = {
    "DualWVSamplerWanLightning": DualWVSampler,
    "DualWVSamplerWanLightningAdvancedAlt": DualWVSamplerAdvancedAlt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualWVSamplerWanLightning": "DualKSampler (WanVideo) - Base + Lightning (StDismas)",
    "DualWVSamplerWanLightningAdvancedAlt": "DualKSampler (WanVideo) - Advanced Alt (StDismas)",
}
