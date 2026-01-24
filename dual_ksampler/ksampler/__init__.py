"""Native ComfyUI KSampler-based DualKSampler nodes.

These nodes implement a **two-stage** sampling cascade using ComfyUI core
``KSamplerAdvanced``:

Stage 1: ``base_model`` denoises from step 0 → ``base_steps`` over a
         computed ``total_base_steps`` so that it aligns perfectly with
         Stage 2 start.

Stage 2: ``lightning_model`` continues from ``lightning_start`` → ``lightning_steps``.

The alignment math and the sigma-shift patching approach are derived from
ComfyUI-TripleKSampler.
"""

from .base import DualKSamplerBase
from .advanced import DualKSamplerAdvanced, DualKSamplerAdvancedAlt
from .simple import DualKSampler

__all__ = [
    "DualKSamplerBase",
    "DualKSampler",
    "DualKSamplerAdvanced",
    "DualKSamplerAdvancedAlt",
]
