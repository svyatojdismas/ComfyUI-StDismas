"""Optional VideoHelperSuite integration for ComfyUI-StDismas."""

from pathlib import Path

from .integration import find_vhs_load_module, register_video_formats


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_vhs = find_vhs_load_module()
if _vhs is None:
    print(
        "[Comfyui-StDismas] VideoHelperSuite not found; "
        "VHS-dependent nodes are disabled."
    )
else:
    try:
        register_video_formats(Path(__file__).with_name("video_formats"))
        from .load_video_ffmpeg_frames import (
            NODE_CLASS_MAPPINGS as VHS_NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS as VHS_NODE_DISPLAY_NAME_MAPPINGS,
        )

        NODE_CLASS_MAPPINGS.update(VHS_NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(VHS_NODE_DISPLAY_NAME_MAPPINGS)
        print("[Comfyui-StDismas] enabled VideoHelperSuite integration")
    except Exception as exc:
        print(f"[Comfyui-StDismas] VideoHelperSuite integration disabled: {exc}")
