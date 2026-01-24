import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

"""
Comfyui-StDismas nodes auto-loader.

This mirrors the KJNodes approach: keep node implementations as individual .py files
inside ./nodes and import them dynamically. Any module that defines
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS will be merged.

If a module fails to import, it will be skipped (so one broken file doesn't hide the whole pack).
"""
import os
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _merge(m):
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    NODE_CLASS_MAPPINGS.update(getattr(m, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(m, "NODE_DISPLAY_NAME_MAPPINGS", {}))

_pkg = __package__  # "Comfyui-StDismas.nodes"
_here = os.path.dirname(__file__)

for fn in sorted(os.listdir(_here)):
    if not fn.endswith(".py"):
        continue
    if fn.startswith("_") or fn == "__init__.py":
        continue
    modname = fn[:-3]
    try:
        m = importlib.import_module(f".{modname}", package=_pkg)
        _merge(m)
        print(f"[Comfyui-StDismas] loaded nodes module: {modname}")
    except Exception as e:
        print(f"[Comfyui-StDismas] FAILED to load module {modname}: {e}")
