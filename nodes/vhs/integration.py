"""Optional VideoHelperSuite discovery and format registration."""

from __future__ import annotations

import importlib
import importlib.machinery
import os
from pathlib import Path
import sys
import types

import folder_paths


_RUNTIME_PACKAGE = "_stdismas_vhs_runtime"
_LOAD_MODULE_SUFFIX = "videohelpersuite.load_video_nodes"
_REQUIRED_LOAD_ATTRIBUTES = (
    "BIGMAX",
    "DIMMAX",
    "ENCODE_ARGS",
    "ProgressBar",
    "calculate_file_hash",
    "ffmpeg_path",
    "floatOrInt",
    "get_load_formats",
    "imageOrLatent",
    "load_video",
    "strip_path",
    "target_size",
    "video_extensions",
)


def _is_compatible_load_module(module) -> bool:
    return module is not None and all(
        hasattr(module, attribute) for attribute in _REQUIRED_LOAD_ATTRIBUTES
    )


def _registered_load_module():
    """Prefer the VHS module already registered by ComfyUI."""
    try:
        import nodes as comfy_nodes

        node_class = comfy_nodes.NODE_CLASS_MAPPINGS.get("VHS_LoadVideoFFmpeg")
        if node_class is not None:
            module = sys.modules.get(node_class.__module__)
            if _is_compatible_load_module(module):
                return module
    except Exception:
        pass

    for name, module in tuple(sys.modules.items()):
        if name.endswith(_LOAD_MODULE_SUFFIX) and _is_compatible_load_module(module):
            return module
    return None


def _candidate_vhs_roots():
    candidates = []
    try:
        custom_node_roots = folder_paths.get_folder_paths("custom_nodes")
    except Exception:
        custom_node_roots = []

    for custom_node_root in custom_node_roots:
        try:
            entries = os.scandir(custom_node_root)
        except OSError:
            continue
        with entries:
            for entry in entries:
                if not entry.is_dir():
                    continue
                root = Path(entry.path)
                package = root / "videohelpersuite"
                if (
                    (package / "load_video_nodes.py").is_file()
                    and (package / "nodes.py").is_file()
                ):
                    candidates.append(root)

    def priority(root: Path):
        normalized = root.name.casefold().replace("_", "-")
        return (normalized != "comfyui-videohelpersuite", normalized, str(root))

    return sorted(set(candidates), key=priority)


def _import_load_module(vhs_root: Path):
    """Load only VHS's helper package when StDismas is initialized first."""
    package_dir = vhs_root / "videohelpersuite"
    package = sys.modules.get(_RUNTIME_PACKAGE)
    if package is None:
        package = types.ModuleType(_RUNTIME_PACKAGE)
        package.__package__ = _RUNTIME_PACKAGE
        package.__path__ = [str(package_dir)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            _RUNTIME_PACKAGE, loader=None, is_package=True
        )
        package.__spec__.submodule_search_locations = [str(package_dir)]
        sys.modules[_RUNTIME_PACKAGE] = package

    module_name = f"{_RUNTIME_PACKAGE}.load_video_nodes"
    try:
        return importlib.import_module(module_name)
    except Exception:
        for name in tuple(sys.modules):
            if name == _RUNTIME_PACKAGE or name.startswith(f"{_RUNTIME_PACKAGE}."):
                sys.modules.pop(name, None)
        raise


_cached_load_module = None
_discovery_complete = False


def find_vhs_load_module():
    """Return a compatible VHS load module, independent of custom-node load order."""
    global _cached_load_module, _discovery_complete
    if _discovery_complete:
        return _cached_load_module

    module = _registered_load_module()
    if module is None:
        for root in _candidate_vhs_roots():
            try:
                candidate = _import_load_module(root)
            except Exception:
                continue
            if _is_compatible_load_module(candidate):
                module = candidate
                break

    _cached_load_module = module
    _discovery_complete = True
    return module


def register_video_formats(formats_dir: Path) -> None:
    """Expose StDismas JSON presets to the stock VHS Video Combine node."""
    key = "VHS_video_formats"
    formats_path = str(formats_dir.resolve())
    current = folder_paths.folder_names_and_paths.get(key)

    if current is None:
        paths, extensions = [], {""}
    else:
        paths, extensions = list(current[0]), set(current[1])

    if formats_path not in paths:
        paths.append(formats_path)
    # VHS currently keeps the suffix for external filenames when it constructs
    # the visible `video/<format>` value. These files intentionally have no
    # suffix so the UI shows `video/h264-mp4-pc`, matching built-in formats.
    extensions.add("")
    folder_paths.folder_names_and_paths[key] = (paths, extensions)

    folder_paths.filename_list_cache.pop(key, None)
    cache_helper = getattr(folder_paths, "cache_helper", None)
    cache = getattr(cache_helper, "cache", None)
    if isinstance(cache, dict):
        cache.pop(key, None)
