from dataclasses import dataclass
from itertools import accumulate
from typing import NamedTuple

import numpy as np
import torch

from ._mmh3_tile_preview_render import render_preview
from ._mmh3_tile_preview_video import save_video_preview


PIXELS_PER_CELL = 16
FRAME_BOUNDARIES = (0, *accumulate((1, 4, 4, 4, 4)))
TOKENS_PER_GROUP = len(FRAME_BOUNDARIES) - 1
FRAMES_PER_GROUP = FRAME_BOUNDARIES[-1]


class AxisTile(NamedTuple):
    start: int
    size: int
    overlap: int


class Chunk(NamedTuple):
    token_start: int
    frame_start: int
    token_end: int
    frame_end: int


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    columns: list[AxisTile]
    rows: list[AxisTile]
    spatial: dict | None
    temporal: dict | None
    chunks: list[Chunk]
    frames: int | None
    duration_source: str


def frames_for_tokens(tokens):
    groups, remainder = divmod(tokens, TOKENS_PER_GROUP)
    return groups * FRAMES_PER_GROUP + FRAME_BOUNDARIES[remainder]


def temporal_chunks(tokens, temporal):
    frames = frames_for_tokens(tokens)
    if temporal is None:
        return [Chunk(0, 0, tokens, frames)]

    length = int(temporal["chunk_length"])
    overlap = int(temporal["temporal_overlap"])
    if not 0 <= overlap < length:
        raise ValueError("MMH3 temporal_overlap must be non-negative and smaller than chunk_length.")

    def snap(frame):
        # MMH3 snaps to the nearest keyframe-phase token boundary (5 tokens / 17 frames).
        group = (frame + FRAMES_PER_GROUP // 2) // FRAMES_PER_GROUP
        return min(group, tokens // TOKENS_PER_GROUP) * TOKENS_PER_GROUP

    chunks = []
    for start in range(0, frames, length - overlap):
        first = min(snap(start), chunks[-1].token_end) if chunks else 0
        end_frame = start + length
        last = tokens if end_frame >= frames else snap(end_frame)
        if last <= first:
            last = min(first + TOKENS_PER_GROUP, tokens)
        chunks.append(Chunk(first, frames_for_tokens(first), last, frames_for_tokens(last)))
        if last == tokens:
            break
    return chunks


def axis_tiles(size, tile, overlap, minimum):
    # MMH3 spaces tiles by tile - overlap, then pulls a short edge tile inward.
    if not 0 <= overlap < tile or not 0 <= minimum <= tile:
        raise ValueError("MMH3 overlap must be smaller than the tile, and min_tile_size must fit it.")
    if size <= tile:
        return [AxisTile(0, size, 0)]

    stride = tile - overlap
    last_index = (size - tile + stride - 1) // stride
    starts = list(range(0, (last_index + 1) * stride, stride))
    if size - starts[-1] < minimum and starts[-2] < size - minimum < starts[-2] + tile:
        starts[-1] = size - minimum

    tiles = []
    previous_end = 0
    for start in starts:
        end = min(start + tile, size)
        tiles.append(AxisTile(start, end - start, max(0, previous_end - start)))
        previous_end = end
    return tiles


def build_layout(latent=None, latent_upscale_param=None, temporal_split_param=None,
                 spatial_split_param=None, planned_frames=0):
    tokens = None
    height = width = 0
    source = ""
    if latent is not None:
        samples = latent.get("samples")
        parts = getattr(samples, "tensors", ())
        if (not getattr(samples, "is_nested", False) or len(parts) != 2
                or parts[0].ndim != 5 or parts[0].shape[:2] != (1, 24)
                or parts[1].ndim != 4 or parts[1].shape[1] != 32):
            raise ValueError("Connect the batch-1 MiniMax H3 AV latent going into MMH3 Ultimate Upscale.")
        tokens, height, width = parts[0].shape[-3:]
        if min(tokens, height, width) < 1:
            raise ValueError("The MiniMax H3 video latent must have positive frame and spatial dimensions.")
        source = "input latent"
    elif planned_frames > 0:
        # Match the H3 Empty Latent / conditioning nodes' 17k + 5 pixel-frame grid.
        groups = (max(5, planned_frames) - 5 + FRAMES_PER_GROUP - 1) // FRAMES_PER_GROUP
        tokens = groups * TOKENS_PER_GROUP + 2
        source = f"planned length ({planned_frames} requested)"

    spatial = spatial_split_param
    if latent_upscale_param is not None:
        target_w = int(latent_upscale_param["width"])
        target_h = int(latent_upscale_param["height"])
        if spatial is not None and (target_w < spatial["tile_width"] or target_h < spatial["tile_height"]):
            raise ValueError(
                f"MMH3 Ultimate Upscale requires the target ({target_w} x {target_h}) to be at least "
                f"the configured tile size ({spatial['tile_width']} x {spatial['tile_height']})."
            )
        target_cells_w = max(1, round(target_w / PIXELS_PER_CELL))
        target_cells_h = max(1, round(target_h / PIXELS_PER_CELL))
        if latent is not None and "model_name" in latent_upscale_param:
            scale = (target_w / (width * PIXELS_PER_CELL) + target_h / (height * PIXELS_PER_CELL)) / 2
            if scale < 1 and (target_cells_w < width or target_cells_h < height):
                raise ValueError("The MMH3 3D upscale model requires an effective scale of at least 1.0.")
        width, height = target_cells_w, target_cells_h
    elif latent is None:
        raise ValueError("Connect latent_upscale_param for the target size, or a MiniMax H3 AV latent.")

    if spatial is None:
        columns = [AxisTile(0, width, 0)]
        rows = [AxisTile(0, height, 0)]
    else:
        minimum = int(spatial["min_tile_size"]) // PIXELS_PER_CELL
        columns = axis_tiles(width, int(spatial["tile_width"]) // PIXELS_PER_CELL,
                             int(spatial["spatial_w_overlap"]) // PIXELS_PER_CELL, minimum)
        rows = axis_tiles(height, int(spatial["tile_height"]) // PIXELS_PER_CELL,
                          int(spatial["spatial_h_overlap"]) // PIXELS_PER_CELL, minimum)

    chunks = temporal_chunks(tokens, temporal_split_param) if tokens is not None else []
    frames = frames_for_tokens(tokens) if tokens is not None else None
    return Layout(width, height, columns, rows, spatial, temporal_split_param, chunks, frames, source)


def format_layout(layout):
    px = PIXELS_PER_CELL
    tile_count = len(layout.columns) * len(layout.rows)
    lines = [
        f"Frame: {layout.width * px} x {layout.height * px} px ({layout.width} x {layout.height} latent cells)",
        f"Grid: {len(layout.columns)} columns x {len(layout.rows)} rows; {tile_count} tiles per chunk",
    ]
    if layout.spatial is not None:
        sp = layout.spatial
        lines.extend([
            f"Stitch: {sp['overlap_mode']} / {sp['overlap_blend']}",
            f"Configured overlap X/Y: {sp['spatial_w_overlap']}/{sp['spatial_h_overlap']} px; "
            f"fade X/Y: {sp['fade_width']}/{sp['fade_height']} px; minimum edge: {sp['min_tile_size']} px",
        ])
        if "dynamic_fade" in sp:
            lines.append(f"Dynamic fade: {sp['dynamic_fade']} (min {sp.get('dynamic_fade_min', 0)} px); "
                         f"masked noise: {sp.get('masked_area_noise', 0):.0%}; "
                         f"brightness match: {sp.get('brightness_match', False)}")

    lines.append("Sampling order and incoming left/top overlaps (px):")
    for i, row in enumerate(layout.rows):
        for j, col in enumerate(layout.columns):
            tile = i * len(layout.columns) + j + 1
            line = (f"  {tile:02d}  x={col.start * px}, y={row.start * px}, "
                    f"{col.size * px} x {row.size * px}; overlap {col.overlap * px}/{row.overlap * px}")
            if col.size % 2 or row.size % 2:
                patch_w = (col.size + col.size % 2) * px
                patch_h = (row.size + row.size % 2) * px
                line += f"; model patch footprint {patch_w} x {patch_h}"
            lines.append(line)

    if layout.frames is None:
        lines.append("Duration: connect the input latent or set planned_frames for the exact chunk ranges.")
    else:
        lines.append(f"Duration: {layout.frames} frames ({layout.duration_source}); "
                     f"{len(layout.chunks)} chunks, {len(layout.chunks) * tile_count} sampler calls")
        previous_end = 0
        for i, chunk in enumerate(layout.chunks, start=1):
            overlap = max(0, min(previous_end, chunk.frame_end) - chunk.frame_start)
            lines.append(f"  {i:02d}  frames [{chunk.frame_start}, {chunk.frame_end}) "
                         f"({chunk.frame_end - chunk.frame_start}); tokens [{chunk.token_start}, {chunk.token_end}); "
                         f"overlap {overlap} frames")
            previous_end = max(previous_end, chunk.frame_end)

    if layout.temporal is not None:
        tp = layout.temporal
        lines.append(f"Configured chunk/overlap: {tp['chunk_length']}/{tp['temporal_overlap']} frames; "
                     f"anchor strength: {tp['anchor_strength']:g}")
    return "\n".join(lines)


class MMH3TilePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "planned_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip":
                    "Source clip length when no latent is linked. The H3 generator snaps it up to 17k+5. "
                    "Zero leaves the timeline undetermined. A linked latent takes priority."}),
            },
            "optional": {
                "show_overlap_and_fade": ("BOOLEAN", {"default": True, "tooltip":
                    "Show spatial overlap and fade, stitch weights, and shared frames on the timeline. "
                    "Turn off for a clean tile and chunk diagram."}),
                "preview_mode": (["diagram", "video"], {"default": "diagram", "tooltip":
                    "Use the still diagram, or play the tile grid over the frames in background_image."}),
                "preview_fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01,
                    "tooltip": "Frame rate of the input frame batch, used in video mode."}),
                "spatial_split_param": ("H3_SPATIAL_PARAM",),
                "temporal_split_param": ("H3_TEMPORAL_PARAM",),
                "latent_upscale_param": ("H3_UPSCALE_PARAM",),
                "latent": ("LATENT", {"tooltip": "Link the same MiniMax H3 AV latent as Ultimate Upscale. "
                    "Uses its clip length and, when latent_upscale_param is disconnected, its frame size."}),
                "background_image": ("IMAGE", {"tooltip": "Source frame for the diagram, or a batch of video frames "
                                                 "for a playable preview. Diagram mode uses the first frame."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("preview", "layout_report", "preview_video")
    CATEGORY = "StDismas/Video"
    FUNCTION = "preview"
    OUTPUT_NODE = True
    DESCRIPTION = "Preview MMH3 Ultimate Upscale tiles on a CPU diagram or over a playable clip."

    def preview(self, planned_frames=0, spatial_split_param=None, temporal_split_param=None,
                latent_upscale_param=None, latent=None, background_image=None, prompt=None, extra_pnginfo=None,
                show_overlap_and_fade=True, preview_mode="diagram", preview_fps=24.0):
        if preview_mode == "video" and (background_image is None or len(background_image) < 2):
            raise ValueError("Video preview needs at least two frames in background_image. "
                             "Connect an IMAGE batch from a video loader or decoder.")

        layout = build_layout(latent, latent_upscale_param, temporal_split_param, spatial_split_param, planned_frames)
        diagram = render_preview(layout, background_image, PIXELS_PER_CELL, show_overlap_and_fade)
        preview = torch.from_numpy(np.array(diagram, dtype=np.float32) / 255.0).unsqueeze(0)
        if preview_mode == "video":
            video_frames, ui = save_video_preview(background_image, preview_fps, layout, PIXELS_PER_CELL,
                                                  show_overlap_and_fade)
        else:
            from comfy_execution.graph_utils import ExecutionBlocker
            from nodes import PreviewImage

            video_frames = ExecutionBlocker(None)
            ui = PreviewImage().save_images(preview, "MMH3_Tile_Preview", prompt, extra_pnginfo)["ui"]
        return {"ui": ui, "result": (preview, format_layout(layout), video_frames)}


NODE_CLASS_MAPPINGS = {"StDismasMMH3TilePreview": MMH3TilePreview}
NODE_DISPLAY_NAME_MAPPINGS = {"StDismasMMH3TilePreview": "MMH3 Tile Preview (StDismas)"}
