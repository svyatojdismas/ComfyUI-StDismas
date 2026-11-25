from fractions import Fraction
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from ._mmh3_tile_preview_render import resize_frame, video_overlay


def encode_video_preview(path, frames, fps, layout, pixels_per_cell, show_overlap_and_fade, on_frame=None):
    import av

    scale = min(pixels_per_cell, 1280 / max(layout.width, layout.height))
    width = max(2, round(layout.width * scale / 2) * 2)
    height = max(2, round(layout.height * scale / 2) * 2)
    size = (width, height)
    overlay = video_overlay(layout, size, pixels_per_cell, show_overlap_and_fade)
    rate = Fraction(str(fps)).limit_denominator(1001)
    if rate <= 0:
        raise ValueError("The MMH3 video preview frame rate must be positive.")

    images = np.empty((len(frames), height, width, 3), dtype=np.float32)
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "18"}
        for index, frame in enumerate(frames):
            if on_frame is not None:
                on_frame(index)
            source = resize_frame(frame, size).convert("RGBA")
            composed = np.asarray(Image.alpha_composite(source, overlay).convert("RGB"))
            np.divide(composed, np.float32(255), out=images[index])
            encoded = av.VideoFrame.from_ndarray(composed, format="rgb24")
            container.mux(stream.encode(encoded))
        container.mux(stream.encode())
    return torch.from_numpy(images)


def save_video_preview(frames, fps, layout, pixels_per_cell, show_overlap_and_fade):
    import folder_paths
    from comfy.model_management import throw_exception_if_processing_interrupted
    from comfy.utils import ProgressBar
    from comfy_api.latest import ui

    progress = ProgressBar(len(frames))

    def on_frame(index):
        throw_exception_if_processing_interrupted()
        progress.update_absolute(index)

    temp_dir = Path(folder_paths.get_temp_directory())
    temp_dir.mkdir(parents=True, exist_ok=True)
    handle, filename = tempfile.mkstemp(prefix="MMH3_Tile_Preview_", suffix=".mp4", dir=temp_dir)
    os.close(handle)
    path = Path(filename)
    try:
        images = encode_video_preview(path, frames, fps, layout, pixels_per_cell, show_overlap_and_fade, on_frame)
        progress.update_absolute(len(frames))
    except BaseException:
        path.unlink(missing_ok=True)
        raise

    playable = ui.PreviewVideo([{"filename": path.name, "subfolder": "", "type": "temp"}])
    return images, playable.as_dict()
