"""Frame-index based FFmpeg upload node built on VideoHelperSuite helpers."""

from __future__ import annotations

import os
import re
import subprocess
import time

import folder_paths
import numpy as np
import torch

from .integration import find_vhs_load_module


VHS = find_vhs_load_module()
if VHS is None:  # Imported directly rather than through nodes.vhs.__init__.
    raise ImportError("ComfyUI-VideoHelperSuite is required for the VHS integration")


def _ffprobe_executable() -> str:
    executable = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    ffmpeg_path = getattr(VHS, "ffmpeg_path", None)
    if ffmpeg_path:
        candidate = os.path.join(os.path.dirname(ffmpeg_path), executable)
        if os.path.isfile(candidate):
            return candidate
    return executable


def _ffprobe_count_frames(video):
    """Return the decoded frame count when ffprobe is available."""
    args = [
        _ffprobe_executable(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        video,
    ]
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        for line in result.stdout.decode(*VHS.ENCODE_ARGS).splitlines():
            if line.strip().isdigit() and int(line) > 0:
                return int(line)
    except Exception:
        pass
    return None


def _ffmpeg_probe(video):
    ffmpeg_path = VHS.ffmpeg_path
    if not ffmpeg_path:
        raise RuntimeError("VideoHelperSuite could not find FFmpeg")

    args_input = ["-i", video]
    args_dummy = [
        ffmpeg_path,
        *args_input,
        "-c",
        "copy",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            args_dummy,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "An error occurred in the FFmpeg probe:\n"
            + exc.stderr.decode(*VHS.ENCODE_ARGS)
        ) from exc

    output = result.stderr.decode(*VHS.ENCODE_ARGS)
    if "Video: vp9 " in output:
        args_input = ["-c:v", "libvpx-vp9", "-i", video]
        args_dummy = [
            ffmpeg_path,
            *args_input,
            "-c",
            "copy",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(
                args_dummy,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "An error occurred in the FFmpeg VP9 probe:\n"
                + exc.stderr.decode(*VHS.ENCODE_ARGS)
            ) from exc
        output = result.stderr.decode(*VHS.ENCODE_ARGS)

    size_base = None
    fps_base = None
    alpha = False
    for line in output.splitlines():
        match = re.search(r"^ *Stream .* Video.*, ([1-9]|\d{2,})x(\d+)", line)
        if match is None:
            continue
        size_base = [int(match.group(1)), int(match.group(2))]
        fps_match = re.search(r", ([\d.]+) fps", line)
        fps_base = float(fps_match.group(1)) if fps_match else 1.0
        alpha = re.search(r"(yuva|rgba|bgra|gbra)", line) is not None
        break
    if size_base is None:
        raise RuntimeError(
            "Failed to parse video information from FFmpeg output:\n" + output
        )

    duration_match = re.search(r"Duration: (\d+:\d+:\d+\.\d+),", output)
    if duration_match:
        hours, minutes, seconds = duration_match.group(1).split(":")
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        duration = 0.0

    return (
        args_input,
        size_base,
        fps_base,
        duration,
        alpha,
        _ffprobe_count_frames(video),
    )


def _estimated_selected_frames(
    source_frame_count,
    duration,
    target_fps,
    force_rate,
    skip_first_frames,
    select_every_nth,
    frame_load_cap,
):
    rate_adjusted = (
        max(int(round(max(duration, 0.0) * target_fps)), 0)
        if force_rate > 0 and duration > 0
        else max(int(source_frame_count), 0)
    )
    selected = max(rate_adjusted - skip_first_frames, 0)
    selected = (selected + select_every_nth - 1) // select_every_nth
    if frame_load_cap > 0:
        selected = min(selected, frame_load_cap)
    return selected


def ffmpeg_frame_generator_frames(
    video,
    force_rate,
    frame_load_cap,
    skip_first_frames=0,
    select_every_nth=1,
    custom_width=0,
    custom_height=0,
    downscale_ratio=8,
    meta_batch=None,
    unique_id=None,
):
    """Decode, resample FPS, then apply exact frame-index selection."""
    select_every_nth = max(int(select_every_nth or 1), 1)
    skip_first_frames = max(int(skip_first_frames or 0), 0)
    frame_load_cap = max(int(frame_load_cap or 0), 0)
    force_rate = max(float(force_rate or 0), 0.0)

    (
        args_input,
        size_base,
        fps_base,
        duration,
        alpha,
        probed_frame_count,
    ) = _ffmpeg_probe(video)
    fps_base = float(fps_base or 1.0)
    source_frame_count = (
        int(probed_frame_count)
        if probed_frame_count
        else int(round(fps_base * duration)) if duration else 0
    )
    target_fps = force_rate or fps_base
    target_frame_time = 1.0 / target_fps

    filters = []
    # FFmpeg's fps filter is responsible for both dropping and duplicating.
    # Python then applies skip/select/cap to the resulting frame sequence.
    if force_rate > 0:
        filters.append(f"fps=fps={force_rate}")

    if custom_width != 0 or custom_height != 0:
        size = VHS.target_size(
            size_base[0],
            size_base[1],
            custom_width,
            custom_height,
            downscale_ratio=downscale_ratio,
        )
        aspect_ratio = float(size[0]) / float(size[1])
        if abs(size_base[0] * aspect_ratio - size_base[1]) >= 1:
            filters.append(
                rf"crop=if(gt({aspect_ratio}\,a)\,iw\,ih*{aspect_ratio})"
                rf":if(gt({aspect_ratio}\,a)\,iw/{aspect_ratio}\,ih)"
            )
        filters.append("scale=" + ":".join(map(str, size)))
    else:
        size = size_base

    args = [VHS.ffmpeg_path, "-v", "error", "-an", *args_input]
    if filters:
        args.extend(["-vf", ",".join(filters)])
    args.extend(["-pix_fmt", "rgba64le", "-f", "rawvideo", "-"])

    estimated_frames = _estimated_selected_frames(
        source_frame_count,
        duration,
        target_fps,
        force_rate,
        skip_first_frames,
        select_every_nth,
        frame_load_cap,
    )
    yield (
        size_base[0],
        size_base[1],
        fps_base,
        duration,
        source_frame_count or fps_base * duration,
        target_frame_time,
        estimated_frames,
        size[0],
        size[1],
        alpha,
    )

    progress = VHS.ProgressBar(estimated_frames)
    frames_added = 0
    resampled_index = 0
    process = None
    try:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        bytes_per_image = size[0] * size[1] * 8
        current_bytes = bytearray(bytes_per_image)
        current_offset = 0

        while True:
            bytes_read = process.stdout.read(bytes_per_image - current_offset)
            if bytes_read is None:
                time.sleep(0.1)
                continue
            if not bytes_read:
                break
            current_bytes[
                current_offset : current_offset + len(bytes_read)
            ] = bytes_read
            current_offset += len(bytes_read)
            if current_offset != bytes_per_image:
                continue

            current_offset = 0
            should_emit = (
                resampled_index >= skip_first_frames
                and (resampled_index - skip_first_frames) % select_every_nth == 0
            )
            resampled_index += 1
            if not should_emit:
                continue

            frame = np.frombuffer(
                current_bytes,
                dtype=np.dtype(np.uint16).newbyteorder("<"),
            ).reshape(size[1], size[0], 4) / (2**16 - 1)
            if not alpha:
                frame = frame[:, :, :-1]
            yield frame
            frames_added += 1
            if estimated_frames:
                progress.update_absolute(frames_added, estimated_frames)
            else:
                progress.update(1)
            if frame_load_cap > 0 and frames_added >= frame_load_cap:
                break

        if process.poll() not in (None, 0):
            raise BrokenPipeError
        if current_offset:
            raise RuntimeError("FFmpeg returned an incomplete video frame")
    except BrokenPipeError as exc:
        error = ""
        if process is not None and process.stderr is not None:
            try:
                error = process.stderr.read().decode(*VHS.ENCODE_ARGS)
            except Exception:
                pass
        raise RuntimeError("An error occurred in the FFmpeg subprocess:\n" + error) from exc
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=1)
            except Exception:
                pass

    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id, None)
        meta_batch.has_closed_inputs = True


class LoadVideoFFmpegUploadFrames:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        for filename in os.listdir(input_dir):
            path = os.path.join(input_dir, filename)
            extension = os.path.splitext(filename)[1].lower().lstrip(".")
            if os.path.isfile(path) and extension in VHS.video_extensions:
                files.append(filename)

        return {
            "required": {
                "video": (sorted(files),),
                "force_rate": (
                    VHS.floatOrInt,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 60,
                        "step": 1,
                        "disable": 0,
                    },
                ),
                "custom_width": (
                    "INT",
                    {"default": 0, "min": 0, "max": VHS.DIMMAX, "disable": 0},
                ),
                "custom_height": (
                    "INT",
                    {"default": 0, "min": 0, "max": VHS.DIMMAX, "disable": 0},
                ),
                "frame_load_cap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": VHS.BIGMAX,
                        "step": 1,
                        "disable": 0,
                    },
                ),
                "skip_first_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": VHS.BIGMAX, "step": 1},
                ),
                "select_every_nth": (
                    "INT",
                    {"default": 1, "min": 1, "max": VHS.BIGMAX, "step": 1},
                ),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "format": VHS.get_load_formats(),
            },
            "hidden": {"force_size": "STRING", "unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "Video Helper Suite 🎥🅟🅗🅢"
    DESCRIPTION = (
        "Loads an uploaded video through FFmpeg. FPS resampling is applied first; "
        "skip_first_frames, select_every_nth and frame_load_cap then operate on "
        "exact frame indexes. Requires ComfyUI-VideoHelperSuite."
    )
    RETURN_TYPES = (VHS.imageOrLatent, "MASK", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "mask", "audio", "video_info")
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        kwargs["video"] = folder_paths.get_annotated_filepath(
            VHS.strip_path(kwargs["video"])
        )
        image, _, audio, video_info = VHS.load_video(
            **kwargs, generator=ffmpeg_frame_generator_frames
        )
        if isinstance(image, dict):
            return image, None, audio, video_info
        if image.size(3) == 4:
            return image[:, :, :, :3], 1 - image[:, :, :, 3], audio, video_info
        mask = torch.zeros(image.size(0), 64, 64, device="cpu")
        return image, mask, audio, video_info

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        return VHS.calculate_file_hash(folder_paths.get_annotated_filepath(video))

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


NODE_CLASS_MAPPINGS = {
    "StDismas_LoadVideoFFmpegFrames": LoadVideoFFmpegUploadFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StDismas_LoadVideoFFmpegFrames": (
        "Load Video FFmpeg (Upload) Frames 🎥🅟🅗🅢"
    ),
}
