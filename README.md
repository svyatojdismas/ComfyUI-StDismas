# ComfyUI-StDismas

Custom nodes for ComfyUI focused on stable batch/video cropping, mask-based compositing, optional face tracking, and practical video loading helpers.

<p align="center"><a href="README.md">English</a> | <a href="README.ru.md">Russian</a></p>

## Batch crop / uncrop

The advanced crop nodes support universal mask tracking, optional face detection, identity-reference tracking, exact affine uncrop through `crop_metadata`, and compatible rectangular uncrop through `BOUNDING_BOX`.

Face dependencies are loaded only when `tracking_mode=face_detection`; mask workflows do not require Ultralytics or InsightFace. Thanks [Carasibana](https://github.com/Carasibana/ComfyUI-H3-FaceRefine) for this implementation.

The detailed documentation is available in English:

- [Batch Crop / Uncrop Advanced](nodes/README_batch_crop_uncrop_by_mask_advanced_en.md)

## Optional VideoHelperSuite integration

When [ComfyUI-VideoHelperSuite](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite)
is installed, StDismas adds:

- `Load Video FFmpeg (Upload) Frames`: FFmpeg FPS resampling followed by exact
  `skip_first_frames`, `select_every_nth`, and `frame_load_cap` selection;
- `video/h264-mp4-pc` and `video/nvenc_h264-mp4-pc` full-range BT.709 presets in
  the standard VHS `Video Combine` node.

The integration is optional. If VideoHelperSuite is absent or incompatible, the
rest of the StDismas node pack continues to load normally.

## MMH3 Tile Preview

`MMH3 Tile Preview (StDismas)` shows the spatial tile grid, sampling overlaps and fade, stitch weights, and temporal chunks for `MMH3 Ultimate Upscale`. It renders a CPU diagram or a playable preview in the node and returns the diagram as `IMAGE` and a `STRING` report with coordinates, sizes, and frame ranges.

1. Branch `spatial_split_param`, `temporal_split_param`, and `latent_upscale_param` from the same [MMH3 parameter nodes](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale) into both nodes. The preview reads the resolved settings, including equal tile sizes from newer versions.
2. Link the same `latent` that enters `MMH3 Ultimate Upscale`. If another node already resized it, connect that node's output to both nodes and leave `latent_upscale_param` disconnected. The preview reads the frame size and clip length from the latent. For planning with an upscale target, leave `latent` disconnected and enter `planned_frames`. It follows the MiniMax H3 generator's `17k+5` rounding. At `0`, the spatial preview still works with a target and the duration stays unknown.
3. Optionally connect a source frame to `background_image`. A batch uses its first image, resized to the target aspect ratio for the overlay.

White lines mark tile bounds in sampling order. Orange and teal show the overlaid incoming protected and fade bands. The stitch map colors blend according to the selected mode, including intersecting seams. With dynamic fade the mask map shows the widest configured fade. The diagram needs neither VAE decoding nor an upscale sampling pass.

Turn off `show_overlap_and_fade` for a clean preview with tile bounds, tile sizes, and chunk ranges. This hides the colored seam overlays, stitch map, and shared-frame highlight. The report still includes the numerical overlap and fade settings.

For playback, set `preview_mode` to `video` and connect a batch of frames to `background_image`, for example the `IMAGE` output of a VHS video loader or `Load Video` → `Get Video Components`. Match `preview_fps` to the frame batch's rate. The tile grid plays over the clip, and `show_overlap_and_fade` toggles the seam colors. The node plays a silent MP4 at up to 1280 pixels on the long edge. The `preview_video` socket outputs an `IMAGE` batch of the annotated frames. Connect it to `VHS Video Combine` and set the frame rate there for export.

## Installation
Clone into ComfyUI/custom_nodes/ and restart ComfyUI:
```
git clone https://github.com/svyatojdismas/ComfyUI-StDismas.git
```
