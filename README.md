# ComfyUI-StDismas
Custom nodes for ComfyUI

## Batch crop / uncrop

The advanced crop nodes support universal mask tracking plus optional face detection and identity-reference tracking. Face dependencies are loaded only when `tracking_mode=face_detection`; mask workflows do not require Ultralytics or InsightFace.

See [the detailed crop/uncrop documentation](nodes/README_batch_crop_uncrop_by_mask_advanced.md).

## Optional VideoHelperSuite integration

When [ComfyUI-VideoHelperSuite](https://github.com/kosinkadink/ComfyUI-VideoHelperSuite)
is installed, StDismas adds:

- `Load Video FFmpeg (Upload) Frames`: FFmpeg FPS resampling followed by exact
  `skip_first_frames`, `select_every_nth`, and `frame_load_cap` selection;
- `video/h264-mp4-pc` and `video/nvenc_h264-mp4-pc` full-range BT.709 presets in
  the standard VHS `Video Combine` node.

The integration is optional. If VideoHelperSuite is absent or incompatible, the
rest of the StDismas node pack continues to load normally.
