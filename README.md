# ComfyUI-StDismas
Custom nodes for ComfyUI

## Batch crop / uncrop

The advanced crop nodes support universal mask tracking plus optional face detection and identity-reference tracking. Face dependencies are loaded only when `tracking_mode=face_detection`; mask workflows do not require Ultralytics or InsightFace.

See [the detailed crop/uncrop documentation](nodes/README_batch_crop_uncrop_by_mask_advanced.md).
