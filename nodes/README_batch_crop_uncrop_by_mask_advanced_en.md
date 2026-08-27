# Batch Image Crop / Uncrop By Mask Advanced — documentation

These nodes are part of [ComfyUI-StDismas](https://github.com/svyatojdismas/ComfyUI-StDismas). They crop image/video batches frame by frame and put processed crops back into the original frames.

## Goals and capabilities

The pair is designed for a stable workflow:

```text
images/video ──► Crop ──► generation/inpaint/detailing ──► Uncrop ──► full-frame result
```

Crop can track any object from a mask or track faces with an optional identity reference. Uncrop can restore the crop using the exact affine transform or a simple integer bounding box.

Key capabilities:

- universal mask tracking for any object;
- optional Ultralytics face detection;
- identity tracking with `identity_reference` and InsightFace;
- interpolation of missing detections;
- independent temporal smoothing for crop center and crop size;
- manual, custom, and automatic output resolution;
- reusable trajectories through `pipe`;
- exact `crop_metadata` output and compatible `BOUNDING_BOX` output;
- mask dilation followed by Gaussian feathering;
- rectangular crop-canvas stitching or full-crop overlay;
- color matching, dropout handling, and chunked processing.

For Crop → Uncrop inside this package, prefer `crop_metadata`: it preserves fractional affine geometry. Use `bboxes` when another node expects the native `BOUNDING_BOX` format.

| Positioning input | Precision | Requirements | Recommended use |
|---|---:|---|---|
| `crop_metadata` | Highest | Metadata from Crop; `base_images` recommended | Normal StDismas workflow |
| `bboxes` | Integer pixels | `BOUNDING_BOX` plus `base_images` | Interoperability with other nodes |

---

# Batch Image Crop By Mask Advanced (StDismas)

## Processing model

For each frame Crop:

1. obtains an object bbox from `crop_mask` or a face detector;
2. interpolates gaps between valid frames;
3. applies `margin_scale`, `size_metric`, offsets, and aspect ratio;
4. smooths center and crop size independently;
5. selects one canvas resolution for the whole batch;
6. creates a per-frame affine transform;
7. samples the image and all masks with the same transform.

Missing mask frames between valid detections do not jump to the frame center. Leading/trailing gaps use the nearest known value.

## Inputs

| Input | Type | Description |
|---|---|---|
| `images` | `IMAGE` | Source image or video batch. |
| `crop_mask` | `MASK` | Main tracking mask. Required only for `tracking_mode = mask`; a single frame can be broadcast to the batch. |
| `masks` | `MASK` | Optional extra mask sampled with the same transform; it does not affect crop geometry. |
| `pipe` | `CROP_PIPE` | Parameters and transforms from another Crop node. |
| `identity_reference` | `IMAGE` | Optional identity reference used in face mode. |

## Tracking modes

### `tracking_mode = mask`

Use this mode for people, faces, objects, inpaint regions, or any other segmentation mask. The non-empty area of `crop_mask` becomes the source bbox. `masks` is an additional output mask and does not change the trajectory.

### `tracking_mode = face_detection`

Ultralytics detects faces frame by frame. `crop_mask` is not required for geometry. `cropped_masks` becomes a rectangular mask covering the selected face bbox.

Face-only widgets appear when this mode is selected:

| Parameter | Default | Description |
|---|---:|---|
| `face_detector` | First available model | Detector from the registered Ultralytics model folders. |
| `face_confidence` | `0.35` | Minimum detector confidence. |
| `face_select` | `largest` | Initial tie-break: largest or most-central face. |
| `identity_track` | `true` | Use continuity and identity matching when needed. |
| `identity_threshold` | `0.28` | Minimum identity similarity. |
| `identity_pack` | `buffalo_l` | InsightFace pack; `buffalo_s` uses less memory. |
| `face_model_device` | `cpu` | `cpu`, `auto`, or `cuda`. |
| `keep_face_models_loaded` | `false` | Keep optional face models in memory after processing. |
| `fallback_detector` | `none` | Optional person/body detector for missing faces. |
| `fallback_head_frac` | `0.5` | Estimated head position inside a fallback person bbox. |

### Face identity tracking

With `identity_reference`, InsightFace extracts an embedding from the first reference image and uses it as the identity anchor. Without a reference, the node may build an anchor from unambiguous frames when multiple people are present.

Continuity is used first: the next face is selected by proximity and overlap with the previous selection. InsightFace is used for ambiguous multi-face frames. A fallback body detector can estimate head position when the face disappears; remaining gaps are interpolated.

The open-source [ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine) project is a useful reference for thoughtful face crop/refine workflows and reverse compositing. StDismas integrates face and identity tracking into the same universal Crop node as mask tracking.

### Face dependencies

Face models are loaded lazily and are not needed for `tracking_mode = mask`. Face mode needs `ultralytics`, `insightface`, exactly one ONNX Runtime build (`onnxruntime` or `onnxruntime-gpu`), and a face detector such as `face_yolov8m.pt` in `ComfyUI/models/ultralytics/bbox`.

See the repository `requirements.txt` for the current dependency list. Do not install CPU and GPU ONNX Runtime builds together. For low VRAM, use `face_model_device = cpu`, `identity_pack = buffalo_s`, and `keep_face_models_loaded = false`.

## Crop geometry and resolution

### Aspect ratio

Preset values:

```text
1:1, 16:9, 9:16, 4:3, 3:4, 4:5, 5:4,
2:3, 3:2, 21:9, 9:21
```

Enable `use_custom_aspect_ratio` and enter `custom_aspect_ratio` as `W:H` (for example `2.39:1`) for any other ratio. The custom field is shown only when enabled.

### Resolution controls

| Parameter | Default | Description |
|---|---:|---|
| `resolution_mode` | `manual` | `manual`, `auto_no_downscale`, or `auto_capped`. |
| `output_resolution_side` | `1024` | Integer size of the selected side in manual mode. |
| `use_long_side` | `true` | Interpret the value as the long or short side. |
| `use_custom_resolution` | `false` | Use exact `width` and `height`. |
| `width` | `1024` | Visible only with custom resolution. |
| `height` | `576` | Visible only with custom resolution. |
| `auto_resolution_cap` | `768` | Maximum long side for `auto_capped`. |
| `divisible_by` | `2` | Make both output dimensions divisible by this value. |

`manual` uses the selected side or custom dimensions. `auto_no_downscale` chooses one canvas large enough for the largest source crop in the batch. `auto_capped` applies the same logic but limits the long side. Automatic modes keep one output size for every frame.

### Object size and framing

| Parameter | Default | Description |
|---|---:|---|
| `margin_scale` | `2.0` | Expands the source bbox before fitting; values below `1.0` are treated as `1.0`. |
| `size_metric` | `bbox_fit` | Controls which bbox dimension drives the crop. |
| `min_zoom` | `0.25` | Minimum affine scale. |
| `max_zoom` | `6.0` | Maximum affine scale. |
| `fit_frame_bounds` | `true` | Keep the crop window inside the source frame. |
| `offset_x` / `offset_y` | `0` | Shift the crop center in source pixels. |

`size_metric` values are `bbox_fit`, `height`, `width`, `max_dimension`, and `area_sqrt`. `height` is often the most stable choice for a face turning in profile.

## Center and size smoothing

The two windows are independent:

- `center_smooth_window` smooths only the crop center, so it controls left/right/up/down movement;
- `size_smooth_window` smooths only the source window size, so it controls zoom breathing.

The value is the number of neighboring frames used to build the filtered value. For example, `center_smooth_window = 21` uses about 10 frames on either side of a central frame. A larger window removes more detector jitter but also reacts more slowly to real changes. The node limits a window to the batch length and makes it odd.

| Parameter | Default | Description |
|---|---:|---|
| `smooth_center` | `true` | Enable center smoothing. |
| `center_smooth_window` | `21` | Temporal center window. |
| `center_smoothing_strength` | `0.25` | Center response to the filtered trajectory. |
| `smooth_zoom` | `true` | Enable size/zoom smoothing. |
| `size_smooth_window` | `51` | Temporal size window. |
| `zoom_smoothing_strength` | `0.25` | Zoom response to the filtered trajectory. |
| `smoothing_method` | `gaussian` | `gaussian`, `savgol`, `moving_average`, `ema`, or `none`. |

Windows are used by `gaussian`, `savgol`, and `moving_average`. They are not used by `ema`; `none` disables smoothing. Strength `0` locks the trajectory to its first value, small values add inertia, and `1` follows the filtered trajectory directly.

Typical values:

| Scene | Center window | Size window |
|---|---:|---:|
| Fast movement | `5–11` | `9–21` |
| Normal talking-head video | `15–31` | `31–61` |
| Static or slow shot | `31–61` | `61–121` |

## Sampling and performance

`interpolation` supports `bilinear` and `bicubic`. Bilinear is faster; bicubic is smoother for substantial scaling. Masks use nearest sampling. `crop_chunk_size` defaults to `128` and controls how many frames are sampled together. Lower it when VRAM/RAM is limited. Geometry and resampling use FP32 internally and return the original image dtype.

## Reusing a trajectory with `pipe`

`pipe` transfers crop parameters and per-frame transforms to another Crop node. This is useful for control images, depth, normals, or other aligned batches. A one-frame pipe can broadcast to a batch; otherwise batch lengths must match. A different source resolution is supported only when its aspect ratio is unchanged, because the transform can then be rescaled proportionally.

`interpolation` and `crop_chunk_size` remain local to the receiving node; other crop settings may be overridden by the pipe.

## Crop outputs

| Output | Type | Description |
|---|---|---|
| `cropped_images` | `IMAGE` | Cropped image batch. |
| `cropped_masks` | `MASK` | Main mask in crop space; rectangular face mask in face mode. |
| `masks` | `MASK` | Extra transformed mask, or `cropped_masks` when no extra mask is supplied. |
| `visualize` | `IMAGE` | Source frames with a red crop rectangle; rendered only when connected. |
| `crop_metadata` | `BBOXES` | Exact affine transforms, sizes, validity flags, and statistics. |
| `bboxes` | `BOUNDING_BOX` | Rounded crop-window rectangles. |
| `pipe` | `CROP_PIPE` | Reusable parameters and metadata. |
| `report` | `STRING` | Tracking, resolution, magnification, jitter, and warnings. |
| `canvas_width` / `canvas_height` | `INT` | Final crop canvas dimensions. |

`bboxes` follows the batch format used by [MaskVidExperiments](https://github.com/drozbay/MaskVidExperiments):

```text
[
  [{"x": 100, "y": 50, "width": 512, "height": 512}],
  [{"x": 104, "y": 52, "width": 512, "height": 512}]
]
```

The rectangles describe the final crop window, not the raw mask bbox. They are rounded to integer pixels, so `crop_metadata` is more accurate for an internal Crop → Uncrop pair.

---

# Batch Image Uncrop By Mask Advanced (StDismas)

## Positioning paths

Uncrop restores `cropped_images` into `base_images` using one of two inputs:

1. `crop_metadata` uses the original affine transform, supports fractional coordinates, color matching, and tracking validity;
2. `bboxes` scales the crop into `{x, y, width, height}` and is interoperable but integer-based.

If both are connected, `crop_metadata` takes precedence. A bbox path requires `base_images`; an affine path can create a black canvas from `orig_size` when no base is supplied.

## Inputs and defaults

| Input / parameter | Type / default | Description |
|---|---|---|
| `cropped_images` | `IMAGE` | Processed crop batch. |
| `base_images` | `IMAGE` | Background batch; required for bbox path. |
| `original_images` | `IMAGE` | Legacy alias for `base_images`. |
| `crop_metadata` | `BBOXES` | Exact metadata from Crop. |
| `bboxes` | `BOUNDING_BOX` | Batch rectangles in native format. |
| `crop_masks` | `MASK` | Crop-space mask for mask-based stitching. |
| `mode` | `overlay_by_mask` | `overlay_by_mask` or `overlay_full`. |
| `blend` | `1.0` | Overall insertion strength from `0.0` to `1.0`. |

## Overlay modes

`overlay_full` uses an all-ones alpha over the crop window. It does not use `crop_masks`, `mask_expand_px`, `feather_radius`, or canvas-mask settings.

`overlay_by_mask` selects its alpha source with `use_crop_canvas_mask`:

- `false` — use `crop_masks`, then dilate and feather it;
- `true` — use an independent rectangular crop-canvas alpha with per-side inset/fade.

The rectangular mode is intentionally separate from `overlay_full`: it still gives a soft, configurable alpha boundary instead of replacing the entire crop window.

## Mask-based stitching

Use:

```text
mode = overlay_by_mask
use_crop_canvas_mask = false
```

Alpha processing is:

```text
crop_masks → dilation(mask_expand_px) → placement/warp → Gaussian feather → blend
```

| Parameter | Default | Description |
|---|---:|---|
| `mask_expand_px` | `16` | Expand the dense mask area by this many crop-canvas pixels. |
| `feather_radius` | `16` | Blur the expanded edge. |
| `border_blending` | `0.25` | Legacy fallback used only when `feather_radius = 0`; it is converted to a feather radius. |

Dilation and feathering are different operations. Dilation makes the new crop cover more of the old object boundary; feathering only softens the new boundary. This is useful when a normal feather leaves a halo from the original layer.

For a hard edge, set `mask_expand_px = 0`, `feather_radius = 0`, and `border_blending = 0`.

## Crop-canvas rectangular mask

Use:

```text
mode = overlay_by_mask
use_crop_canvas_mask = true
```

The mask is rectangular regardless of the object silhouette. Historical widget names still use `square_mask_*`, but the mask works with any aspect ratio.

| Parameter | Default | Description |
|---|---:|---|
| `square_mask_inset_left_px` / `right_px` | `8` | Move the fully active rectangle inward from the side. |
| `square_mask_inset_top_px` / `bottom_px` | `8` | Move it inward from the top/bottom. |
| `square_mask_fade_left_px` / `right_px` | `16` | Fade width on the side. |
| `square_mask_fade_top_px` / `bottom_px` | `16` | Fade width on the top/bottom. |
| `square_mask_units` | `crop_pixels` | `crop_pixels` preserves legacy behavior; `source_pixels` compensates for affine zoom in the exact path. |

`inset` sets where the fully active rectangle begins. `fade` sets the transition width from alpha `0` to `1`; they are independent controls.

## Color matching and detection dropouts

Color matching is available in the affine `crop_metadata` path:

- `color_match_mode`: `off`, `mean`, `mean_std`, or `luminance`;
- `color_match_strength`: correction strength from `0.0` to `1.0`.

With face metadata, `undetected_frames` controls frames whose face detection is not valid:

- `fade_out` smoothly reduces the inserted contribution near a dropout;
- `skip` leaves those frames unchanged;
- `composite_anyway` inserts every frame using the interpolated trajectory.

`dropout_fade_window` defaults to `9` and affects `fade_out` only.

## Uncrop performance

`uncrop_chunk_size` defaults to `4`. `uncrop_memory_limit_mb` defaults to `512` and can reduce the effective chunk automatically. `crop_rescale` defaults to `1.0` and applies to the legacy/bbox path; it does not change the current affine path.

Full-frame affine warps need more memory than Crop. Reduce the chunk size when processing high-resolution video.

## Recommended connections

Exact path:

```text
Crop.cropped_images ─► processing ─► Uncrop.cropped_images
Crop.crop_metadata  ─────────────────► Uncrop.crop_metadata
source frames       ─────────────────► Uncrop.base_images
Crop.cropped_masks  ─────────────────► Uncrop.crop_masks
```

BOUNDING_BOX path:

```text
Crop.bboxes ─────────► Uncrop.bboxes
source frames ───────► Uncrop.base_images
processed crop ──────► Uncrop.cropped_images
```

## Troubleshooting

### Crop jitters

Enable `smooth_center`, use `gaussian`, increase `center_smooth_window`, or lower `center_smoothing_strength`. For several people, use `identity_reference`.

### Crop lags behind a fast subject

Reduce `center_smooth_window` and increase `center_smoothing_strength`. `ema` or `none` gives the least temporal lag.

### Zoom breathes

Enable `smooth_zoom`, increase `size_smooth_window`, lower `zoom_smoothing_strength`, and try `size_metric = height` for faces.

### Crop leaves a halo

In mask mode, increase `mask_expand_px` first and then tune `feather_radius`. Blur alone does not enlarge the replacement area.

### Uncrop has rectangular seams

Use `overlay_by_mask`, tune fade per side, try `square_mask_units = source_pixels` when zoom changes, and use a mild color-match mode.

### Face detection fails or selects the wrong person

Check the detector location, lower `face_confidence`, use an appropriate `fallback_detector`, connect a clear single-face `identity_reference`, and inspect the `report` identity/continuity counters.

### Out of memory

Lower `crop_chunk_size`, `uncrop_chunk_size`, or `uncrop_memory_limit_mb`. For face mode use CPU face models, `buffalo_s`, and `keep_face_models_loaded = false`.

## Workflow compatibility

The frontend migrates saved workflows from the former Crop By Mask Or Face node to the universal Crop node. It also migrates `output_long_side` to `output_resolution_side` and `use_square_mask` to `use_crop_canvas_mask`.

After updating the custom node, restart ComfyUI and refresh the browser so the node schema and dynamic widgets are reloaded.
