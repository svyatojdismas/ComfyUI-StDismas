import math
import torch
import torch.nn.functional as F

try:
    from comfy.utils import common_upscale
except Exception:
    common_upscale = None

MAX_RESOLUTION = 16384
ASPECT_RATIO_CHOICES = [
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "4:5",
    "5:4",
    "2:3",
    "3:2",
]


def _ensure_mask_hw(mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    mask: (B,H,W) or (B,H,W,1) or (B,1,H,W)
    returns (B,H,W) float in [0,1]
    """
    if mask is None:
        return None
    if mask.dim() == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask[:, 0, :, :]
    if mask.dim() != 3:
        raise ValueError(f"MASK must be (B,H,W) or (B,H,W,1) or (B,1,H,W), got {tuple(mask.shape)}")

    BM, HM, WM = mask.shape
    if (HM != H) or (WM != W):
        mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode="nearest-exact").squeeze(1)
    return mask.clamp(0.0, 1.0)


def _mask_bbox(mask2d: torch.Tensor):
    """
    mask2d: (H,W)
    returns (min_x, min_y, max_x_excl, max_y_excl) or None if empty

    This version avoids torch.nonzero() over every positive pixel. It only finds
    occupied rows/columns, which is much lighter for large filled masks.
    """
    m = mask2d > 0
    rows = torch.any(m, dim=1)
    cols = torch.any(m, dim=0)

    if not bool(rows.any().item()) or not bool(cols.any().item()):
        return None

    y_idx = torch.where(rows)[0]
    x_idx = torch.where(cols)[0]
    min_y = int(y_idx[0].item())
    max_y = int(y_idx[-1].item()) + 1
    min_x = int(x_idx[0].item())
    max_x = int(x_idx[-1].item()) + 1
    return (min_x, min_y, max_x, max_y)


def _choose_upscale_method(in_w, in_h, out_w, out_h):
    if out_w <= in_w and out_h <= in_h:
        return "lanczos"
    return "bicubic"


def _resize_image(img_hwc: torch.Tensor, out_w: int, out_h: int) -> torch.Tensor:
    """
    img_hwc: (h,w,3)
    returns (out_h,out_w,3)
    """
    if out_w <= 0 or out_h <= 0:
        raise ValueError("Invalid output size for resize")

    in_h, in_w, c = img_hwc.shape
    if in_h == out_h and in_w == out_w:
        return img_hwc

    if common_upscale is None:
        x = img_hwc.permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return x.squeeze(0).permute(1, 2, 0)

    method = _choose_upscale_method(in_w, in_h, out_w, out_h)
    x = img_hwc.permute(2, 0, 1).unsqueeze(0)
    x = common_upscale(x, out_w, out_h, method, "disabled")
    return x.squeeze(0).permute(1, 2, 0)


def _resize_mask(mask_hw: torch.Tensor, out_w: int, out_h: int) -> torch.Tensor:
    """
    mask_hw: (h,w)
    returns (out_h,out_w)
    """
    in_h, in_w = mask_hw.shape
    if in_h == out_h and in_w == out_w:
        return mask_hw
    x = mask_hw.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(out_h, out_w), mode="nearest")
    return x.squeeze(0).squeeze(0)


def _parse_aspect_ratio(ratio_str: str) -> float:
    parts = ratio_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid aspect ratio '{ratio_str}'")
    w = float(parts[0])
    h = float(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid aspect ratio '{ratio_str}'")
    return w / h


def _compute_crop_size(output_side: int, aspect_ratio: float, use_long_side: bool = True, divisible_by: int = 1) -> tuple[int, int]:
    """
    Compute output crop size.
    - use_long_side=True: output_side controls the long side
    - use_long_side=False: output_side controls the short side
    - if divisible_by > 1, both output sides are snapped to multiples of that value
      while keeping the aspect ratio only approximately (close, not exact).
    """
    target = max(1, int(output_side))
    d = max(1, int(divisible_by))

    if aspect_ratio >= 1.0:
        # landscape / square
        if use_long_side:
            pref_w = float(target)
            pref_h = float(target) / aspect_ratio
            drive = "w"
        else:
            pref_h = float(target)
            pref_w = float(target) * aspect_ratio
            drive = "h"
    else:
        # portrait
        if use_long_side:
            pref_h = float(target)
            pref_w = float(target) * aspect_ratio
            drive = "h"
        else:
            pref_w = float(target)
            pref_h = float(target) / aspect_ratio
            drive = "w"

    if d <= 1:
        return max(1, int(round(pref_w))), max(1, int(round(pref_h)))

    def q(v: float) -> int:
        return max(d, int(round(v / d)) * d)

    candidates = []
    if drive == "w":
        base = q(pref_w)
        for delta in range(-4 * d, 4 * d + 1, d):
            w = max(d, base + delta)
            h = q(w / aspect_ratio)
            candidates.append((w, h))
    else:
        base = q(pref_h)
        for delta in range(-4 * d, 4 * d + 1, d):
            h = max(d, base + delta)
            w = q(h * aspect_ratio)
            candidates.append((w, h))

    # Also add direct rounding candidates around the preferred pair.
    pw = q(pref_w)
    ph = q(pref_h)
    candidates.extend([(pw, ph), (max(d, pw - d), ph), (pw, max(d, ph - d)), (pw + d, ph), (pw, ph + d)])

    best = None
    best_score = None
    target_side = float(target)
    for w, h in candidates:
        if w <= 0 or h <= 0:
            continue
        ratio_err = abs((float(w) / float(h)) - aspect_ratio)
        side = max(w, h) if use_long_side else min(w, h)
        side_err = abs(float(side) - target_side)
        area_err = abs(float(w) - pref_w) + abs(float(h) - pref_h)
        score = (side_err, ratio_err, area_err)
        if best_score is None or score < best_score:
            best_score = score
            best = (int(w), int(h))

    return best


def _snap_dimension_to_divisible(value: int, divisible_by: int) -> int:
    d = max(1, int(divisible_by))
    v = max(1, int(value))
    if d <= 1:
        return v
    return max(d, int(round(v / d)) * d)


def _affine_forward_matrix(scale: float, cx: float, cy: float, crop_w: int, crop_h: int):
    uc = crop_w * 0.5
    vc = crop_h * 0.5
    tx = uc - scale * cx
    ty = vc - scale * cy
    return [
        [float(scale), 0.0, float(tx)],
        [0.0, float(scale), float(ty)],
    ]


def _affine_inverse_matrix(scale: float, cx: float, cy: float, crop_w: int, crop_h: int):
    uc = crop_w * 0.5
    vc = crop_h * 0.5
    tx = uc - scale * cx
    ty = vc - scale * cy
    inv_s = 1.0 / scale if scale != 0 else 0.0
    return [
        [float(inv_s), 0.0, float(-tx * inv_s)],
        [0.0, float(inv_s), float(-ty * inv_s)],
    ]


def _make_pixel_grid(out_w: int, out_h: int, device, dtype):
    ys = torch.arange(out_h, device=device, dtype=dtype)
    xs = torch.arange(out_w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def _normalize_grid(src_x: torch.Tensor, src_y: torch.Tensor, in_w: int, in_h: int):
    if in_w > 1:
        src_x = (src_x + 0.5) / in_w * 2.0 - 1.0
    else:
        src_x = torch.zeros_like(src_x)
    if in_h > 1:
        src_y = (src_y + 0.5) / in_h * 2.0 - 1.0
    else:
        src_y = torch.zeros_like(src_y)
    return src_x, src_y


def _build_affine_grid_from_pixel_grid(affine_2x3, grid_x: torch.Tensor, grid_y: torch.Tensor, in_w: int, in_h: int):
    a, b, tx = affine_2x3[0]
    c, d, ty = affine_2x3[1]

    src_x = a * grid_x + b * grid_y + tx
    src_y = c * grid_x + d * grid_y + ty
    src_x, src_y = _normalize_grid(src_x, src_y, in_w, in_h)
    grid = torch.stack((src_x, src_y), dim=-1)
    return grid.unsqueeze(0)


def _build_affine_grid_batch(affines_2x3: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor, in_w: int, in_h: int):
    """
    affines_2x3: (N,2,3)
    returns grid: (N,H,W,2)
    """
    gx = grid_x.unsqueeze(0)
    gy = grid_y.unsqueeze(0)

    a = affines_2x3[:, 0, 0].view(-1, 1, 1)
    b = affines_2x3[:, 0, 1].view(-1, 1, 1)
    tx = affines_2x3[:, 0, 2].view(-1, 1, 1)
    c = affines_2x3[:, 1, 0].view(-1, 1, 1)
    d = affines_2x3[:, 1, 1].view(-1, 1, 1)
    ty = affines_2x3[:, 1, 2].view(-1, 1, 1)

    src_x = a * gx + b * gy + tx
    src_y = c * gx + d * gy + ty
    src_x, src_y = _normalize_grid(src_x, src_y, in_w, in_h)
    return torch.stack((src_x, src_y), dim=-1)


def _build_affine_grid(affine_2x3, out_w: int, out_h: int, in_w: int, in_h: int, device, dtype):
    grid_x, grid_y = _make_pixel_grid(out_w, out_h, device=device, dtype=dtype)
    return _build_affine_grid_from_pixel_grid(affine_2x3, grid_x, grid_y, in_w, in_h)


def _feather_alpha(alpha_hw: torch.Tensor, feather_px: int) -> torch.Tensor:
    """
    alpha_hw: (h,w) in [0,1]
    feather_px: blur radius in pixels
    """
    if feather_px <= 0:
        return alpha_hw.clamp(0.0, 1.0)

    k = feather_px * 2 + 1
    x = alpha_hw.unsqueeze(0).unsqueeze(0)
    pad = feather_px
    x = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=k, stride=1)
    x = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=k, stride=1)
    return x.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def _make_square_alpha(
    height: int,
    width: int,
    inset_px: int = 0,
    inset_left_px: int | None = None,
    inset_right_px: int | None = None,
    inset_top_px: int | None = None,
    inset_bottom_px: int | None = None,
    fade_left_px: int = 0,
    fade_right_px: int = 0,
    fade_top_px: int = 0,
    fade_bottom_px: int = 0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Build a rectangular alpha mask with independent inset and fade for each side.

    inset_* moves the fully-active rectangle inward from each crop border.
    fade_* controls how softly that side fades from 1 to 0.
    """
    h = max(1, int(height))
    w = max(1, int(width))

    # Backward-compatible common inset: used only when per-side inset is not passed.
    common_inset = max(0, int(inset_px))
    left_inset = max(0, int(common_inset if inset_left_px is None else inset_left_px))
    right_inset = max(0, int(common_inset if inset_right_px is None else inset_right_px))
    top_inset = max(0, int(common_inset if inset_top_px is None else inset_top_px))
    bottom_inset = max(0, int(common_inset if inset_bottom_px is None else inset_bottom_px))

    # Clamp paired insets so they cannot invert the active rectangle.
    if left_inset + right_inset >= w:
        scale = max(0.0, float(w - 1) / max(1.0, float(left_inset + right_inset)))
        left_inset = int(left_inset * scale)
        right_inset = int(right_inset * scale)
        while left_inset + right_inset >= w and right_inset > 0:
            right_inset -= 1
        while left_inset + right_inset >= w and left_inset > 0:
            left_inset -= 1

    if top_inset + bottom_inset >= h:
        scale = max(0.0, float(h - 1) / max(1.0, float(top_inset + bottom_inset)))
        top_inset = int(top_inset * scale)
        bottom_inset = int(bottom_inset * scale)
        while top_inset + bottom_inset >= h and bottom_inset > 0:
            bottom_inset -= 1
        while top_inset + bottom_inset >= h and top_inset > 0:
            top_inset -= 1

    fade_left = max(0, int(fade_left_px))
    fade_right = max(0, int(fade_right_px))
    fade_top = max(0, int(fade_top_px))
    fade_bottom = max(0, int(fade_bottom_px))

    alpha = torch.zeros((h, w), device=device, dtype=dtype)
    x0 = int(left_inset)
    x1 = int(w - right_inset)
    y0 = int(top_inset)
    y1 = int(h - bottom_inset)

    if x1 <= x0 or y1 <= y0:
        return alpha

    alpha[y0:y1, x0:x1] = 1.0

    if fade_left <= 0 and fade_right <= 0 and fade_top <= 0 and fade_bottom <= 0:
        return alpha

    y = torch.arange(h, device=device, dtype=dtype).view(h, 1)
    x = torch.arange(w, device=device, dtype=dtype).view(1, w)

    if fade_left > 0:
        left_dist = x - float(x0)
        fade_left_t = ((left_dist + 1.0) / float(fade_left)).clamp(0.0, 1.0)
    else:
        fade_left_t = torch.ones((1, w), device=device, dtype=dtype)

    if fade_right > 0:
        right_dist = float(x1 - 1) - x
        fade_right_t = ((right_dist + 1.0) / float(fade_right)).clamp(0.0, 1.0)
    else:
        fade_right_t = torch.ones((1, w), device=device, dtype=dtype)

    if fade_top > 0:
        top_dist = y - float(y0)
        fade_top_t = ((top_dist + 1.0) / float(fade_top)).clamp(0.0, 1.0)
    else:
        fade_top_t = torch.ones((h, 1), device=device, dtype=dtype)

    if fade_bottom > 0:
        bottom_dist = float(y1 - 1) - y
        fade_bottom_t = ((bottom_dist + 1.0) / float(fade_bottom)).clamp(0.0, 1.0)
    else:
        fade_bottom_t = torch.ones((h, 1), device=device, dtype=dtype)

    fade_t = fade_left_t * fade_right_t * fade_top_t * fade_bottom_t
    return (alpha * fade_t).clamp(0.0, 1.0)


def _fit_crop_to_frame_bounds(
    cx: float,
    cy: float,
    scale: float,
    crop_w: int,
    crop_h: int,
    frame_w: int,
    frame_h: int,
):
    """
    Adjust crop center/scale so the source-space crop window stays fully inside the frame
    while preserving the requested output aspect ratio.
    """
    fit_scale = max(float(scale), float(crop_w) / max(1.0, float(frame_w)), float(crop_h) / max(1.0, float(frame_h)))

    win_w = float(crop_w) / fit_scale
    win_h = float(crop_h) / fit_scale

    half_w = win_w * 0.5
    half_h = win_h * 0.5

    min_cx = half_w
    max_cx = float(frame_w) - half_w
    min_cy = half_h
    max_cy = float(frame_h) - half_h

    if min_cx > max_cx:
        cx_fit = float(frame_w) * 0.5
    else:
        cx_fit = min(max(float(cx), min_cx), max_cx)

    if min_cy > max_cy:
        cy_fit = float(frame_h) * 0.5
    else:
        cy_fit = min(max(float(cy), min_cy), max_cy)

    return float(cx_fit), float(cy_fit), float(fit_scale)


def _draw_crop_visualize(image_hwc: torch.Tensor, cx: float, cy: float, scale: float, crop_w: int, crop_h: int) -> torch.Tensor:
    """
    Fast visualization of crop bounds on the full-resolution frame.
    Draws a red rectangular stroke using tensor slicing only.
    """
    H, W, C = image_hwc.shape
    if C < 3:
        return image_hwc

    win_w = float(crop_w) / float(scale) if scale != 0 else float(W)
    win_h = float(crop_h) / float(scale) if scale != 0 else float(H)

    x0 = int(round(cx - win_w * 0.5))
    y0 = int(round(cy - win_h * 0.5))
    x1 = int(round(cx + win_w * 0.5))
    y1 = int(round(cy + win_h * 0.5))

    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W))
    y1 = max(y0 + 1, min(y1, H))

    stroke = max(1, min(4, int(round(min(H, W) / 512.0))))
    out = image_hwc.clone()

    # red stroke, preserve visibility by zeroing G/B on the border
    out[y0:min(y0 + stroke, y1), x0:x1, 0] = 1.0
    out[y0:min(y0 + stroke, y1), x0:x1, 1:] = 0.0

    out[max(y1 - stroke, y0):y1, x0:x1, 0] = 1.0
    out[max(y1 - stroke, y0):y1, x0:x1, 1:] = 0.0

    out[y0:y1, x0:min(x0 + stroke, x1), 0] = 1.0
    out[y0:y1, x0:min(x0 + stroke, x1), 1:] = 0.0

    out[y0:y1, max(x1 - stroke, x0):x1, 0] = 1.0
    out[y0:y1, max(x1 - stroke, x0):x1, 1:] = 0.0

    return out


class BatchImageCropByMaskAdvanced_StDismas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image batch (B,H,W,C)"}),
                "crop_mask": ("MASK", {"tooltip": "Main mask used to compute crop region"}),
                "aspect_ratio": (ASPECT_RATIO_CHOICES, {"default": "16:9", "tooltip": "Output aspect ratio"}),
                "output_long_side": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 1, "tooltip": "Target size of the selected output side in pixels"}),
                "use_long_side": ("BOOLEAN", {"default": True, "tooltip": "If enabled, output_long_side controls the long side; if disabled, it controls the short side"}),
                "use_custom_resolution": ("BOOLEAN", {"default": False, "tooltip": "If enabled, use custom width and height for the crop output instead of aspect_ratio/output_long_side"}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1, "tooltip": "Custom crop output width when use_custom_resolution is enabled"}),
                "height": ("INT", {"default": 576, "min": 1, "max": 8192, "step": 1, "tooltip": "Custom crop output height when use_custom_resolution is enabled"}),
                "margin_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Expands mask bbox before cropping"}),
                "smooth_center": ("BOOLEAN", {"default": True, "tooltip": "Enable temporal smoothing for crop center movement"}),
                "center_smoothing_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Center smoothing strength (0 = locked to previous, 1 = follow current center)"}),
                "smooth_zoom": ("BOOLEAN", {"default": True, "tooltip": "Enable temporal smoothing for zoom changes"}),
                "zoom_smoothing_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Zoom smoothing strength (0 = locked to previous, 1 = follow current zoom)"}),
                "offset_x": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip": "Horizontal offset from mask center in source pixels"}),
                "offset_y": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1, "tooltip": "Vertical offset from mask center in source pixels"}),
                "min_zoom": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Minimum zoom limit"}),
                "max_zoom": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01, "tooltip": "Maximum zoom limit"}),
                "interpolation": (["bilinear", "bicubic"], {"default": "bilinear", "tooltip": "Sampling method for image crop"}),
                "fit_frame_bounds": ("BOOLEAN", {"default": False, "tooltip": "Keep the crop window fully inside the source frame while preserving aspect ratio"}),
                "divisible_by": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1, "tooltip": "Make both output crop dimensions divisible by this value"}),
                "enable_visualize": ("BOOLEAN", {"default": False, "tooltip": "Draw full-frame crop preview. Disable for better speed and much lower memory use on video batches."}),
                "crop_chunk_size": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1, "tooltip": "How many frames are sampled per grid_sample batch. Lower uses less memory; higher can be faster."}),
            },
            "optional": {
                "masks": ("MASK", {"tooltip": "Optional extra mask that is cropped with the same transform but does not affect crop computation"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "BBOXES")
    RETURN_NAMES = ("cropped_images", "cropped_masks", "masks", "visualize", "crop_metadata")
    FUNCTION = "crop"
    CATEGORY = "Comfyui-StDismas/masking"

    def crop(
        self,
        images,
        crop_mask,
        aspect_ratio,
        output_long_side,
        use_long_side,
        use_custom_resolution,
        width,
        height,
        margin_scale,
        smooth_center,
        center_smoothing_strength,
        smooth_zoom,
        zoom_smoothing_strength,
        offset_x,
        offset_y,
        min_zoom,
        max_zoom,
        interpolation,
        fit_frame_bounds,
        divisible_by,
        enable_visualize=False,
        crop_chunk_size=16,
        masks=None,
    ):
        B, H, W, C = images.shape
        if crop_mask.shape[0] != B:
            raise ValueError(f"Batch size mismatch: images={B}, crop_mask={crop_mask.shape[0]}")
        crop_mask = _ensure_mask_hw(crop_mask, H, W)

        has_extra_masks = masks is not None
        if has_extra_masks:
            if masks.shape[0] != B:
                raise ValueError(f"Batch size mismatch: images={B}, masks={masks.shape[0]}")
            masks = _ensure_mask_hw(masks, H, W)

        device = images.device
        dtype = images.dtype
        chunk_size = max(1, int(crop_chunk_size))

        ratio = _parse_aspect_ratio(aspect_ratio)

        if use_custom_resolution:
            crop_w = _snap_dimension_to_divisible(width, divisible_by)
            crop_h = _snap_dimension_to_divisible(height, divisible_by)
        else:
            crop_w, crop_h = _compute_crop_size(
                output_long_side,
                ratio,
                use_long_side=use_long_side,
                divisible_by=divisible_by,
            )

        # Preallocate outputs to avoid list -> stack memory spikes.
        out_imgs = torch.empty((B, crop_h, crop_w, C), device=device, dtype=dtype)
        out_crop_masks = torch.empty((B, crop_h, crop_w), device=device, dtype=dtype)
        if has_extra_masks:
            out_masks = torch.empty((B, crop_h, crop_w), device=device, dtype=dtype)
        else:
            out_masks = None

        out_frames = []
        inverse_affines = []
        centers_scales = []

        prev_center = None
        prev_bbox = None
        prev_scale = None
        margin_eff = max(float(margin_scale), 1.0)

        # Pass 1: compute bbox, smoothing, metadata. Kept sequential by design.
        for i in range(B):
            crop_mask_i = crop_mask[i]
            bb = _mask_bbox(crop_mask_i)
            if bb is None:
                if prev_bbox is not None:
                    min_x, min_y, max_x, max_y = prev_bbox
                else:
                    default_size = max(1.0, min(W, H) * 0.25)
                    cx0 = W * 0.5
                    cy0 = H * 0.5
                    min_x = cx0 - default_size * 0.5
                    max_x = cx0 + default_size * 0.5
                    min_y = cy0 - default_size * 0.5
                    max_y = cy0 + default_size * 0.5
                    min_x, min_y, max_x, max_y = float(min_x), float(min_y), float(max_x), float(max_y)
            else:
                min_x, min_y, max_x, max_y = bb
                min_x, min_y, max_x, max_y = float(min_x), float(min_y), float(max_x), float(max_y)

            bbox_w = max(1.0, max_x - min_x)
            bbox_h = max(1.0, max_y - min_y)
            cx = (min_x + max_x) * 0.5 + float(offset_x)
            cy = (min_y + max_y) * 0.5 + float(offset_y)

            bbox_w_exp = bbox_w * margin_eff
            bbox_h_exp = bbox_h * margin_eff

            sx = crop_w / bbox_w_exp
            sy = crop_h / bbox_h_exp
            scale = min(sx, sy)
            scale = max(float(min_zoom), min(float(max_zoom), scale))

            if smooth_center and prev_center is not None:
                a = float(center_smoothing_strength)
                cx = prev_center[0] * (1.0 - a) + cx * a
                cy = prev_center[1] * (1.0 - a) + cy * a
            if smooth_zoom and prev_scale is not None:
                a = float(zoom_smoothing_strength)
                scale = prev_scale * (1.0 - a) + scale * a

            if fit_frame_bounds:
                cx, cy, scale = _fit_crop_to_frame_bounds(
                    cx=cx,
                    cy=cy,
                    scale=scale,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    frame_w=W,
                    frame_h=H,
                )

            forward_affine = _affine_forward_matrix(scale, cx, cy, crop_w, crop_h)
            inverse_affine = _affine_inverse_matrix(scale, cx, cy, crop_w, crop_h)
            inverse_affines.append(inverse_affine)
            centers_scales.append((cx, cy, scale))

            bbox_exp = [
                float(cx - bbox_w_exp * 0.5),
                float(cy - bbox_h_exp * 0.5),
                float(cx + bbox_w_exp * 0.5),
                float(cy + bbox_h_exp * 0.5),
            ]
            out_frames.append({
                "orig_size": [int(W), int(H)],
                "crop_size": [int(crop_w), int(crop_h)],
                "S": float(scale),
                "center": [float(cx), float(cy)],
                "offset": [float(offset_x), float(offset_y)],
                "fit_frame_bounds": bool(fit_frame_bounds),
                "divisible_by": int(divisible_by),
                "use_long_side": bool(use_long_side),
                "use_custom_resolution": bool(use_custom_resolution),
                "forward_affine_2x3": forward_affine,
                "inverse_affine_2x3": inverse_affine,
                "mask_bbox": [float(min_x), float(min_y), float(max_x), float(max_y)],
                "mask_bbox_exp": bbox_exp,
            })

            prev_center = (cx, cy)
            prev_bbox = (min_x, min_y, max_x, max_y)
            prev_scale = scale

        # Pass 2: sample images/masks in chunks. This keeps the exact same metadata contract,
        # but avoids per-frame Python grid_sample calls and avoids building one huge video grid.
        base_grid_x, base_grid_y = _make_pixel_grid(crop_w, crop_h, device=device, dtype=dtype)
        for start in range(0, B, chunk_size):
            end = min(B, start + chunk_size)
            affines = torch.tensor(inverse_affines[start:end], device=device, dtype=dtype)
            grid = _build_affine_grid_batch(affines, base_grid_x, base_grid_y, W, H)

            img_nchw = images[start:end].permute(0, 3, 1, 2)
            sampled_imgs = F.grid_sample(
                img_nchw,
                grid,
                mode=interpolation,
                padding_mode="zeros",
                align_corners=False,
            )
            out_imgs[start:end] = sampled_imgs.permute(0, 2, 3, 1)

            crop_mask_nchw = crop_mask[start:end].unsqueeze(1)
            sampled_crop_masks = F.grid_sample(
                crop_mask_nchw,
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            )
            out_crop_masks[start:end] = sampled_crop_masks.squeeze(1)

            if has_extra_masks:
                extra_mask_nchw = masks[start:end].unsqueeze(1)
                sampled_extra_masks = F.grid_sample(
                    extra_mask_nchw,
                    grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )
                out_masks[start:end] = sampled_extra_masks.squeeze(1)

        if not has_extra_masks:
            # Do not allocate/copy a duplicate output. The secondary masks output mirrors cropped_masks.
            out_masks = out_crop_masks

        if enable_visualize:
            out_visualize = torch.empty_like(images)
            for i, (cx, cy, scale) in enumerate(centers_scales):
                out_visualize[i] = _draw_crop_visualize(
                    images[i],
                    cx=cx,
                    cy=cy,
                    scale=scale,
                    crop_w=crop_w,
                    crop_h=crop_h,
                )
        else:
            # Preserve the output slot without cloning the whole full-frame video batch.
            out_visualize = images

        metadata = {
            "version": "crop_by_mask_v2",
            "frames": out_frames,
        }

        return (out_imgs, out_crop_masks, out_masks, out_visualize, metadata)


class BatchImageUncropByMaskAdvanced_StDismas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cropped_images": ("IMAGE", {"tooltip": "Cropped image batch to place back into original frame"}),
                "crop_metadata": ("BBOXES", {"tooltip": "Metadata produced by the crop node"}),
                "mode": (["overlay_full", "overlay_by_mask"], {"default": "overlay_full", "tooltip": "Overlay full crop or blend using crop mask"}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend strength for uncrop result"}),
            },
            "optional": {
                "base_images": ("IMAGE", {"tooltip": "Base image batch to composite onto"}),
                "original_images": ("IMAGE", {"tooltip": "Legacy alias for base_images"}),
                "crop_masks": ("MASK", {"tooltip": "Mask used when mode is overlay_by_mask and use_square_mask is disabled"}),
                "border_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Legacy feather control; used if feather_radius is 0"}),
                "feather_radius": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, "tooltip": "Edge feathering radius in pixels"}),
                "crop_rescale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01, "tooltip": "Scale cropped patch before placing in legacy bbox mode"}),
                "use_square_mask": ("BOOLEAN", {"default": True, "tooltip": "Use rectangular patch compositing instead of crop mask alpha during uncrop"}),
                "square_mask_inset_left_px": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1, "tooltip": "Inset square composite mask from the left crop border"}),
                "square_mask_inset_right_px": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1, "tooltip": "Inset square composite mask from the right crop border"}),
                "square_mask_inset_top_px": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1, "tooltip": "Inset square composite mask from the top crop border"}),
                "square_mask_inset_bottom_px": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1, "tooltip": "Inset square composite mask from the bottom crop border"}),
                "square_mask_fade_left_px": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1, "tooltip": "Fade width for the left edge of square composite mask"}),
                "square_mask_fade_right_px": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1, "tooltip": "Fade width for the right edge of square composite mask"}),
                "square_mask_fade_top_px": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1, "tooltip": "Fade width for the top edge of square composite mask"}),
                "square_mask_fade_bottom_px": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1, "tooltip": "Fade width for the bottom edge of square composite mask"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "uncrop"
    CATEGORY = "Comfyui-StDismas/masking"

    def uncrop(
        self,
        cropped_images,
        crop_metadata,
        mode="overlay_full",
        blend=1.0,
        base_images=None,
        original_images=None,
        crop_masks=None,
        border_blending=0.25,
        feather_radius=0,
        crop_rescale=1.0,
        use_square_mask=True,
        square_mask_inset_left_px=8,
        square_mask_inset_right_px=8,
        square_mask_inset_top_px=8,
        square_mask_inset_bottom_px=8,
        square_mask_fade_left_px=16,
        square_mask_fade_right_px=16,
        square_mask_fade_top_px=16,
        square_mask_fade_bottom_px=16,
    ):
        if base_images is None and original_images is not None:
            base_images = original_images

        Bc, Hc, Wc, Cc = cropped_images.shape
        if base_images is not None:
            B, H, W, C = base_images.shape
        else:
            B = Bc
            H = None
            W = None
            C = None

        if Bc != B:
            raise ValueError(f"Batch size mismatch: base_images={B}, cropped_images={Bc}")
        if crop_masks is not None and crop_masks.shape[0] != B:
            raise ValueError(f"Batch size mismatch: cropped_images={Bc}, crop_masks={crop_masks.shape[0]}")

        if crop_masks is not None:
            crop_masks = _ensure_mask_hw(crop_masks, Hc, Wc)

        feather_px = int(feather_radius)
        if feather_px <= 0:
            feather_px = int(round(float(border_blending) * 32.0))

        if isinstance(crop_metadata, dict) and crop_metadata.get("version") == "crop_by_mask_v2":
            frames = crop_metadata.get("frames", [])
            if len(frames) != B:
                raise ValueError(f"crop_metadata frames must match batch size {B}, got {len(frames)}")

            device = cropped_images.device
            dtype = cropped_images.dtype

            if base_images is None:
                first = frames[0]
                W = int(first["orig_size"][0])
                H = int(first["orig_size"][1])
                out = torch.zeros((B, H, W, Cc), device=device, dtype=dtype)
            else:
                first = frames[0]
                if int(first["orig_size"][0]) != W or int(first["orig_size"][1]) != H:
                    raise ValueError("base_images size must match crop_metadata orig_size.")
                out = base_images.clone()

            square_alpha_cache = {}
            ones_alpha_cache = {}

            for i in range(B):
                frame = frames[i]
                orig_w, orig_h = frame["orig_size"]
                crop_w, crop_h = frame["crop_size"]
                forward_affine = frame["forward_affine_2x3"]

                grid = _build_affine_grid(
                    forward_affine,
                    orig_w,
                    orig_h,
                    crop_w,
                    crop_h,
                    device=device,
                    dtype=dtype,
                )

                patch = cropped_images[i].permute(2, 0, 1).unsqueeze(0)
                warped = F.grid_sample(
                    patch,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

                if mode == "overlay_by_mask":
                    if use_square_mask:
                        key = (
                            crop_h,
                            crop_w,
                            int(square_mask_inset_left_px),
                            int(square_mask_inset_right_px),
                            int(square_mask_inset_top_px),
                            int(square_mask_inset_bottom_px),
                            int(square_mask_fade_left_px),
                            int(square_mask_fade_right_px),
                            int(square_mask_fade_top_px),
                            int(square_mask_fade_bottom_px),
                        )
                        alpha_patch = square_alpha_cache.get(key)
                        if alpha_patch is None:
                            alpha_patch = _make_square_alpha(
                                crop_h,
                                crop_w,
                                inset_left_px=square_mask_inset_left_px,
                                inset_right_px=square_mask_inset_right_px,
                                inset_top_px=square_mask_inset_top_px,
                                inset_bottom_px=square_mask_inset_bottom_px,
                                fade_left_px=square_mask_fade_left_px,
                                fade_right_px=square_mask_fade_right_px,
                                fade_top_px=square_mask_fade_top_px,
                                fade_bottom_px=square_mask_fade_bottom_px,
                                device=device,
                                dtype=dtype,
                            ).unsqueeze(0).unsqueeze(0)
                            square_alpha_cache[key] = alpha_patch
                    else:
                        if crop_masks is None:
                            raise ValueError("mode='overlay_by_mask' with use_square_mask=False requires crop_masks.")
                        alpha_patch = crop_masks[i].unsqueeze(0).unsqueeze(0)
                else:
                    # overlay_full should still only affect the valid crop rectangle, not the whole frame.
                    key = (crop_h, crop_w)
                    alpha_patch = ones_alpha_cache.get(key)
                    if alpha_patch is None:
                        alpha_patch = torch.ones((1, 1, crop_h, crop_w), device=device, dtype=dtype)
                        ones_alpha_cache[key] = alpha_patch

                warped_mask = F.grid_sample(
                    alpha_patch,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

                alpha = warped_mask.clamp(0.0, 1.0)
                if mode == "overlay_by_mask" and not use_square_mask:
                    alpha = _feather_alpha(alpha, feather_px)
                alpha = alpha * float(blend)
                alpha3 = alpha.unsqueeze(-1)
                out[i] = out[i] * (1.0 - alpha3) + warped * alpha3

            return (out.to(device=device, dtype=dtype),)

        if base_images is None:
            raise ValueError("Legacy crop_metadata requires base_images/original_images.")

        if crop_masks is None:
            raise ValueError("Legacy crop_metadata requires crop_masks.")

        cropped_masks = _ensure_mask_hw(crop_masks, Hc, Wc)

        if isinstance(crop_metadata, (list, tuple)):
            if len(crop_metadata) == 1 and B > 1:
                bboxes_use = [crop_metadata[0] for _ in range(B)]
            elif len(crop_metadata) == B:
                bboxes_use = list(crop_metadata)
            else:
                raise ValueError(f"legacy bboxes length must be 1 or B({B}), got {len(crop_metadata)}")
        else:
            bboxes_use = [crop_metadata for _ in range(B)]

        device = base_images.device
        dtype = base_images.dtype
        out = base_images.clone()

        square_alpha_cache = {}
        for i in range(B):
            info = bboxes_use[i]
            x0 = int(info["x0"]); y0 = int(info["y0"]); x1 = int(info["x1"]); y1 = int(info["y1"])
            win_w = int(info.get("win_w", x1 - x0))
            win_h = int(info.get("win_h", y1 - y0))

            x0 = max(0, min(x0, W))
            x1 = max(0, min(x1, W))
            y0 = max(0, min(y0, H))
            y1 = max(0, min(y1, H))

            win_w = max(1, x1 - x0)
            win_h = max(1, y1 - y0)

            tgt_w = max(1, int(round(win_w * float(crop_rescale))))
            tgt_h = max(1, int(round(win_h * float(crop_rescale))))

            patch = _resize_image(cropped_images[i], tgt_w, tgt_h)
            if use_square_mask:
                key = (
                    tgt_h,
                    tgt_w,
                    int(square_mask_inset_left_px),
                    int(square_mask_inset_right_px),
                    int(square_mask_inset_top_px),
                    int(square_mask_inset_bottom_px),
                    int(square_mask_fade_left_px),
                    int(square_mask_fade_right_px),
                    int(square_mask_fade_top_px),
                    int(square_mask_fade_bottom_px),
                )
                alpha = square_alpha_cache.get(key)
                if alpha is None:
                    alpha = _make_square_alpha(
                        tgt_h,
                        tgt_w,
                        inset_left_px=square_mask_inset_left_px,
                        inset_right_px=square_mask_inset_right_px,
                        inset_top_px=square_mask_inset_top_px,
                        inset_bottom_px=square_mask_inset_bottom_px,
                        fade_left_px=square_mask_fade_left_px,
                        fade_right_px=square_mask_fade_right_px,
                        fade_top_px=square_mask_fade_top_px,
                        fade_bottom_px=square_mask_fade_bottom_px,
                        device=device,
                        dtype=dtype,
                    )
                    square_alpha_cache[key] = alpha
            else:
                alpha = _resize_mask(cropped_masks[i], tgt_w, tgt_h).to(device=device, dtype=dtype)
                alpha = _feather_alpha(alpha, feather_px)

            dst_x0, dst_y0 = x0, y0
            dst_x1, dst_y1 = x1, y1

            if tgt_w != win_w or tgt_h != win_h:
                place_w = min(tgt_w, win_w)
                place_h = min(tgt_h, win_h)

                px0 = max(0, (tgt_w - place_w) // 2)
                py0 = max(0, (tgt_h - place_h) // 2)
                patch = patch[py0:py0 + place_h, px0:px0 + place_w, :]
                alpha = alpha[py0:py0 + place_h, px0:px0 + place_w]

                ox = (win_w - place_w) // 2
                oy = (win_h - place_h) // 2
                dst_x0 = x0 + ox
                dst_y0 = y0 + oy
                dst_x1 = dst_x0 + place_w
                dst_y1 = dst_y0 + place_h

            dst_x0 = max(0, min(dst_x0, W))
            dst_x1 = max(0, min(dst_x1, W))
            dst_y0 = max(0, min(dst_y0, H))
            dst_y1 = max(0, min(dst_y1, H))

            ph = dst_y1 - dst_y0
            pw = dst_x1 - dst_x0
            if ph <= 0 or pw <= 0:
                continue

            patch = patch[:ph, :pw, :]
            alpha = alpha[:ph, :pw]

            base = out[i, dst_y0:dst_y1, dst_x0:dst_x1, :]
            alpha3 = alpha.unsqueeze(-1).expand(-1, -1, Cc)
            out[i, dst_y0:dst_y1, dst_x0:dst_x1, :] = base * (1.0 - alpha3) + patch * alpha3

        return (out.to(device=device, dtype=dtype),)


NODE_CLASS_MAPPINGS = {
    "BatchImageCropByMaskAdvanced_StDismas": BatchImageCropByMaskAdvanced_StDismas,
    "BatchImageUncropByMaskAdvanced_StDismas": BatchImageUncropByMaskAdvanced_StDismas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageCropByMaskAdvanced_StDismas": "Batch Image Crop By Mask Advanced (StDismas)",
    "BatchImageUncropByMaskAdvanced_StDismas": "Batch Image Uncrop By Mask Advanced (StDismas)",
}
