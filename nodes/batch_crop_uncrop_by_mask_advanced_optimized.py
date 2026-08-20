import gc
import math
import os

import numpy as np
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

SMOOTHING_METHODS = ["gaussian", "savgol", "moving_average", "ema", "none"]
SIZE_METRICS = ["bbox_fit", "height", "width", "max_dimension", "area_sqrt"]
RESOLUTION_MODES = ["manual", "auto_no_downscale", "auto_capped"]
COLOR_MATCH_MODES = ["off", "mean", "mean_std", "luminance"]

_FACE_DETECTOR_CACHE = {}
_FACE_RECOGNISER_CACHE = {}


def _interrupt_if_requested():
    try:
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _face_detector_list():
    names = []
    try:
        import folder_paths

        for key in ("ultralytics_bbox", "ultralytics"):
            try:
                names.extend(folder_paths.get_filename_list(key))
            except Exception:
                pass
    except Exception:
        pass

    result = []
    seen = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result or ["face_yolov8m.pt"]


def _resolve_detector_path(name: str):
    try:
        import folder_paths
    except Exception as exc:
        raise RuntimeError("Face detection requires ComfyUI's folder_paths module.") from exc

    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            path = folder_paths.get_full_path(key, name)
        except Exception:
            path = None
        if path:
            return path

    base = getattr(folder_paths, "models_dir", "models")
    for subdir in ("ultralytics/bbox", "ultralytics", "ultralytics/segm"):
        candidate = os.path.join(base, *subdir.split("/"), name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Detector '{name}' was not found in models/ultralytics/bbox, "
        "models/ultralytics, or models/ultralytics/segm."
    )


def _load_face_detector(name: str, keep_loaded: bool):
    if keep_loaded and name in _FACE_DETECTOR_CACHE:
        return _FACE_DETECTOR_CACHE[name]
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Face tracking is optional, but requires a usable 'ultralytics' installation "
            f"when enabled. Import failed: {exc}"
        ) from exc
    model = YOLO(_resolve_detector_path(name))
    if keep_loaded:
        _FACE_DETECTOR_CACHE[name] = model
    return model


def _face_recogniser(pack: str, device: str, keep_loaded: bool):
    key = (pack, device)
    if keep_loaded and key in _FACE_RECOGNISER_CACHE:
        return _FACE_RECOGNISER_CACHE[key]
    try:
        import folder_paths
        import insightface
    except Exception as exc:
        raise RuntimeError(
            "identity_reference/identity_track requires the optional 'insightface' package."
        ) from exc

    providers = ["CPUExecutionProvider"]
    ctx_id = -1
    if device in ("auto", "cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0

    root = os.path.join(getattr(folder_paths, "models_dir", "models"), "insightface")
    app = insightface.app.FaceAnalysis(
        name=pack,
        root=root,
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    if keep_loaded:
        _FACE_RECOGNISER_CACHE[key] = app
    return app


def _release_optional_face_models(*models):
    for model in models:
        if model is None:
            continue
        try:
            model.to("cpu")
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_bgr_u8(image_hwc: torch.Tensor) -> np.ndarray:
    rgb = (image_hwc[..., :3].detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    return rgb[..., ::-1].copy()


def _embed_faces(app, bgr: np.ndarray):
    faces = []
    for face in app.get(bgr):
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            continue
        faces.append((face.bbox.tolist(), np.asarray(embedding, dtype=np.float32)))
    return faces


def _best_identity_match(candidates, reference_embedding):
    if not candidates or reference_embedding is None:
        return None, -1.0
    similarities = [float(np.dot(embedding, reference_embedding)) for _, embedding in candidates]
    index = int(np.argmax(similarities))
    return index, similarities[index]


def _box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = iw * ih
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def _continuity_cost(box, last):
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    size = box[3] - box[1]
    distance = math.hypot(cx - last[0], cy - last[1])
    return distance + abs(size - last[2]) * 2.0


def _predict_boxes(model, image_hwc, confidence: float, device: str):
    kwargs = {"conf": float(confidence), "verbose": False}
    if device != "auto":
        kwargs["device"] = device
    result = model.predict(_to_bgr_u8(image_hwc), **kwargs)[0]
    if not len(result.boxes):
        return [], [], []
    boxes = result.boxes.xyxy.tolist()
    confidences = (
        result.boxes.conf.tolist()
        if getattr(result.boxes, "conf", None) is not None
        else [1.0] * len(boxes)
    )
    classes = (
        result.boxes.cls.tolist()
        if getattr(result.boxes, "cls", None) is not None
        else [0] * len(boxes)
    )
    return boxes, confidences, classes


def _interp_gaps(values: np.ndarray, valid: np.ndarray, fallback: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.size == 0:
        return values
    if not valid.any():
        return np.full(values.shape, float(fallback), dtype=np.float64)
    indices = np.arange(values.size)
    return np.interp(indices, indices[valid], values[valid])


def _apply_trajectory_response(
    values: np.ndarray, filtered: np.ndarray, response_strength: float
) -> np.ndarray:
    """Follow a filtered trajectory using the legacy 0..1 response control."""
    alpha = min(1.0, max(0.0, float(response_strength)))
    result = np.asarray(filtered, dtype=np.float64).copy()
    result[0] = values[0]
    for i in range(1, len(result)):
        result[i] = result[i - 1] * (1.0 - alpha) + result[i] * alpha
    return result


def _smooth_trajectory(
    values: np.ndarray, window: int, method: str, response_strength: float
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if method == "none" or len(values) < 2:
        return values.copy()
    if method == "ema":
        return _apply_trajectory_response(values, values, response_strength)

    window = min(max(1, int(window)), len(values))
    if window % 2 == 0:
        window = max(1, window - 1)
    if window < 3:
        return _apply_trajectory_response(values, values, response_strength)
    pad = window // 2
    padded = np.pad(values, pad, mode="reflect")
    if method == "savgol":
        try:
            from scipy.signal import savgol_filter

            order = 2 if window > 3 else 1
            filtered = np.asarray(savgol_filter(padded, window, order))[pad:pad + len(values)]
            return _apply_trajectory_response(values, filtered, response_strength)
        except Exception:
            method = "gaussian"
    if method == "gaussian":
        axis = np.arange(window, dtype=np.float64) - pad
        sigma = max(window / 6.0, 0.5)
        kernel = np.exp(-(axis ** 2) / (2.0 * sigma ** 2))
        kernel /= kernel.sum()
    else:
        kernel = np.ones(window, dtype=np.float64) / float(window)
    filtered = np.convolve(padded, kernel, mode="valid")[:len(values)]
    return _apply_trajectory_response(values, filtered, response_strength)


def _bbox_size_value(width: float, height: float, metric: str, output_aspect: float) -> tuple[float, float]:
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    if metric == "height":
        source_h = height
        source_w = source_h * output_aspect
    elif metric == "width":
        source_w = width
        source_h = source_w / output_aspect
    else:
        if metric == "max_dimension":
            size = max(width, height)
            width = height = size
        elif metric == "area_sqrt":
            size = math.sqrt(width * height)
            width = height = size
        source_w = max(width, height * output_aspect)
        source_h = source_w / output_aspect
    return max(1.0, source_w), max(1.0, source_h)


def _ceil_divisible(value: float, step: int) -> int:
    step = max(1, int(step))
    return max(step, int(math.ceil(float(value) / step)) * step)


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


def _gaussian_blur_mask(mask_nchw: torch.Tensor, feather_px: int) -> torch.Tensor:
    """Memory-conscious separable Gaussian blur for one-channel masks."""
    radius = max(0, int(feather_px))
    if radius <= 0:
        return mask_nchw.clamp(0.0, 1.0)

    shortest = min(mask_nchw.shape[-2], mask_nchw.shape[-1])
    kernel_size = min(radius * 2 + 1, max(1, shortest if shortest % 2 == 1 else shortest - 1))
    if kernel_size < 3:
        return mask_nchw.clamp(0.0, 1.0)
    radius = kernel_size // 2
    axis = torch.arange(kernel_size, device=mask_nchw.device, dtype=torch.float32) - radius
    sigma = max(kernel_size / 6.0, 0.5)
    kernel = torch.exp(-(axis ** 2) / (2.0 * sigma ** 2))
    kernel = (kernel / kernel.sum()).to(mask_nchw.dtype)
    horizontal = kernel.view(1, 1, 1, kernel_size)
    vertical = kernel.view(1, 1, kernel_size, 1)
    result = F.conv2d(
        F.pad(mask_nchw, (radius, radius, 0, 0), mode="replicate"),
        horizontal,
    )
    result = F.conv2d(
        F.pad(result, (0, 0, radius, radius), mode="replicate"),
        vertical,
    )
    return result.clamp(0.0, 1.0)


def _feather_alpha(alpha_hw: torch.Tensor, feather_px: int) -> torch.Tensor:
    return _gaussian_blur_mask(alpha_hw.unsqueeze(0).unsqueeze(0), feather_px).squeeze(0).squeeze(0)


def _color_match_patch(patch_nhwc, base_nhwc, alpha_nhwc, mode: str, strength: float):
    strength = min(1.0, max(0.0, float(strength)))
    if mode == "off" or strength <= 0.0:
        return patch_nhwc

    weights = alpha_nhwc.float()
    patch = patch_nhwc.float()
    base = base_nhwc.float()
    weight_sum = weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    if mode == "luminance":
        coefficients = torch.tensor(
            [0.2126, 0.7152, 0.0722], device=patch.device, dtype=torch.float32
        ).view(1, 1, 1, 3)
        patch_luma = (patch[..., :3] * coefficients).sum(dim=-1, keepdim=True)
        base_luma = (base[..., :3] * coefficients).sum(dim=-1, keepdim=True)
        patch_mean = (patch_luma * weights).sum(dim=(1, 2), keepdim=True) / weight_sum
        base_mean = (base_luma * weights).sum(dim=(1, 2), keepdim=True) / weight_sum
        patch_std = (((patch_luma - patch_mean) ** 2 * weights).sum(dim=(1, 2), keepdim=True)
                     / weight_sum).sqrt().clamp_min(1e-6)
        base_std = (((base_luma - base_mean) ** 2 * weights).sum(dim=(1, 2), keepdim=True)
                    / weight_sum).sqrt().clamp_min(1e-6)
        target_luma = (patch_luma - patch_mean) * (base_std / patch_std) + base_mean
        adjusted = patch + (target_luma - patch_luma)
    else:
        patch_mean = (patch * weights).sum(dim=(1, 2), keepdim=True) / weight_sum
        base_mean = (base * weights).sum(dim=(1, 2), keepdim=True) / weight_sum
        if mode == "mean_std":
            patch_std = (((patch - patch_mean) ** 2 * weights).sum(dim=(1, 2), keepdim=True)
                         / weight_sum).sqrt().clamp_min(1e-6)
            base_std = (((base - base_mean) ** 2 * weights).sum(dim=(1, 2), keepdim=True)
                        / weight_sum).sqrt().clamp_min(1e-6)
            adjusted = (patch - patch_mean) * (base_std / patch_std) + base_mean
        else:
            adjusted = patch + (base_mean - patch_mean)

    return (patch + (adjusted - patch) * strength).clamp(0.0, 1.0).to(patch_nhwc.dtype)


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


CROP_PIPE_VERSION = "crop_by_mask_pipe_v1"

CROP_PIPE_OVERRIDABLE_KEYS = (
    "aspect_ratio",
    "output_long_side",
    "use_long_side",
    "use_custom_resolution",
    "width",
    "height",
    "margin_scale",
    "smooth_center",
    "center_smoothing_strength",
    "smooth_zoom",
    "zoom_smoothing_strength",
    "offset_x",
    "offset_y",
    "min_zoom",
    "max_zoom",
    "fit_frame_bounds",
    "divisible_by",
    "resolution_mode",
    "auto_resolution_cap",
    "smoothing_method",
    "center_smooth_window",
    "size_smooth_window",
    "size_metric",
)


def _broadcast_mask_to_batch(mask, B, name="mask"):
    """
    If mask has batch 1 and images have batch B, repeat it.
    Otherwise require exact batch match.
    """
    if mask is None:
        return None

    if mask.shape[0] == 1 and B > 1:
        return mask.repeat(B, 1, 1).contiguous()

    if mask.shape[0] != B:
        raise ValueError(
            f"Batch size mismatch: images={B}, {name}={mask.shape[0]}"
        )

    return mask


def _extract_crop_pipe_frames(pipe, B, W, H):
    """
    Extract per-frame crop transforms from pipe.

    Returns:
        pipe_frames: list[dict] or None
        pipe_params: dict
    """
    if pipe is None:
        return None, {}

    if not isinstance(pipe, dict):
        raise ValueError("pipe must be a dictionary produced by Batch Image Crop By Mask Advanced (StDismas).")

    pipe_params = dict(pipe.get("params", {}) or {})
    pipe_meta = pipe.get("crop_metadata", None)

    if not isinstance(pipe_meta, dict) or pipe_meta.get("version") != "crop_by_mask_v2":
        return None, pipe_params

    frames_in = pipe_meta.get("frames", [])
    if not isinstance(frames_in, (list, tuple)) or len(frames_in) == 0:
        return None, pipe_params

    if B <= 0:
        return [], pipe_params

    # Batch rules:
    # - if target batch is 1, take first pipe frame;
    # - if pipe has 1 frame, broadcast it to target batch;
    # - otherwise pipe frame count must match target batch.
    if B == 1:
        frames = [dict(frames_in[0])]
    elif len(frames_in) == 1:
        frames = [dict(frames_in[0]) for _ in range(B)]
    elif len(frames_in) == B:
        frames = [dict(f) for f in frames_in]
    else:
        raise ValueError(
            f"pipe crop_metadata has {len(frames_in)} frames, but current image batch has {B}. "
            f"Use the same batch size, a single-frame pipe, or a single image target."
        )

    first = frames[0]

    try:
        orig_w = int(first["orig_size"][0])
        orig_h = int(first["orig_size"][1])
        crop_w = int(first["crop_size"][0])
        crop_h = int(first["crop_size"][1])
    except Exception as exc:
        raise ValueError("pipe crop_metadata frames must contain orig_size and crop_size.") from exc

    # If target resolution differs but aspect ratio is the same, proportionally rescale crop.
    # If aspect ratio differs, exact identical crop is impossible.
    rescale = False
    scale_x = 1.0
    scale_y = 1.0
    scale_factor = 1.0

    if orig_w != W or orig_h != H:
        old_ar = float(orig_w) / max(1.0, float(orig_h))
        new_ar = float(W) / max(1.0, float(H))

        if abs(old_ar - new_ar) > 1e-4:
            raise ValueError(
                f"pipe crop data was created for {orig_w}x{orig_h}, "
                f"but current images are {W}x{H} and aspect ratio is different. "
                f"For identical crop use the same resolution/aspect ratio."
            )

        rescale = True
        scale_x = float(W) / float(orig_w)
        scale_y = float(H) / float(orig_h)
        scale_factor = (scale_x + scale_y) * 0.5

        # Keep pipe params consistent after rescale.
        if "offset_x" in pipe_params:
            try:
                pipe_params["offset_x"] = int(round(float(pipe_params["offset_x"]) * scale_x))
            except Exception:
                pass

        if "offset_y" in pipe_params:
            try:
                pipe_params["offset_y"] = int(round(float(pipe_params["offset_y"]) * scale_y))
            except Exception:
                pass

    for idx, frame in enumerate(frames):
        try:
            f_orig_w = int(frame["orig_size"][0])
            f_orig_h = int(frame["orig_size"][1])
            f_crop_w = int(frame["crop_size"][0])
            f_crop_h = int(frame["crop_size"][1])
            cx = float(frame["center"][0])
            cy = float(frame["center"][1])
            S = float(frame["S"])
        except Exception as exc:
            raise ValueError(f"pipe crop_metadata frame {idx} is missing required fields.") from exc

        if f_orig_w != orig_w or f_orig_h != orig_h:
            raise ValueError("All frames in pipe crop_metadata must have the same orig_size.")

        if f_crop_w != crop_w or f_crop_h != crop_h:
            raise ValueError("All frames in pipe crop_metadata must have the same crop_size.")

        if S <= 0:
            raise ValueError(f"pipe crop_metadata frame {idx} has invalid scale S={S}.")

        if rescale:
            cx = cx * scale_x
            cy = cy * scale_y
            S = S / scale_factor

            if "offset" in frame:
                try:
                    frame["offset"] = [
                        float(frame["offset"][0]) * scale_x,
                        float(frame["offset"][1]) * scale_y,
                    ]
                except Exception:
                    pass

            for key in ("mask_bbox", "mask_bbox_exp"):
                if key in frame and isinstance(frame[key], (list, tuple)) and len(frame[key]) == 4:
                    frame[key] = [
                        float(frame[key][0]) * scale_x,
                        float(frame[key][1]) * scale_y,
                        float(frame[key][2]) * scale_x,
                        float(frame[key][3]) * scale_y,
                    ]

            if bool(frame.get("fit_frame_bounds", False)):
                cx, cy, S = _fit_crop_to_frame_bounds(
                    cx=cx,
                    cy=cy,
                    scale=S,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    frame_w=W,
                    frame_h=H,
                )

            frame["inverse_affine_2x3"] = _affine_inverse_matrix(S, cx, cy, crop_w, crop_h)
            frame["forward_affine_2x3"] = _affine_forward_matrix(S, cx, cy, crop_w, crop_h)
        else:
            if frame.get("inverse_affine_2x3") is None:
                frame["inverse_affine_2x3"] = _affine_inverse_matrix(S, cx, cy, crop_w, crop_h)

            if frame.get("forward_affine_2x3") is None:
                frame["forward_affine_2x3"] = _affine_forward_matrix(S, cx, cy, crop_w, crop_h)

        frame["orig_size"] = [int(W), int(H)]
        frame["crop_size"] = [int(crop_w), int(crop_h)]
        frame["center"] = [float(cx), float(cy)]
        frame["S"] = float(S)

    return frames, pipe_params


def _track_faces(
    images,
    detector_name,
    confidence,
    select,
    identity_reference,
    identity_track,
    identity_threshold,
    identity_pack,
    face_device,
    keep_models_loaded,
    fallback_detector,
    fallback_head_frac,
):
    """Return face boxes plus validity/confidence without making face support mandatory."""
    B, H, W, _ = images.shape
    detector = _load_face_detector(detector_name, keep_models_loaded)
    recogniser = None
    body_detector = None
    detections = []
    warnings = []

    try:
        for i in range(B):
            _interrupt_if_requested()
            detections.append(_predict_boxes(detector, images[i], confidence, face_device))

        multiple_people = any(len(boxes) > 1 for boxes, _, _ in detections)
        reference_embedding = None
        anchor_source = "none"
        if identity_track and (identity_reference is not None or multiple_people):
            try:
                recogniser = _face_recogniser(identity_pack, face_device, keep_models_loaded)
                if identity_reference is not None:
                    candidates = _embed_faces(recogniser, _to_bgr_u8(identity_reference[0]))
                    if candidates:
                        idx = max(
                            range(len(candidates)),
                            key=lambda j: candidates[j][0][3] - candidates[j][0][1],
                        )
                        reference_embedding = candidates[idx][1]
                        anchor_source = "identity_reference"
                    else:
                        warnings.append("InsightFace found no usable face in identity_reference.")

                if reference_embedding is None:
                    embeddings = []
                    step = max(1, B // 24)
                    for i in range(0, B, step):
                        boxes = detections[i][0]
                        if not boxes:
                            continue
                        heights = sorted((box[3] - box[1] for box in boxes), reverse=True)
                        if len(heights) > 1 and heights[0] < heights[1] * 1.6:
                            continue
                        candidates = _embed_faces(recogniser, _to_bgr_u8(images[i]))
                        if not candidates:
                            continue
                        idx = max(
                            range(len(candidates)),
                            key=lambda j: candidates[j][0][3] - candidates[j][0][1],
                        )
                        embeddings.append(candidates[idx][1])
                    if embeddings:
                        anchor = np.mean(np.stack(embeddings), axis=0)
                        norm = np.linalg.norm(anchor)
                        reference_embedding = anchor / norm if norm > 0 else anchor
                        anchor_source = f"clip ({len(embeddings)} samples)"
            except Exception as exc:
                warnings.append(f"Identity matching unavailable; continuity was used ({exc}).")
                recogniser = None
                reference_embedding = None

        selected = [None] * B
        selected_confidence = np.zeros(B, dtype=np.float64)
        valid = np.zeros(B, dtype=bool)
        via_body = np.zeros(B, dtype=bool)
        last = None
        identity_resolved = 0
        continuity_resolved = 0
        ambiguous = 0

        for i, (boxes, confidences, _) in enumerate(detections):
            if not boxes:
                continue
            chosen = None
            chosen_confidence = 0.0
            if len(boxes) == 1:
                chosen = boxes[0]
                chosen_confidence = confidences[0]
                continuity_resolved += 1
            elif last is None:
                if reference_embedding is not None and recogniser is not None:
                    candidates = _embed_faces(recogniser, _to_bgr_u8(images[i]))
                    idx, score = _best_identity_match(candidates, reference_embedding)
                    if idx is not None and score >= float(identity_threshold):
                        chosen = candidates[idx][0]
                        chosen_confidence = max(0.0, score)
                        identity_resolved += 1
                if chosen is None:
                    if select == "most_central":
                        chosen = min(
                            boxes,
                            key=lambda box: ((box[0] + box[2]) * 0.5 - W * 0.5) ** 2
                            + ((box[1] + box[3]) * 0.5 - H * 0.5) ** 2,
                        )
                    else:
                        chosen = max(boxes, key=lambda box: box[3] - box[1])
                    chosen_confidence = confidences[boxes.index(chosen)]
            else:
                ranked = sorted(boxes, key=lambda box: _continuity_cost(box, last))
                best, second = ranked[0], ranked[1]
                best_cost = _continuity_cost(best, last)
                second_cost = _continuity_cost(second, last)
                conflict = second_cost < best_cost * 2.0 or _box_iou(best, second) > 0.2
                if conflict and reference_embedding is not None and recogniser is not None:
                    ambiguous += 1
                    nearby = [
                        box for box in boxes
                        if _continuity_cost(box, last) < max(best_cost * 3.0, 1.0)
                    ] or boxes
                    candidates = [
                        candidate for candidate in _embed_faces(recogniser, _to_bgr_u8(images[i]))
                        if any(_box_iou(candidate[0], box) > 0.3 for box in nearby)
                    ]
                    idx, score = _best_identity_match(candidates, reference_embedding)
                    if idx is not None and score >= float(identity_threshold):
                        chosen = candidates[idx][0]
                        chosen_confidence = max(0.0, score)
                        identity_resolved += 1
                if chosen is None:
                    chosen = best
                    chosen_confidence = confidences[boxes.index(best)]
                    continuity_resolved += 1

            selected[i] = [float(value) for value in chosen]
            selected_confidence[i] = float(chosen_confidence)
            valid[i] = True
            last = (
                (chosen[0] + chosen[2]) * 0.5,
                (chosen[1] + chosen[3]) * 0.5,
                chosen[3] - chosen[1],
            )

        if not valid.any():
            raise ValueError(
                "No face was detected in any frame. Lower face_confidence or use tracking_mode='mask'."
            )

        widths = np.zeros(B, dtype=np.float64)
        heights = np.zeros(B, dtype=np.float64)
        centers_x = np.zeros(B, dtype=np.float64)
        centers_y = np.zeros(B, dtype=np.float64)
        for i, box in enumerate(selected):
            if box is None:
                continue
            widths[i] = max(1.0, box[2] - box[0])
            heights[i] = max(1.0, box[3] - box[1])
            centers_x[i] = (box[0] + box[2]) * 0.5
            centers_y[i] = (box[1] + box[3]) * 0.5

        height_seed = _interp_gaps(heights, valid, max(8.0, min(W, H) * 0.1))
        width_seed = _interp_gaps(widths, valid, height_seed.mean() * 0.8)
        if fallback_detector != "none" and (~valid).any():
            try:
                body_detector = _load_face_detector(fallback_detector, keep_models_loaded)
                for i in np.nonzero(~valid)[0]:
                    _interrupt_if_requested()
                    boxes, confidences, classes = _predict_boxes(
                        body_detector, images[i], confidence, face_device
                    )
                    people = [
                        (box, score) for box, score, cls in zip(boxes, confidences, classes)
                        if int(cls) == 0
                    ] or list(zip(boxes, confidences))
                    if not people:
                        continue
                    person, score = max(
                        people,
                        key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]),
                    )
                    centers_x[i] = (person[0] + person[2]) * 0.5
                    centers_y[i] = person[1] + float(fallback_head_frac) * max(height_seed[i], 8.0)
                    widths[i] = width_seed[i]
                    heights[i] = height_seed[i]
                    selected_confidence[i] = float(score) * 0.75
                    via_body[i] = True
            except Exception as exc:
                warnings.append(f"Body fallback '{fallback_detector}' failed ({exc}).")

        known_center = valid | via_body
        centers_x = _interp_gaps(centers_x, known_center, W * 0.5)
        centers_y = _interp_gaps(centers_y, known_center, H * 0.5)
        widths = _interp_gaps(widths, valid, max(8.0, min(W, H) * 0.08))
        heights = _interp_gaps(heights, valid, max(8.0, min(W, H) * 0.1))

        report = {
            "anchor_source": anchor_source,
            "identity_resolved": identity_resolved,
            "continuity_resolved": continuity_resolved,
            "ambiguous": ambiguous,
            "face_detected": int(valid.sum()),
            "body_fallback": int(via_body.sum()),
            "interpolated": int(B - known_center.sum()),
            "warnings": warnings,
        }
        return centers_x, centers_y, widths, heights, valid, via_body, selected_confidence, report
    finally:
        if not keep_models_loaded:
            _release_optional_face_models(detector, recogniser, body_detector)
            detector = None
            recogniser = None
            body_detector = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class BatchImageCropByMaskAdvanced_StDismas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input image batch (B,H,W,C)"}),
                "crop_mask": ("MASK", {"tooltip": "Main mask used to compute crop region"}),
                "tracking_mode": (
                    ["mask", "face_detection"],
                    {
                        "default": "mask",
                        "tooltip": "mask keeps the universal mask workflow. face_detection ignores crop_mask geometry and tracks a detected face.",
                    },
                ),
                "aspect_ratio": (
                    ASPECT_RATIO_CHOICES,
                    {"default": "16:9", "tooltip": "Output aspect ratio"},
                ),
                "output_long_side": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Target size of the selected output side in pixels",
                    },
                ),
                "use_long_side": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If enabled, output_long_side controls the long side; if disabled, it controls the short side",
                    },
                ),
                "use_custom_resolution": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, use custom width and height for the crop output instead of aspect_ratio/output_long_side",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Custom crop output width when use_custom_resolution is enabled",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 576,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Custom crop output height when use_custom_resolution is enabled",
                    },
                ),
                "margin_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Expands mask bbox before cropping",
                    },
                ),
                "smooth_center": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable temporal smoothing for crop center movement"},
                ),
                "smoothing_method": (
                    SMOOTHING_METHODS,
                    {
                        "default": "gaussian",
                        "tooltip": "Filter used for center and zoom. The legacy strength controls set how quickly the crop follows the filtered trajectory.",
                    },
                ),
                "center_smoothing_strength": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Center smoothing strength (0 = locked to previous, 1 = follow current center)",
                    },
                ),
                "center_smooth_window": (
                    "INT",
                    {"default": 21, "min": 1, "max": 401, "step": 2, "tooltip": "Temporal filter window for crop center."},
                ),
                "smooth_zoom": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable temporal smoothing for zoom changes"},
                ),
                "zoom_smoothing_strength": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Zoom smoothing strength (0 = locked to previous, 1 = follow current zoom)",
                    },
                ),
                "size_smooth_window": (
                    "INT",
                    {"default": 51, "min": 1, "max": 401, "step": 2, "tooltip": "Independent temporal filter window for crop size/zoom."},
                ),
                "offset_x": (
                    "INT",
                    {
                        "default": 0,
                        "min": -8192,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Horizontal offset from mask center in source pixels",
                    },
                ),
                "offset_y": (
                    "INT",
                    {
                        "default": 0,
                        "min": -8192,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Vertical offset from mask center in source pixels",
                    },
                ),
                "min_zoom": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Minimum zoom limit"},
                ),
                "max_zoom": (
                    "FLOAT",
                    {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.01, "tooltip": "Maximum zoom limit"},
                ),
                "interpolation": (
                    ["bilinear", "bicubic"],
                    {"default": "bilinear", "tooltip": "Sampling method for image crop"},
                ),
                "fit_frame_bounds": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Keep the crop window fully inside the source frame while preserving aspect ratio",
                    },
                ),
                "divisible_by": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "Make both output crop dimensions divisible by this value",
                    },
                ),
                "enable_visualize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Draw full-frame crop preview. Disable for better speed and much lower memory use on video batches.",
                    },
                ),
                "crop_chunk_size": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "How many frames are sampled per grid_sample batch. Lower uses less memory; higher can be faster.",
                    },
                ),
                "size_metric": (
                    SIZE_METRICS,
                    {
                        "default": "bbox_fit",
                        "tooltip": "bbox_fit is generic. height is more stable for faces turning in profile.",
                    },
                ),
                "resolution_mode": (
                    RESOLUTION_MODES,
                    {
                        "default": "manual",
                        "tooltip": "manual keeps the selected canvas. auto_no_downscale sizes to the largest source crop. auto_capped does the same up to auto_resolution_cap.",
                    },
                ),
                "auto_resolution_cap": (
                    "INT",
                    {"default": 768, "min": 128, "max": 8192, "step": 32, "tooltip": "Maximum long side for auto_capped."},
                ),
            },
            "optional": {
                "masks": (
                    "MASK",
                    {
                        "tooltip": "Optional extra mask that is cropped with the same transform but does not affect crop computation"
                    },
                ),
                "pipe": (
                    "CROP_PIPE",
                    {
                        "tooltip": (
                            "Pipe from another Batch Image Crop By Mask Advanced node. "
                            "Overrides all crop settings except interpolation, enable_visualize and crop_chunk_size. "
                            "Also reuses computed crop transforms for identical cropping."
                        )
                    },
                ),
                "identity_reference": (
                    "IMAGE",
                    {"tooltip": "Optional face identity reference used only in face_detection mode."},
                ),
                "face_detector": (
                    _face_detector_list(),
                    {"tooltip": "Ultralytics face detector model. Loaded only in face_detection mode."},
                ),
                "face_confidence": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05},
                ),
                "face_select": (
                    ["largest", "most_central"],
                    {"default": "largest", "tooltip": "Tie-break before identity/continuity has a track."},
                ),
                "identity_track": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Use continuity first and InsightFace only on ambiguous multi-face frames."},
                ),
                "identity_threshold": (
                    "FLOAT",
                    {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "identity_pack": (
                    ["buffalo_l", "buffalo_s"],
                    {"default": "buffalo_l", "tooltip": "InsightFace model pack. buffalo_s uses less memory."},
                ),
                "face_model_device": (
                    ["cpu", "auto", "cuda"],
                    {"default": "cpu", "tooltip": "CPU is safest for low-VRAM/Ranpod runs; CUDA is faster."},
                ),
                "keep_face_models_loaded": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "False releases optional face models before downstream generation to protect VRAM/RAM."},
                ),
                "fallback_detector": (
                    ["none"] + _face_detector_list(),
                    {"default": "none", "tooltip": "Optional person/body detector for frames where the face is missing."},
                ),
                "fallback_head_frac": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.5, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "IMAGE", "BBOXES", "CROP_PIPE", "STRING", "INT", "INT")
    RETURN_NAMES = ("cropped_images", "cropped_masks", "masks", "visualize", "crop_metadata", "pipe", "report", "canvas_width", "canvas_height")
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
        tracking_mode="mask",
        smoothing_method="gaussian",
        center_smooth_window=21,
        size_smooth_window=51,
        size_metric="bbox_fit",
        resolution_mode="manual",
        auto_resolution_cap=768,
        masks=None,
        pipe=None,
        identity_reference=None,
        face_detector="face_yolov8m.pt",
        face_confidence=0.35,
        face_select="largest",
        identity_track=True,
        identity_threshold=0.28,
        identity_pack="buffalo_l",
        face_model_device="cpu",
        keep_face_models_loaded=False,
        fallback_detector="none",
        fallback_head_frac=0.5,
    ):
        B, H, W, C = images.shape

        if tracking_mode == "mask":
            if crop_mask is None:
                raise ValueError("crop_mask is required when tracking_mode='mask'.")
            crop_mask = _ensure_mask_hw(crop_mask, H, W)
            crop_mask = _broadcast_mask_to_batch(crop_mask, B, name="crop_mask")

        has_extra_masks = masks is not None
        if has_extra_masks:
            masks = _ensure_mask_hw(masks, H, W)
            masks = _broadcast_mask_to_batch(masks, B, name="masks")

        device = images.device
        dtype = images.dtype
        chunk_size = max(1, int(crop_chunk_size))

        manual_params = {
            "aspect_ratio": aspect_ratio,
            "output_long_side": output_long_side,
            "use_long_side": use_long_side,
            "use_custom_resolution": use_custom_resolution,
            "width": width,
            "height": height,
            "margin_scale": margin_scale,
            "smooth_center": smooth_center,
            "center_smoothing_strength": center_smoothing_strength,
            "smooth_zoom": smooth_zoom,
            "zoom_smoothing_strength": zoom_smoothing_strength,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "min_zoom": min_zoom,
            "max_zoom": max_zoom,
            "fit_frame_bounds": fit_frame_bounds,
            "divisible_by": divisible_by,
            "resolution_mode": resolution_mode,
            "auto_resolution_cap": auto_resolution_cap,
            "smoothing_method": smoothing_method,
            "center_smooth_window": center_smooth_window,
            "size_smooth_window": size_smooth_window,
            "size_metric": size_metric,
        }

        pipe_frames, pipe_params = _extract_crop_pipe_frames(pipe, B, W, H)

        if pipe is not None:
            for key in CROP_PIPE_OVERRIDABLE_KEYS:
                if key in pipe_params:
                    manual_params[key] = pipe_params[key]

        aspect_ratio = manual_params["aspect_ratio"]
        output_long_side = manual_params["output_long_side"]
        use_long_side = manual_params["use_long_side"]
        use_custom_resolution = manual_params["use_custom_resolution"]
        width = manual_params["width"]
        height = manual_params["height"]
        margin_scale = manual_params["margin_scale"]
        smooth_center = manual_params["smooth_center"]
        center_smoothing_strength = manual_params["center_smoothing_strength"]
        smooth_zoom = manual_params["smooth_zoom"]
        zoom_smoothing_strength = manual_params["zoom_smoothing_strength"]
        offset_x = manual_params["offset_x"]
        offset_y = manual_params["offset_y"]
        min_zoom = manual_params["min_zoom"]
        max_zoom = manual_params["max_zoom"]
        fit_frame_bounds = manual_params["fit_frame_bounds"]
        divisible_by = manual_params["divisible_by"]
        resolution_mode = manual_params.get("resolution_mode", resolution_mode)
        auto_resolution_cap = manual_params.get("auto_resolution_cap", auto_resolution_cap)
        smoothing_method = manual_params.get("smoothing_method", smoothing_method)
        center_smooth_window = manual_params.get("center_smooth_window", center_smooth_window)
        size_smooth_window = manual_params.get("size_smooth_window", size_smooth_window)
        size_metric = manual_params.get("size_metric", size_metric)

        ratio = (
            float(width) / max(1.0, float(height))
            if use_custom_resolution
            else _parse_aspect_ratio(aspect_ratio)
        )
        face_report = {}
        inverse_affines = []
        centers_scales = []
        out_frames = []

        if pipe_frames is not None and len(pipe_frames) > 0:
            crop_w = int(pipe_frames[0]["crop_size"][0])
            crop_h = int(pipe_frames[0]["crop_size"][1])
            for frame in pipe_frames:
                cx = float(frame["center"][0])
                cy = float(frame["center"][1])
                scale = float(frame["S"])
                inverse_affines.append(frame["inverse_affine_2x3"])
                centers_scales.append((cx, cy, scale))
                frame_out = dict(frame)
                frame_out["orig_size"] = [int(W), int(H)]
                frame_out["crop_size"] = [int(crop_w), int(crop_h)]
                frame_out.setdefault("offset", [float(offset_x), float(offset_y)])
                frame_out.setdefault("fit_frame_bounds", bool(fit_frame_bounds))
                frame_out.setdefault("divisible_by", int(divisible_by))
                frame_out.setdefault("use_long_side", bool(use_long_side))
                frame_out.setdefault("use_custom_resolution", bool(use_custom_resolution))
                frame_out.setdefault("valid", True)
                frame_out.setdefault("tracking_confidence", 1.0)
                frame_out.setdefault("via_body", False)
                out_frames.append(frame_out)
        else:
            pipe_frames = None
            raw_cx = np.zeros(B, dtype=np.float64)
            raw_cy = np.zeros(B, dtype=np.float64)
            raw_bw = np.zeros(B, dtype=np.float64)
            raw_bh = np.zeros(B, dtype=np.float64)
            valid = np.zeros(B, dtype=bool)
            via_body = np.zeros(B, dtype=bool)
            tracking_confidence = np.zeros(B, dtype=np.float64)

            if tracking_mode == "face_detection":
                (raw_cx, raw_cy, raw_bw, raw_bh, valid, via_body,
                 tracking_confidence, face_report) = _track_faces(
                    images=images,
                    detector_name=face_detector,
                    confidence=face_confidence,
                    select=face_select,
                    identity_reference=identity_reference,
                    identity_track=identity_track,
                    identity_threshold=identity_threshold,
                    identity_pack=identity_pack,
                    face_device=face_model_device,
                    keep_models_loaded=keep_face_models_loaded,
                    fallback_detector=fallback_detector,
                    fallback_head_frac=fallback_head_frac,
                )
            else:
                for i in range(B):
                    _interrupt_if_requested()
                    bbox = _mask_bbox(crop_mask[i])
                    if bbox is None:
                        continue
                    min_x, min_y, max_x, max_y = (float(value) for value in bbox)
                    raw_cx[i] = (min_x + max_x) * 0.5
                    raw_cy[i] = (min_y + max_y) * 0.5
                    raw_bw[i] = max(1.0, max_x - min_x)
                    raw_bh[i] = max(1.0, max_y - min_y)
                    valid[i] = True
                    tracking_confidence[i] = 1.0

                default_size = max(1.0, min(W, H) * 0.25)
                raw_cx = _interp_gaps(raw_cx, valid, W * 0.5)
                raw_cy = _interp_gaps(raw_cy, valid, H * 0.5)
                raw_bw = _interp_gaps(raw_bw, valid, default_size)
                raw_bh = _interp_gaps(raw_bh, valid, default_size)

            raw_cx = raw_cx + float(offset_x)
            raw_cy = raw_cy + float(offset_y)
            margin_eff = max(float(margin_scale), 1.0)
            source_w = np.zeros(B, dtype=np.float64)
            source_h = np.zeros(B, dtype=np.float64)
            for i in range(B):
                source_w[i], source_h[i] = _bbox_size_value(
                    raw_bw[i] * margin_eff,
                    raw_bh[i] * margin_eff,
                    size_metric,
                    ratio,
                )
                if fit_frame_bounds:
                    fit = min(1.0, W / source_w[i], H / source_h[i])
                    source_w[i] *= fit
                    source_h[i] *= fit

            centers_x = (
                _smooth_trajectory(raw_cx, center_smooth_window, smoothing_method, center_smoothing_strength)
                if smooth_center else raw_cx.copy()
            )
            centers_y = (
                _smooth_trajectory(raw_cy, center_smooth_window, smoothing_method, center_smoothing_strength)
                if smooth_center else raw_cy.copy()
            )
            smoothed_source_w = (
                _smooth_trajectory(source_w, size_smooth_window, smoothing_method, zoom_smoothing_strength)
                if smooth_zoom else source_w.copy()
            )
            smoothed_source_h = (
                _smooth_trajectory(source_h, size_smooth_window, smoothing_method, zoom_smoothing_strength)
                if smooth_zoom else source_h.copy()
            )

            if resolution_mode == "manual":
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
            else:
                auto_step = math.lcm(32, max(1, int(divisible_by)))
                need_w = float(smoothed_source_w.max())
                need_h = float(smoothed_source_h.max())
                target_side = need_w if ratio >= 1.0 else need_h
                target_side = _ceil_divisible(target_side, auto_step)
                auto_use_long_side = True
                crop_w, crop_h = _compute_crop_size(
                    target_side,
                    ratio,
                    use_long_side=auto_use_long_side,
                    divisible_by=auto_step,
                )
                while (crop_w + 1e-6 < need_w or crop_h + 1e-6 < need_h) and max(crop_w, crop_h) < MAX_RESOLUTION:
                    target_side += auto_step
                    crop_w, crop_h = _compute_crop_size(
                        target_side,
                        ratio,
                        use_long_side=auto_use_long_side,
                        divisible_by=auto_step,
                    )
                if resolution_mode == "auto_capped":
                    cap = max(auto_step, (int(auto_resolution_cap) // auto_step) * auto_step)
                    if max(crop_w, crop_h) > cap:
                        crop_w, crop_h = _compute_crop_size(
                            cap,
                            ratio,
                            use_long_side=True,
                            divisible_by=auto_step,
                        )
                crop_w = min(MAX_RESOLUTION, int(crop_w))
                crop_h = min(MAX_RESOLUTION, int(crop_h))

            for i in range(B):
                source_width = max(1.0, float(smoothed_source_w[i]))
                source_height = max(1.0, float(smoothed_source_h[i]))
                scale = min(crop_w / source_width, crop_h / source_height)
                scale = max(float(min_zoom), min(float(max_zoom), scale))
                cx = float(centers_x[i])
                cy = float(centers_y[i])
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

                min_x = float(raw_cx[i] - float(offset_x) - raw_bw[i] * 0.5)
                max_x = float(raw_cx[i] - float(offset_x) + raw_bw[i] * 0.5)
                min_y = float(raw_cy[i] - float(offset_y) - raw_bh[i] * 0.5)
                max_y = float(raw_cy[i] - float(offset_y) + raw_bh[i] * 0.5)
                forward_affine = _affine_forward_matrix(scale, cx, cy, crop_w, crop_h)
                inverse_affine = _affine_inverse_matrix(scale, cx, cy, crop_w, crop_h)
                inverse_affines.append(inverse_affine)
                centers_scales.append((cx, cy, scale))
                out_frames.append({
                    "orig_size": [int(W), int(H)],
                    "crop_size": [int(crop_w), int(crop_h)],
                    "S": float(scale),
                    "center": [cx, cy],
                    "offset": [float(offset_x), float(offset_y)],
                    "fit_frame_bounds": bool(fit_frame_bounds),
                    "divisible_by": int(divisible_by),
                    "use_long_side": bool(use_long_side),
                    "use_custom_resolution": bool(use_custom_resolution),
                    "tracking_mode": tracking_mode,
                    "size_metric": size_metric,
                    "valid": bool(valid[i]),
                    "via_body": bool(via_body[i]),
                    "tracking_confidence": float(tracking_confidence[i]),
                    "forward_affine_2x3": forward_affine,
                    "inverse_affine_2x3": inverse_affine,
                    "mask_bbox": [min_x, min_y, max_x, max_y],
                    "mask_bbox_exp": [
                        cx - source_width * 0.5,
                        cy - source_height * 0.5,
                        cx + source_width * 0.5,
                        cy + source_height * 0.5,
                    ],
                })

        out_imgs = torch.empty((B, crop_h, crop_w, C), device=device, dtype=dtype)
        out_crop_masks = torch.zeros((B, crop_h, crop_w), device=device, dtype=dtype)
        out_masks = (
            torch.empty((B, crop_h, crop_w), device=device, dtype=dtype)
            if has_extra_masks else None
        )

        # Coordinate calculations and resampling stay in FP32 even for FP16/BF16 images.
        base_grid_x, base_grid_y = _make_pixel_grid(
            crop_w, crop_h, device=device, dtype=torch.float32
        )
        for start in range(0, B, chunk_size):
            _interrupt_if_requested()
            end = min(B, start + chunk_size)
            affines = torch.tensor(
                inverse_affines[start:end], device=device, dtype=torch.float32
            )
            grid = _build_affine_grid_batch(affines, base_grid_x, base_grid_y, W, H)
            sampled_imgs = F.grid_sample(
                images[start:end].permute(0, 3, 1, 2).float(),
                grid,
                mode=interpolation,
                padding_mode="zeros",
                align_corners=False,
            )
            out_imgs[start:end] = sampled_imgs.permute(0, 2, 3, 1).to(dtype)

            if tracking_mode == "mask" or pipe_frames is not None:
                sampled_crop_masks = F.grid_sample(
                    crop_mask[start:end].unsqueeze(1).float(),
                    grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )
                out_crop_masks[start:end] = sampled_crop_masks.squeeze(1).to(dtype)
            else:
                for local_index, frame_index in enumerate(range(start, end)):
                    x0, y0, x1, y1 = out_frames[frame_index]["mask_bbox"]
                    scale = float(out_frames[frame_index]["S"])
                    tx = float(out_frames[frame_index]["forward_affine_2x3"][0][2])
                    ty = float(out_frames[frame_index]["forward_affine_2x3"][1][2])
                    cx0 = max(0, min(crop_w, int(math.floor(scale * x0 + tx))))
                    cx1 = max(0, min(crop_w, int(math.ceil(scale * x1 + tx))))
                    cy0 = max(0, min(crop_h, int(math.floor(scale * y0 + ty))))
                    cy1 = max(0, min(crop_h, int(math.ceil(scale * y1 + ty))))
                    if cx1 > cx0 and cy1 > cy0:
                        out_crop_masks[frame_index, cy0:cy1, cx0:cx1] = 1.0

            if has_extra_masks:
                sampled_extra_masks = F.grid_sample(
                    masks[start:end].unsqueeze(1).float(),
                    grid,
                    mode="nearest",
                    padding_mode="zeros",
                    align_corners=False,
                )
                out_masks[start:end] = sampled_extra_masks.squeeze(1).to(dtype)

        if not has_extra_masks:
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
            out_visualize = images

        metadata = {
            "version": "crop_by_mask_v2",
            "frames": out_frames,
            "tracking_mode": tracking_mode,
        }

        magnifications = [float(frame["S"]) for frame in out_frames]
        valid_count = sum(bool(frame.get("valid", True)) for frame in out_frames)
        center_before = (
            float(np.mean(np.abs(np.diff(raw_cx))) + np.mean(np.abs(np.diff(raw_cy)))) * 0.5
            if pipe_frames is None and B > 1 else 0.0
        )
        center_after = (
            float(np.mean(np.abs(np.diff([item[0] for item in centers_scales])))
                  + np.mean(np.abs(np.diff([item[1] for item in centers_scales])))) * 0.5
            if B > 1 else 0.0
        )
        report_lines = [
            f"mode={tracking_mode} frames={B} valid={valid_count} interpolated={B - valid_count}",
            f"canvas={crop_w}x{crop_h} resolution_mode={resolution_mode}",
            f"magnification min={min(magnifications):.3f}x mean={np.mean(magnifications):.3f}x max={max(magnifications):.3f}x",
            f"center jitter {center_before:.3f} -> {center_after:.3f} px/frame ({smoothing_method})",
        ]
        downscaled = sum(value < 1.0 for value in magnifications)
        if downscaled:
            report_lines.append(
                f"WARNING: {downscaled}/{B} frames are downscaled before processing (magnification < 1.0x)."
            )
        if face_report:
            report_lines.append(
                "face tracking: "
                f"continuity={face_report.get('continuity_resolved', 0)} "
                f"identity={face_report.get('identity_resolved', 0)} "
                f"ambiguous={face_report.get('ambiguous', 0)} "
                f"body={face_report.get('body_fallback', 0)} "
                f"anchor={face_report.get('anchor_source', 'none')}"
            )
            report_lines.extend(f"WARNING: {warning}" for warning in face_report.get("warnings", []))
        report = "\n".join(report_lines)
        metadata["report"] = report
        metadata["magnification"] = {
            "min": float(min(magnifications)),
            "mean": float(np.mean(magnifications)),
            "max": float(max(magnifications)),
        }

        out_params = dict(manual_params)
        out_params["width"] = int(crop_w)
        out_params["height"] = int(crop_h)

        # These are included for completeness, but pipe consumer intentionally
        # does NOT override them from pipe.
        out_params["interpolation"] = interpolation
        out_params["enable_visualize"] = enable_visualize
        out_params["crop_chunk_size"] = crop_chunk_size

        out_pipe = {
            "version": CROP_PIPE_VERSION,
            "params": out_params,
            "crop_metadata": metadata,
        }

        return (
            out_imgs,
            out_crop_masks,
            out_masks,
            out_visualize,
            metadata,
            out_pipe,
            report,
            int(crop_w),
            int(crop_h),
        )


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
                "square_mask_units": (["crop_pixels", "source_pixels"], {"default": "crop_pixels", "tooltip": "source_pixels keeps inset/fade physically constant while zoom changes; crop_pixels preserves legacy behavior."}),
                "color_match_mode": (COLOR_MATCH_MODES, {"default": "off", "tooltip": "off disables matching; mean, mean_std, and luminance provide increasingly strong matching options."}),
                "color_match_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "undetected_frames": (["fade_out", "skip", "composite_anyway"], {"default": "fade_out", "tooltip": "Controls only compositing; all crop frames remain available to the generator."}),
                "dropout_fade_window": ("INT", {"default": 9, "min": 1, "max": 101, "step": 2}),
                "uncrop_chunk_size": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1, "tooltip": "Maximum frames warped together."}),
                "uncrop_memory_limit_mb": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64, "tooltip": "Safety budget for temporary full-frame FP32 warp tensors; lowers the effective chunk automatically."}),
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
        square_mask_units="crop_pixels",
        color_match_mode="off",
        color_match_strength=1.0,
        undetected_frames="fade_out",
        dropout_fade_window=9,
        uncrop_chunk_size=4,
        uncrop_memory_limit_mb=512,
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
                if C != Cc:
                    raise ValueError(f"Channel mismatch: base_images={C}, cropped_images={Cc}")
                out = base_images.clone()

            square_alpha_cache = {}
            ones_alpha_cache = {}

            detected = np.asarray(
                [bool(frame.get("valid", True)) for frame in frames], dtype=np.float64
            )
            if undetected_frames == "composite_anyway":
                detection_weights = np.ones(B, dtype=np.float64)
            elif undetected_frames == "skip":
                detection_weights = detected
            else:
                detection_weights = np.clip(
                    _smooth_trajectory(detected, dropout_fade_window, "gaussian", 1.0),
                    0.0,
                    1.0,
                )

            # Estimate every large temporary tensor, then cap the user-selected chunk.
            bytes_per_pixel = 4 * (2 + Cc + 1 + Cc + Cc)
            per_frame_mb = max(1e-6, H * W * bytes_per_pixel / (1024.0 ** 2))
            memory_chunk = max(1, int(float(uncrop_memory_limit_mb) / per_frame_mb))
            chunk_size = max(1, min(int(uncrop_chunk_size), memory_chunk, B))
            full_grid_x, full_grid_y = _make_pixel_grid(
                W, H, device=device, dtype=torch.float32
            )

            for start in range(0, B, chunk_size):
                _interrupt_if_requested()
                end = min(B, start + chunk_size)
                chunk_frames = frames[start:end]
                affines = torch.tensor(
                    [frame["forward_affine_2x3"] for frame in chunk_frames],
                    device=device,
                    dtype=torch.float32,
                )
                crop_w, crop_h = (int(value) for value in chunk_frames[0]["crop_size"])
                if any(tuple(frame["crop_size"]) != (crop_w, crop_h) for frame in chunk_frames):
                    raise ValueError("All crop_metadata frames in a batch must use the same crop_size.")
                grid = _build_affine_grid_batch(
                    affines, full_grid_x, full_grid_y, crop_w, crop_h
                )
                warped = F.grid_sample(
                    cropped_images[start:end].permute(0, 3, 1, 2).float(),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

                if mode == "overlay_by_mask" and not use_square_mask:
                    if crop_masks is None:
                        raise ValueError("mode='overlay_by_mask' with use_square_mask=False requires crop_masks.")
                    alpha_patch = crop_masks[start:end].unsqueeze(1).float().to(device)
                else:
                    alpha_items = []
                    for frame in chunk_frames:
                        if mode == "overlay_by_mask":
                            factor = float(frame.get("S", 1.0)) if square_mask_units == "source_pixels" else 1.0
                            values = tuple(
                                int(round(float(value) * factor))
                                for value in (
                                    square_mask_inset_left_px,
                                    square_mask_inset_right_px,
                                    square_mask_inset_top_px,
                                    square_mask_inset_bottom_px,
                                    square_mask_fade_left_px,
                                    square_mask_fade_right_px,
                                    square_mask_fade_top_px,
                                    square_mask_fade_bottom_px,
                                )
                            )
                            key = (crop_h, crop_w, *values)
                            item = square_alpha_cache.get(key)
                            if item is None:
                                item = _make_square_alpha(
                                    crop_h,
                                    crop_w,
                                    inset_left_px=values[0],
                                    inset_right_px=values[1],
                                    inset_top_px=values[2],
                                    inset_bottom_px=values[3],
                                    fade_left_px=values[4],
                                    fade_right_px=values[5],
                                    fade_top_px=values[6],
                                    fade_bottom_px=values[7],
                                    device=device,
                                    dtype=torch.float32,
                                )
                                square_alpha_cache[key] = item
                        else:
                            key = (crop_h, crop_w)
                            item = ones_alpha_cache.get(key)
                            if item is None:
                                item = torch.ones((crop_h, crop_w), device=device, dtype=torch.float32)
                                ones_alpha_cache[key] = item
                        alpha_items.append(item)
                    alpha_patch = torch.stack(alpha_items, dim=0).unsqueeze(1)

                alpha = F.grid_sample(
                    alpha_patch,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).clamp(0.0, 1.0)
                if mode == "overlay_by_mask" and not use_square_mask:
                    alpha = _gaussian_blur_mask(alpha, feather_px)
                alpha_nhwc = alpha.permute(0, 2, 3, 1)

                base = out[start:end].to(device=device, dtype=torch.float32)
                if color_match_mode != "off" and color_match_strength > 0.0 and Cc >= 3:
                    matched_rgb = _color_match_patch(
                        warped[..., :3],
                        base[..., :3],
                        alpha_nhwc,
                        color_match_mode,
                        color_match_strength,
                    )
                    warped = (
                        torch.cat((matched_rgb, warped[..., 3:]), dim=-1)
                        if Cc > 3 else matched_rgb
                    )

                weights = torch.tensor(
                    detection_weights[start:end], device=device, dtype=torch.float32
                ).view(-1, 1, 1, 1)
                effective_alpha = alpha_nhwc * weights * float(blend)
                composited = base * (1.0 - effective_alpha) + warped * effective_alpha
                out[start:end] = composited.to(device=out.device, dtype=out.dtype)

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


class BatchImageCropByMaskOrFaceAdvanced_StDismas(BatchImageCropByMaskAdvanced_StDismas):
    """Optional-mask entry point for standalone face detection without breaking old graphs."""

    @classmethod
    def INPUT_TYPES(cls):
        input_types = super().INPUT_TYPES()
        required = dict(input_types["required"])
        optional = dict(input_types["optional"])
        crop_mask_type = required.pop("crop_mask")
        optional = {"crop_mask": crop_mask_type, **optional}
        return {"required": required, "optional": optional}

    def crop(self, *args, **kwargs):
        kwargs.setdefault("crop_mask", None)
        return super().crop(*args, **kwargs)


NODE_CLASS_MAPPINGS = {
    "BatchImageCropByMaskAdvanced_StDismas": BatchImageCropByMaskAdvanced_StDismas,
    "BatchImageCropByMaskOrFaceAdvanced_StDismas": BatchImageCropByMaskOrFaceAdvanced_StDismas,
    "BatchImageUncropByMaskAdvanced_StDismas": BatchImageUncropByMaskAdvanced_StDismas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageCropByMaskAdvanced_StDismas": "Batch Image Crop By Mask Advanced (StDismas)",
    "BatchImageCropByMaskOrFaceAdvanced_StDismas": "Batch Image Crop By Mask or Face Advanced (StDismas)",
    "BatchImageUncropByMaskAdvanced_StDismas": "Batch Image Uncrop By Mask Advanced (StDismas)",
}
