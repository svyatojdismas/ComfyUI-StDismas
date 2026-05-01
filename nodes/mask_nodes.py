import torch
import torch.nn.functional as F

import numpy as np
import scipy.ndimage


def _ensure_mask_4d(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        return mask.unsqueeze(0).unsqueeze(0).float()
    if mask.ndim == 3:
        return mask.unsqueeze(1).float()
    raise ValueError(f"Expected MASK with 2 or 3 dims, got shape {tuple(mask.shape)}")


def _restore_mask_dims(mask_4d: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return mask_4d.squeeze(1).to(original.dtype)


def _dilate(mask_4d: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask_4d
    kernel = 2 * pixels + 1
    padded = F.pad(mask_4d, (pixels, pixels, pixels, pixels), mode="constant", value=0.0)
    return F.max_pool2d(padded, kernel_size=kernel, stride=1)


def _erode(mask_4d: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask_4d
    kernel = 2 * pixels + 1
    padded = F.pad(mask_4d, (pixels, pixels, pixels, pixels), mode="constant", value=1.0)
    return -F.max_pool2d(-padded, kernel_size=kernel, stride=1)


def _offset_mask(mask_4d: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels > 0:
        return _dilate(mask_4d, pixels)
    if pixels < 0:
        return _erode(mask_4d, -pixels)
    return mask_4d


def _find_nonzero_bbox(mask_2d: torch.Tensor):
    ys, xs = torch.where(mask_2d > 0.5)
    if ys.numel() == 0:
        return None
    return int(ys.min().item()), int(ys.max().item()), int(xs.min().item()), int(xs.max().item())


def _smoothstep_np(x: np.ndarray) -> np.ndarray:
    return x * x * (3.0 - 2.0 * x)


def _prepare_binary_source(
    binary: np.ndarray,
    source_smooth_px: int,
    closing_px: int,
    fill_holes: bool,
) -> np.ndarray:
    binary = binary.astype(bool)

    if fill_holes:
        binary = scipy.ndimage.binary_fill_holes(binary)

    if closing_px > 0:
        structure = np.ones((3, 3), dtype=bool)
        binary = scipy.ndimage.binary_closing(
            binary,
            structure=structure,
            iterations=int(closing_px),
        )

    if source_smooth_px > 0:
        sigma = max(float(source_smooth_px) / 3.0, 0.01)
        softened = scipy.ndimage.gaussian_filter(binary.astype(np.float32), sigma=sigma)
        binary = softened > 0.5

    if fill_holes:
        binary = scipy.ndimage.binary_fill_holes(binary)

    return binary


def _build_feathered_transition(
    base_crop_4d: torch.Tensor,
    expand: int,
    anchor_offset: int,
    feathering: int,
    source_smooth_px: int,
    closing_px: int,
    fill_holes: bool,
) -> torch.Tensor:
    binary = base_crop_4d.squeeze(0).squeeze(0).detach().cpu().numpy() > 0.5

    if not binary.any():
        return torch.zeros_like(base_crop_4d)

    binary = _prepare_binary_source(
        binary=binary,
        source_smooth_px=int(source_smooth_px),
        closing_px=int(closing_px),
        fill_holes=bool(fill_holes),
    )

    if not binary.any():
        return torch.zeros_like(base_crop_4d)

    dist_inside = scipy.ndimage.distance_transform_edt(binary)
    dist_outside = scipy.ndimage.distance_transform_edt(~binary)
    signed_dist = dist_inside - dist_outside

    anchor_threshold = -float(anchor_offset)
    final_threshold = -float(expand)

    white_threshold = max(anchor_threshold, final_threshold)
    black_threshold = min(anchor_threshold, final_threshold)

    if feathering <= 0 or abs(white_threshold - black_threshold) < 1e-6:
        out_np = (signed_dist >= final_threshold).astype(np.float32)
    else:
        t = (signed_dist - black_threshold) / (white_threshold - black_threshold)
        t = np.clip(t, 0.0, 1.0)

        out_np = _smoothstep_np(t).astype(np.float32)
        out_np[signed_dist >= white_threshold] = 1.0
        out_np[signed_dist <= black_threshold] = 0.0

    out = torch.from_numpy(out_np).to(
        device=base_crop_4d.device,
        dtype=base_crop_4d.dtype,
    ).unsqueeze(0).unsqueeze(0)

    return out.clamp_(0.0, 1.0)


class ExpandMaskBySides:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "expand_bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "expand_left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "expand_right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "expand_mask"
    CATEGORY = "StDismas/Mask"

    def expand_mask(self, mask, expand_top, expand_bottom, expand_left, expand_right):
        mask_4d = _ensure_mask_4d(mask).clamp_(0.0, 1.0)

        if expand_top == 0 and expand_bottom == 0 and expand_left == 0 and expand_right == 0:
            return (_restore_mask_dims(mask_4d, mask),)

        kernel_h = expand_top + expand_bottom + 1
        kernel_w = expand_left + expand_right + 1

        padded = F.pad(mask_4d, (expand_left, expand_right, expand_top, expand_bottom), mode="constant", value=0.0)
        expanded = F.max_pool2d(padded, kernel_size=(kernel_h, kernel_w), stride=1)
        expanded = expanded.clamp_(0.0, 1.0)

        return (_restore_mask_dims(expanded, mask),)


class ExpandMaskWithFeathering:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 16, "min": -8192, "max": 8192, "step": 1}),
                "anchor_offset": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "feathering": ("INT", {"default": 12, "min": 0, "max": 8192, "step": 1}),
                "source_smooth_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "closing_px": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "fill_holes": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("mask", "mask_inverted")
    FUNCTION = "expand_mask_with_feathering"
    CATEGORY = "StDismas/Mask"

    def _process_single_mask(
        self,
        mask_2d: torch.Tensor,
        expand: int,
        anchor_offset: int,
        feathering: int,
        source_smooth_px: int,
        closing_px: int,
        fill_holes: bool,
    ) -> torch.Tensor:
        bbox = _find_nonzero_bbox(mask_2d)
        if bbox is None:
            return torch.zeros_like(mask_2d)

        h, w = mask_2d.shape

        margin = (
            max(abs(expand), abs(anchor_offset))
            + max(source_smooth_px, closing_px * 2)
            + 4
        )

        y0, y1, x0, x1 = bbox
        y0 = max(0, y0 - margin)
        y1 = min(h - 1, y1 + margin)
        x0 = max(0, x0 - margin)
        x1 = min(w - 1, x1 + margin)

        crop = mask_2d[y0:y1 + 1, x0:x1 + 1].to(dtype=torch.float32)
        crop = (crop > 0.5).float().unsqueeze(0).unsqueeze(0)

        out_crop = _build_feathered_transition(
            base_crop_4d=crop,
            expand=int(expand),
            anchor_offset=int(anchor_offset),
            feathering=int(feathering),
            source_smooth_px=int(source_smooth_px),
            closing_px=int(closing_px),
            fill_holes=bool(fill_holes),
        )

        out = torch.zeros((h, w), dtype=torch.float32)
        out[y0:y1 + 1, x0:x1 + 1] = out_crop.squeeze(0).squeeze(0).cpu()

        return out

    def expand_mask_with_feathering(
        self,
        mask,
        expand,
        anchor_offset,
        feathering,
        source_smooth_px,
        closing_px,
        fill_holes,
    ):
        src = _ensure_mask_4d(mask).clamp(0.0, 1.0)
        batch = src.squeeze(1)

        outputs = []
        for i in range(batch.shape[0]):
            outputs.append(
                self._process_single_mask(
                    batch[i],
                    int(expand),
                    int(anchor_offset),
                    int(feathering),
                    int(source_smooth_px),
                    int(closing_px),
                    bool(fill_holes),
                )
            )

        out = torch.stack(outputs, dim=0).unsqueeze(1).clamp_(0.0, 1.0)
        out_inv = (1.0 - out).clamp_(0.0, 1.0)

        return (_restore_mask_dims(out, mask), _restore_mask_dims(out_inv, mask))


NODE_CLASS_MAPPINGS = {
    "ExpandMaskBySides": ExpandMaskBySides,
    "ExpandMaskWithFeathering": ExpandMaskWithFeathering,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExpandMaskBySides": "Expand Mask By Sides",
    "ExpandMaskWithFeathering": "Expand Mask With Feathering",
}