import math
import torch
import torch.nn.functional as F

try:
    from comfy.utils import common_upscale
except Exception:
    common_upscale = None

MAX_RESOLUTION = 16384


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
    """
    y_idx, x_idx = torch.nonzero(mask2d > 0, as_tuple=True)
    if y_idx.numel() == 0 or x_idx.numel() == 0:
        return None
    min_y = int(y_idx.min().item())
    max_y = int(y_idx.max().item()) + 1
    min_x = int(x_idx.min().item())
    max_x = int(x_idx.max().item()) + 1
    return (min_x, min_y, max_x, max_y)


def _choose_upscale_method(in_w, in_h, out_w, out_h):
    # Keep it stable and predictable.
    # Lanczos is great for downscale (less aliasing); bicubic is nice for upscale.
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
        # Fallback: torch interpolate
        x = img_hwc.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return x.squeeze(0).permute(1, 2, 0)

    method = _choose_upscale_method(in_w, in_h, out_w, out_h)
    x = img_hwc.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
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
    x = mask_hw.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    x = F.interpolate(x, size=(out_h, out_w), mode="nearest")
    return x.squeeze(0).squeeze(0)


def _crop_with_padding_2d(mask_hw: torch.Tensor, x0: int, y0: int, out_w: int, out_h: int) -> torch.Tensor:
    """
    mask_hw: (H,W)
    returns (out_h,out_w) with zero padding when crop exceeds bounds
    """
    H, W = mask_hw.shape
    x1 = x0 + out_w
    y1 = y0 + out_h

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(W, x1)
    src_y1 = min(H, y1)

    out = torch.zeros((out_h, out_w), device=mask_hw.device, dtype=mask_hw.dtype)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = mask_hw[src_y0:src_y1, src_x0:src_x1]
    return out


def _crop_with_padding_3d(img_hwc: torch.Tensor, x0: int, y0: int, out_w: int, out_h: int) -> torch.Tensor:
    """
    img_hwc: (H,W,C)
    returns (out_h,out_w,C) with zero padding when crop exceeds bounds
    """
    H, W, C = img_hwc.shape
    x1 = x0 + out_w
    y1 = y0 + out_h

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(W, x1)
    src_y1 = min(H, y1)

    out = torch.zeros((out_h, out_w, C), device=img_hwc.device, dtype=img_hwc.dtype)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    out[dst_y0:dst_y1, dst_x0:dst_x1, :] = img_hwc[src_y0:src_y1, src_x0:src_x1, :]
    return out


def _paste_with_padding_2d(canvas_hw: torch.Tensor, patch_hw: torch.Tensor, x0: int, y0: int) -> torch.Tensor:
    """
    Paste patch into canvas at (x0,y0). Returns canvas.
    """
    H, W = canvas_hw.shape
    ph, pw = patch_hw.shape

    x1 = x0 + pw
    y1 = y0 + ph

    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(W, x1)
    dst_y1 = min(H, y1)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return canvas_hw

    src_x0 = dst_x0 - x0
    src_y0 = dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    canvas_hw[dst_y0:dst_y1, dst_x0:dst_x1] = patch_hw[src_y0:src_y1, src_x0:src_x1]
    return canvas_hw


def _paste_with_padding_3d(canvas_hwc: torch.Tensor, patch_hwc: torch.Tensor, x0: int, y0: int) -> torch.Tensor:
    """
    Paste patch into canvas at (x0,y0). Returns canvas.
    """
    H, W, C = canvas_hwc.shape
    ph, pw, pc = patch_hwc.shape
    if pc != C:
        raise ValueError("Patch channels mismatch")

    x1 = x0 + pw
    y1 = y0 + ph

    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(W, x1)
    dst_y1 = min(H, y1)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return canvas_hwc

    src_x0 = dst_x0 - x0
    src_y0 = dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    canvas_hwc[dst_y0:dst_y1, dst_x0:dst_x1, :] = patch_hwc[src_y0:src_y1, src_x0:src_x1, :]
    return canvas_hwc


def _feather_alpha(alpha_hw: torch.Tensor, feather_px: int) -> torch.Tensor:
    """
    alpha_hw: (h,w) in [0,1]
    feather_px: blur radius in pixels
    """
    if feather_px <= 0:
        return alpha_hw.clamp(0.0, 1.0)

    # Fast blur approximation using average pooling (box blur) repeated.
    # This is torch-only (no PIL), stable for ComfyUI portable envs.
    k = feather_px * 2 + 1
    x = alpha_hw.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    pad = feather_px
    # 2 passes makes it closer to gaussian-ish
    x = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=k, stride=1)
    x = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=k, stride=1)
    return x.squeeze(0).squeeze(0).clamp(0.0, 1.0)


class BatchImageCropByMaskAdvanced_StDismas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BBOXES")
    RETURN_NAMES = ("cropped_images", "cropped_masks", "bboxes")
    FUNCTION = "crop"
    CATEGORY = "Comfyui-StDismas/masking"

    """
    ЛОГИКА КРОПА — та же идея, что и раньше:
    - bbox по маске + padding
    - масштабируем изображение (uniform) так, чтобы bbox+padding полностью помещался
      в целевой размер width/height без искажений (zoom in/out).
    - кроп вырезается в масштабированном пространстве, при выходе за границы добавляется черный pad.
    """

    def crop(self, images, masks, width, height, padding):
        B, H, W, C = images.shape
        masks = _ensure_mask_hw(masks, H, W)

        out_imgs = []
        out_masks = []
        out_bboxes = []

        device = images.device
        dtype = images.dtype

        for i in range(B):
            mask_i = masks[i]
            bb = _mask_bbox(mask_i)
            if bb is None:
                # пустая маска: центрируем кроп без искажений, добавляя паддинг если нужно
                scale = 1.0
                scaled_w = W
                scaled_h = H
                cx_s = W * 0.5
                cy_s = H * 0.5
                crop_x0 = int(round(cx_s - width / 2))
                crop_y0 = int(round(cy_s - height / 2))

                crop_img = _crop_with_padding_3d(images[i], crop_x0, crop_y0, width, height)
                crop_m = _crop_with_padding_2d(mask_i, crop_x0, crop_y0, width, height)

                out_imgs.append(crop_img)
                out_masks.append(crop_m)
                out_bboxes.append({
                    "scale": float(scale),
                    "scaled_w": int(scaled_w),
                    "scaled_h": int(scaled_h),
                    "crop_x0": int(crop_x0),
                    "crop_y0": int(crop_y0),
                    "crop_w": int(width),
                    "crop_h": int(height),
                    "orig_w": int(W),
                    "orig_h": int(H),
                })
                continue

            min_x, min_y, max_x, max_y = bb

            # apply padding to bbox
            min_xp = max(0, min_x - padding)
            min_yp = max(0, min_y - padding)
            max_xp = min(W, max_x + padding)
            max_yp = min(H, max_y + padding)

            bbox_w = max_xp - min_xp
            bbox_h = max_yp - min_yp

            bbox_w = max(1, bbox_w)
            bbox_h = max(1, bbox_h)

            # scale uniformly so bbox+padding fully fits in target size
            scale = min(width / bbox_w, height / bbox_h)
            scale = max(scale, 1e-6)

            scaled_w = max(1, int(round(W * scale)))
            scaled_h = max(1, int(round(H * scale)))

            scaled_img = _resize_image(images[i], scaled_w, scaled_h)
            scaled_mask = _resize_mask(mask_i, scaled_w, scaled_h)

            cx = (min_xp + max_xp) * 0.5
            cy = (min_yp + max_yp) * 0.5
            cx_s = cx * scale
            cy_s = cy * scale

            crop_x0 = int(round(cx_s - width / 2))
            crop_y0 = int(round(cy_s - height / 2))

            crop_img = _crop_with_padding_3d(scaled_img, crop_x0, crop_y0, width, height)
            crop_m = _crop_with_padding_2d(scaled_mask, crop_x0, crop_y0, width, height)

            out_imgs.append(crop_img)
            out_masks.append(crop_m)

            out_bboxes.append({
                "scale": float(scale),
                "scaled_w": int(scaled_w),
                "scaled_h": int(scaled_h),
                "crop_x0": int(crop_x0),
                "crop_y0": int(crop_y0),
                "crop_w": int(width),
                "crop_h": int(height),
                "orig_w": int(W),
                "orig_h": int(H),
            })

        out_imgs = torch.stack(out_imgs, dim=0).to(device=device, dtype=dtype)
        out_masks = torch.stack(out_masks, dim=0).to(device=device, dtype=dtype)

        return (out_imgs, out_masks, out_bboxes)


class BatchImageUncropByMaskAdvanced_StDismas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_images": ("IMAGE",),
                "cropped_masks": ("MASK",),
                "bboxes": ("BBOXES",),
                "border_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_rescale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01}),
                "use_square_mask": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "uncrop"
    CATEGORY = "Comfyui-StDismas/masking"

    """
    Здесь ВАЖНОЕ отличие от простой версии:
    - Мы НЕ пересчитываем bbox по маске.
    - Мы используем bboxes, полученные из Crop Advanced.
    """

    def uncrop(self, original_images, cropped_images, cropped_masks, bboxes,
               border_blending=0.25, crop_rescale=1.0, use_square_mask=True):
        B, H, W, C = original_images.shape
        Bc, Hc, Wc, Cc = cropped_images.shape

        if Bc != B:
            raise ValueError(f"Batch size mismatch: original_images={B}, cropped_images={Bc}")

        cropped_masks = _ensure_mask_hw(cropped_masks, Hc, Wc)

        # bboxes may be a single bbox reused (edge cases) or list length B
        if isinstance(bboxes, (list, tuple)):
            if len(bboxes) == 1 and B > 1:
                bboxes_use = [bboxes[0] for _ in range(B)]
            elif len(bboxes) == B:
                bboxes_use = list(bboxes)
            else:
                raise ValueError(f"bboxes length must be 1 or B({B}), got {len(bboxes)}")
        else:
            # fallback: single object
            bboxes_use = [bboxes for _ in range(B)]

        device = original_images.device
        dtype = original_images.dtype

        out = original_images.clone()

        for i in range(B):
            info = bboxes_use[i]
            if "scale" in info:
                scale = float(info.get("scale", 1.0))
                scaled_w = int(info.get("scaled_w", int(round(W * scale))))
                scaled_h = int(info.get("scaled_h", int(round(H * scale))))
                crop_x0 = int(info.get("crop_x0", 0))
                crop_y0 = int(info.get("crop_y0", 0))
                crop_w = int(info.get("crop_w", Wc))
                crop_h = int(info.get("crop_h", Hc))

                crop_w = max(1, crop_w)
                crop_h = max(1, crop_h)

                # rescale patch within crop window if needed
                tgt_w = max(1, int(round(crop_w * float(crop_rescale))))
                tgt_h = max(1, int(round(crop_h * float(crop_rescale))))

                patch = _resize_image(cropped_images[i], tgt_w, tgt_h)
                if use_square_mask:
                    alpha = torch.ones((tgt_h, tgt_w), device=device, dtype=dtype)
                else:
                    alpha = _resize_mask(cropped_masks[i], tgt_w, tgt_h).to(device=device, dtype=dtype)

                # center patch into crop-sized canvas
                crop_patch = torch.zeros((crop_h, crop_w, 3), device=device, dtype=dtype)
                crop_alpha = torch.zeros((crop_h, crop_w), device=device, dtype=dtype)

                place_w = min(tgt_w, crop_w)
                place_h = min(tgt_h, crop_h)
                px0 = max(0, (tgt_w - place_w) // 2)
                py0 = max(0, (tgt_h - place_h) // 2)

                patch = patch[py0:py0 + place_h, px0:px0 + place_w, :]
                alpha = alpha[py0:py0 + place_h, px0:px0 + place_w]

                ox = (crop_w - place_w) // 2
                oy = (crop_h - place_h) // 2
                crop_patch[oy:oy + place_h, ox:ox + place_w, :] = patch
                crop_alpha[oy:oy + place_h, ox:ox + place_w] = alpha

                scaled_patch = torch.zeros((scaled_h, scaled_w, 3), device=device, dtype=dtype)
                scaled_alpha = torch.zeros((scaled_h, scaled_w), device=device, dtype=dtype)

                scaled_patch = _paste_with_padding_3d(scaled_patch, crop_patch, crop_x0, crop_y0)
                scaled_alpha = _paste_with_padding_2d(scaled_alpha, crop_alpha, crop_x0, crop_y0)

                # resize back to original resolution
                patch_full = _resize_image(scaled_patch, W, H)
                alpha_full = _resize_mask(scaled_alpha, W, H).to(device=device, dtype=dtype)

                feather_px = int(round(float(border_blending) * 32.0))
                alpha_full = _feather_alpha(alpha_full, feather_px)

                alpha3 = alpha_full.unsqueeze(-1).expand(-1, -1, 3)
                out[i] = out[i] * (1.0 - alpha3) + patch_full * alpha3
            else:
                x0 = int(info["x0"]); y0 = int(info["y0"]); x1 = int(info["x1"]); y1 = int(info["y1"])
                win_w = int(info.get("win_w", x1 - x0))
                win_h = int(info.get("win_h", y1 - y0))

                # window sanity
                x0 = max(0, min(x0, W))
                x1 = max(0, min(x1, W))
                y0 = max(0, min(y0, H))
                y1 = max(0, min(y1, H))

                win_w = max(1, x1 - x0)
                win_h = max(1, y1 - y0)

                # rescale patch within window if needed
                tgt_w = max(1, int(round(win_w * float(crop_rescale))))
                tgt_h = max(1, int(round(win_h * float(crop_rescale))))

                # resize cropped image/mask from (Wc,Hc) to (tgt_w,tgt_h)
                patch = _resize_image(cropped_images[i], tgt_w, tgt_h)
                if use_square_mask:
                    alpha = torch.ones((tgt_h, tgt_w), device=device, dtype=dtype)
                else:
                    alpha = _resize_mask(cropped_masks[i], tgt_w, tgt_h).to(device=device, dtype=dtype)

                # border blending: interpret 0..1 into a practical pixel feather width
                # 0.25 -> ~8px, 1.0 -> ~32px
                feather_px = int(round(float(border_blending) * 32.0))
                alpha = _feather_alpha(alpha, feather_px)

                # paste coords (centered in the window if rescaled)
                # base window is [x0:x1, y0:y1]
                # if tgt bigger than window -> clamp & center crop
                dst_x0, dst_y0 = x0, y0
                dst_x1, dst_y1 = x1, y1

                # If rescaled patch differs from window size, we center it.
                if tgt_w != win_w or tgt_h != win_h:
                    # compute centered placement inside the window bounds
                    place_w = min(tgt_w, win_w)
                    place_h = min(tgt_h, win_h)

                    # crop patch if larger
                    px0 = max(0, (tgt_w - place_w) // 2)
                    py0 = max(0, (tgt_h - place_h) // 2)
                    patch = patch[py0:py0 + place_h, px0:px0 + place_w, :]
                    alpha = alpha[py0:py0 + place_h, px0:px0 + place_w]

                    # place into window centered
                    ox = (win_w - place_w) // 2
                    oy = (win_h - place_h) // 2
                    dst_x0 = x0 + ox
                    dst_y0 = y0 + oy
                    dst_x1 = dst_x0 + place_w
                    dst_y1 = dst_y0 + place_h

                # final safety clamp
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
                alpha3 = alpha.unsqueeze(-1).expand(-1, -1, 3)

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
