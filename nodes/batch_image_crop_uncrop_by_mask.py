import torch
import torch.nn.functional as F
from comfy.utils import common_upscale

from PIL import Image, ImageFilter, ImageDraw
import numpy as np

MAX_RESOLUTION = 8192


# =========================
# Helpers (PIL <-> Tensor)
# =========================
def tensor2pil(images: torch.Tensor):
    """
    images: [B,H,W,C] float 0..1
    returns: list[PIL.Image] RGB
    """
    if images is None:
        return []
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    images = images.detach().cpu().clamp(0, 1)
    out = []
    for i in range(images.shape[0]):
        img = (images[i].numpy() * 255.0).astype(np.uint8)
        out.append(Image.fromarray(img, mode="RGB"))
    return out


def mask2pil(masks: torch.Tensor):
    """
    masks: [B,H,W] float 0..1
    returns: list[PIL.Image] L
    """
    if masks is None:
        return []
    if len(masks.shape) == 2:
        masks = masks.unsqueeze(0)
    masks = masks.detach().cpu().clamp(0, 1)
    out = []
    for i in range(masks.shape[0]):
        m = (masks[i].numpy() * 255.0).astype(np.uint8)
        out.append(Image.fromarray(m, mode="L"))
    return out


def pil2tensor(images):
    """
    images: list[PIL.Image] RGB
    returns: [B,H,W,3] float 0..1
    """
    if not images:
        return torch.zeros((0, 0, 0, 3), dtype=torch.float32)
    arr = []
    for im in images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        a = np.array(im).astype(np.float32) / 255.0
        arr.append(torch.from_numpy(a))
    return torch.stack(arr, dim=0)


# =========================
# Crop Node
# =========================
class BatchImageCropByMask_StDismas:
    """
    Batch Image Crop By Mask (StDismas)

    - Builds an aspect-matched crop window (aspect = width/height) that CONTAINS the padded mask bbox.
    - Crops that window from the original frame and resizes directly to (width,height).
      => output always fully filled (no black bars), and bbox always inside crop.
    - Correct frame-to-mask pairing (animated masks work).
    """

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

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "crop"
    CATEGORY = "Comfyui-StDismas/masking"
    DESCRIPTION = "Aspect-matched crop by mask. No bars. BBox guaranteed inside output."

    @staticmethod
    def _ensure_mask_res(masks: torch.Tensor, H: int, W: int) -> torch.Tensor:
        BM, HM, WM = masks.shape
        if HM != H or WM != W:
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode="nearest-exact").squeeze(1)
        return masks

    @staticmethod
    def _bbox_from_mask(mask2d: torch.Tensor, H: int, W: int, padding: int):
        y_idx, x_idx = torch.nonzero(mask2d > 0, as_tuple=True)
        if y_idx.numel() == 0 or x_idx.numel() == 0:
            cy, cx = H // 2, W // 2
            min_y = max(0, cy - 1)
            max_y = min(H, cy + 2)
            min_x = max(0, cx - 1)
            max_x = min(W, cx + 2)
        else:
            min_y = max(0, int(y_idx.min().item()) - padding)
            max_y = min(H, int(y_idx.max().item()) + 1 + padding)
            min_x = max(0, int(x_idx.min().item()) - padding)
            max_x = min(W, int(x_idx.max().item()) + 1 + padding)
        return min_x, min_y, max_x, max_y

    @staticmethod
    def _aspect_window_containing_bbox(min_x, min_y, max_x, max_y, W, H, target_w, target_h):
        bw = max(1, max_x - min_x)
        bh = max(1, max_y - min_y)

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        target_aspect = float(target_w) / float(target_h)
        bbox_aspect = float(bw) / float(bh)

        if bbox_aspect < target_aspect:
            win_h = bh
            win_w = int(torch.ceil(torch.tensor(win_h * target_aspect)).item())
        else:
            win_w = bw
            win_h = int(torch.ceil(torch.tensor(win_w / target_aspect)).item())

        if win_w > W:
            win_w = W
            win_h = int(torch.floor(torch.tensor(win_w / target_aspect)).item())
            win_h = max(1, min(win_h, H))
        if win_h > H:
            win_h = H
            win_w = int(torch.floor(torch.tensor(win_h * target_aspect)).item())
            win_w = max(1, min(win_w, W))

        x0 = int(round(cx - win_w / 2.0))
        y0 = int(round(cy - win_h / 2.0))
        x1 = x0 + win_w
        y1 = y0 + win_h

        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > W:
            dx = x1 - W
            x0 -= dx
            x1 = W
        if y1 > H:
            dy = y1 - H
            y0 -= dy
            y1 = H

        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        x1 = max(x0 + 1, min(x1, W))
        y1 = max(y0 + 1, min(y1, H))

        contains = (x0 <= min_x) and (y0 <= min_y) and (x1 >= max_x) and (y1 >= max_y)
        return x0, y0, x1, y1, contains

    @staticmethod
    def _tight_bbox_window(min_x, min_y, max_x, max_y, W, H):
        x0 = max(0, min_x)
        y0 = max(0, min_y)
        x1 = min(W, max_x)
        y1 = min(H, max_y)
        x1 = max(x0 + 1, x1)
        y1 = max(y0 + 1, y1)
        return x0, y0, x1, y1

    def crop(self, images, masks, width, height, padding):
        B, H, W, _ = images.shape
        BM, _, _ = masks.shape
        masks = self._ensure_mask_res(masks, H, W)

        if BM == B:
            count = B
            img_idx = lambda i: i
            mask_idx = lambda i: i
        elif BM == 1:
            count = B
            img_idx = lambda i: i
            mask_idx = lambda i: 0
        elif B == 1 and BM > 1:
            count = BM
            img_idx = lambda i: 0
            mask_idx = lambda i: i
        else:
            count = min(B, BM)
            img_idx = lambda i: i
            mask_idx = lambda i: i

        out_w = int(width)
        out_h = int(height)
        pad = int(padding)

        out_imgs = []
        out_masks = []

        for i in range(count):
            img = images[img_idx(i)]
            m = masks[mask_idx(i)]

            min_x, min_y, max_x, max_y = self._bbox_from_mask(m, H, W, pad)

            x0, y0, x1, y1, contains = self._aspect_window_containing_bbox(
                min_x, min_y, max_x, max_y, W, H, out_w, out_h
            )
            if not contains:
                x0, y0, x1, y1 = self._tight_bbox_window(min_x, min_y, max_x, max_y, W, H)

            crop_img = img[y0:y1, x0:x1, :]
            crop_mask = m[y0:y1, x0:x1]

            crop_img_nchw = crop_img.permute(2, 0, 1).unsqueeze(0)
            resized_img = common_upscale(crop_img_nchw, out_w, out_h, "lanczos", "disabled").squeeze(0).permute(1, 2, 0)

            crop_mask_nchw = crop_mask.unsqueeze(0).unsqueeze(0)
            resized_mask = F.interpolate(crop_mask_nchw, size=(out_h, out_w), mode="nearest").squeeze(0).squeeze(0)

            out_imgs.append(resized_img)
            out_masks.append(resized_mask)

        if not out_imgs:
            device = images.device
            return (
                torch.zeros((0, out_h, out_w, 3), dtype=images.dtype, device=device),
                torch.zeros((0, out_h, out_w), dtype=masks.dtype, device=device),
            )

        return (torch.stack(out_imgs, dim=0), torch.stack(out_masks, dim=0))


# =========================
# Uncrop Node (stitch back)
# =========================
class BatchImageUncropByMask_StDismas:
    """
    Batch Image Uncrop By Mask (StDismas)

    Rebuilds the SAME crop window (from original masks + width/height/padding),
    then stitches processed cropped_images back into original_images with feather blending
    (like KJNodes Batch Uncrop Advanced).

    Inputs:
    - original_images: IMAGE (full frames)
    - cropped_images: IMAGE (processed crops, same W/H as used in crop node)
    - masks: MASK (original masks in full-frame space; animated ok)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_images": ("IMAGE",),
                "masks": ("MASK",),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "border_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_rescale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "use_square_mask": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "uncrop"
    CATEGORY = "Comfyui-StDismas/masking"
    DESCRIPTION = "Stitches processed crops back into originals using the same mask-derived crop window."

    # reuse same window logic as crop
    _ensure_mask_res = staticmethod(BatchImageCropByMask_StDismas._ensure_mask_res)
    _bbox_from_mask = staticmethod(BatchImageCropByMask_StDismas._bbox_from_mask)
    _aspect_window_containing_bbox = staticmethod(BatchImageCropByMask_StDismas._aspect_window_containing_bbox)
    _tight_bbox_window = staticmethod(BatchImageCropByMask_StDismas._tight_bbox_window)

    @staticmethod
    def _inset_border(mask_block: Image.Image, border_width: int, border_color=0) -> Image.Image:
        if border_width <= 0:
            return mask_block
        w, h = mask_block.size
        draw = ImageDraw.Draw(mask_block)
        draw.rectangle((0, 0, w - 1, h - 1), outline=border_color, width=border_width)
        return mask_block

    @staticmethod
    def _clamp_region(x0, y0, x1, y1, W, H):
        x0 = max(0, min(x0, W - 1))
        y0 = max(0, min(y0, H - 1))
        x1 = max(x0 + 1, min(x1, W))
        y1 = max(y0 + 1, min(y1, H))
        return x0, y0, x1, y1

    def uncrop(self, original_images, cropped_images, masks, width, height, padding, border_blending, crop_rescale, use_square_mask):
        if len(original_images.shape) != 4 or original_images.shape[-1] != 3:
            raise ValueError("original_images must be IMAGE tensor [B,H,W,3]")
        if len(cropped_images.shape) != 4 or cropped_images.shape[-1] != 3:
            raise ValueError("cropped_images must be IMAGE tensor [B,h,w,3]")
        if len(masks.shape) != 3:
            raise ValueError("masks must be MASK tensor [B,H,W] or [1,H,W]")

        B, H, W, _ = original_images.shape
        BM, _, _ = masks.shape

        masks = self._ensure_mask_res(masks, H, W)

        # Pairing logic across original/cropped/masks
        # Prefer strict 1:1; otherwise broadcast mask or crop if needed.
        def idx_map(n_src, n_dst):
            if n_src == n_dst:
                return lambda i: i
            if n_src == 1:
                return lambda i: 0
            return lambda i: min(i, n_src - 1)

        orig_i = idx_map(B, B)
        crop_i = idx_map(cropped_images.shape[0], B)
        mask_i = idx_map(BM, B)

        out_w = int(width)
        out_h = int(height)
        pad = int(padding)

        # Convert to PIL for KJNodes-like blending quality
        orig_pil = tensor2pil(original_images)
        crop_pil = tensor2pil(cropped_images)
        masks_pil_full = mask2pil(masks)

        # clamp blending
        border_blending = float(max(0.0, min(1.0, border_blending)))
        crop_rescale = float(max(0.01, crop_rescale))

        out = []

        for i in range(B):
            img = orig_pil[orig_i(i)]
            crop_img = crop_pil[crop_i(i)]
            mask_full = masks_pil_full[mask_i(i)]

            # recompute crop window exactly like crop node
            # NOTE: bbox from tensor mask (more reliable), so use masks tensor for bbox
            m_tensor = masks[mask_i(i)]
            min_x, min_y, max_x, max_y = self._bbox_from_mask(m_tensor, H, W, pad)

            x0, y0, x1, y1, contains = self._aspect_window_containing_bbox(
                min_x, min_y, max_x, max_y, W, H, out_w, out_h
            )
            if not contains:
                x0, y0, x1, y1 = self._tight_bbox_window(min_x, min_y, max_x, max_y, W, H)

            # optional rescale of paste region (same idea as KJNodes)
            if crop_rescale != 1.0:
                x0 = round(x0 * crop_rescale)
                y0 = round(y0 * crop_rescale)
                x1 = round(x1 * crop_rescale)
                y1 = round(y1 * crop_rescale)
                x0, y0, x1, y1 = self._clamp_region(x0, y0, x1, y1, W, H)

            paste_w = max(1, x1 - x0)
            paste_h = max(1, y1 - y0)
            paste_region = (x0, y0, x1, y1)

            # resize processed crop to paste region
            crop_resized = crop_img.resize((paste_w, paste_h), resample=Image.Resampling.LANCZOS).convert("RGB")

            # border blending amount (KJNodes-like)
            blend_ratio = (max(paste_w, paste_h) / 2.0) * border_blending
            blur_r = max(0.0, blend_ratio / 4.0)
            border_w = int(round(blend_ratio / 2.0))

            # build alpha mask on full image size
            if use_square_mask:
                alpha = Image.new("L", img.size, 0)
                block = Image.new("L", (paste_w, paste_h), 255)
                block = self._inset_border(block, border_w, 0)
                alpha.paste(block, paste_region)
            else:
                # use original (full-frame) mask, but restrict to paste region
                # and resize it to paste size (so alpha aligns with pasted crop)
                m_crop = mask_full.crop(paste_region).resize((paste_w, paste_h), resample=Image.Resampling.NEAREST)
                alpha = Image.new("L", img.size, 0)
                alpha.paste(m_crop, paste_region)

            # soften edges
            if blur_r > 0:
                alpha = alpha.filter(ImageFilter.BoxBlur(radius=blur_r))
                alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_r))

            # composite
            base_rgba = img.convert("RGBA")
            blend = Image.new("RGBA", img.size, (0, 0, 0, 0))
            blend.paste(crop_resized, paste_region)
            blend.putalpha(alpha)

            out_img = Image.alpha_composite(base_rgba, blend).convert("RGB")
            out.append(out_img)

        return (pil2tensor(out),)


NODE_CLASS_MAPPINGS = {
    "BatchImageCropByMask_StDismas": BatchImageCropByMask_StDismas,
    "BatchImageUncropByMask_StDismas": BatchImageUncropByMask_StDismas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageCropByMask_StDismas": "Batch Image Crop By Mask (StDismas)",
    "BatchImageUncropByMask_StDismas": "Batch Image Uncrop By Mask (StDismas)",
}
