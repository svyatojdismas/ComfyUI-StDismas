import torch


BBOX_TYPE = "BBOXES"


def _ensure_mask_bhw(mask):
    if mask is None:
        raise ValueError("mask input is required")
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3:
        return mask
    raise ValueError(f"Expected MASK with 2 or 3 dims, got shape {tuple(mask.shape)}")


def _parse_rgb_string(color):
    if isinstance(color, str):
        parts = [p.strip() for p in color.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid color '{color}'. Use RGB format like 255, 0, 0"
            )
        try:
            rgb = [int(p) for p in parts]
        except ValueError as e:
            raise ValueError(
                f"Invalid color '{color}'. RGB values must be integers in range 0-255"
            ) from e
    else:
        raise ValueError("color must be a string in RGB format like 255, 0, 0")

    rgb = [max(0, min(255, c)) for c in rgb]
    return torch.tensor(rgb, dtype=torch.float32) / 255.0


def _mask_to_bbox_single(mask_2d):
    nz = torch.nonzero(mask_2d > 0, as_tuple=False)
    h, w = mask_2d.shape
    if nz.numel() == 0:
        cx = w // 2
        cy = h // 2
        return {
            "x0": int(cx),
            "y0": int(cy),
            "x1": int(cx),
            "y1": int(cy),
            "is_empty": True,
        }

    y0 = int(nz[:, 0].min().item())
    y1 = int(nz[:, 0].max().item()) + 1
    x0 = int(nz[:, 1].min().item())
    x1 = int(nz[:, 1].max().item()) + 1
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "is_empty": False,
    }


def _normalize_bbox_frame(frame, width, height):
    x0 = int(frame["x0"])
    y0 = int(frame["y0"])
    x1 = int(frame["x1"])
    y1 = int(frame["y1"])

    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))

    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    return {
        "x0": int(x0),
        "y0": int(y0),
        "x1": int(x1),
        "y1": int(y1),
        "is_empty": bool(frame.get("is_empty", False) or x0 == x1 or y0 == y1),
    }


class MaskToBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = (BBOX_TYPE,)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "mask_to_bbox"
    CATEGORY = "StDismas/BBox"

    def mask_to_bbox(self, mask):
        mask = _ensure_mask_bhw(mask)
        b, h, w = mask.shape

        frames = []
        for i in range(b):
            frame = _mask_to_bbox_single(mask[i])
            frame["index"] = i
            frames.append(frame)

        bbox = {
            "version": "std_bbox_v1",
            "source": "mask",
            "orig_size": [int(w), int(h)],
            "frames": frames,
        }
        return (bbox,)


class BBoxRescale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": (BBOX_TYPE,),
                "left": ("INT", {"default": 0, "min": -16384, "max": 16384, "step": 1}),
                "right": ("INT", {"default": 0, "min": -16384, "max": 16384, "step": 1}),
                "top": ("INT", {"default": 0, "min": -16384, "max": 16384, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": -16384, "max": 16384, "step": 1}),
                "color": ("STRING", {"default": "0, 0, 0", "multiline": False}),
                "inverted": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = (BBOX_TYPE, "MASK", "IMAGE")
    RETURN_NAMES = ("bbox", "mask", "image_masked")
    FUNCTION = "rescale_bbox"
    CATEGORY = "StDismas/BBox"

    def _resolve_size(self, bbox, image):
        if image is not None:
            _, h, w, _ = image.shape
            return int(w), int(h)

        orig_size = bbox.get("orig_size")
        if not orig_size or len(orig_size) != 2:
            raise ValueError("bbox does not contain orig_size, so image input is required")
        return int(orig_size[0]), int(orig_size[1])

    def _make_mask_and_preview(self, frames, width, height, batch, color_rgb, image=None, inverted=False):
        device = image.device if image is not None else torch.device("cpu")
        dtype = image.dtype if image is not None else torch.float32

        masks = torch.zeros((batch, height, width), device=device, dtype=dtype)

        for i, frame in enumerate(frames):
            x0, y0, x1, y1 = frame["x0"], frame["y0"], frame["x1"], frame["y1"]
            if x1 > x0 and y1 > y0:
                masks[i, y0:y1, x0:x1] = 1.0

        if inverted:
            masks = 1.0 - masks

        if image is not None:
            preview = image.clone()
        else:
            preview = torch.zeros((batch, height, width, 3), device=device, dtype=dtype)

        color_rgb = color_rgb.to(device=device, dtype=dtype).view(1, 1, 1, 3)
        alpha = masks.unsqueeze(-1).clamp(0.0, 1.0)
        preview = preview * (1.0 - alpha) + color_rgb * alpha
        preview = preview.clamp(0.0, 1.0)

        return masks, preview

    def rescale_bbox(self, bbox, left, right, top, bottom, color, inverted, image=None):
        if not isinstance(bbox, dict) or "frames" not in bbox:
            raise ValueError("Unsupported bbox format. Expected BBOXES metadata with frames.")

        color_rgb = _parse_rgb_string(color)
        width, height = self._resolve_size(bbox, image)
        frames_in = bbox.get("frames", [])

        if image is not None and image.shape[0] != len(frames_in):
            raise ValueError(
                f"Batch size mismatch: image batch={image.shape[0]}, bbox frames={len(frames_in)}"
            )

        frames_out = []
        for i, frame in enumerate(frames_in):
            src = _normalize_bbox_frame(frame, width, height)
            x0 = src["x0"] - int(left)
            x1 = src["x1"] + int(right)
            y0 = src["y0"] - int(top)
            y1 = src["y1"] + int(bottom)

            dst = _normalize_bbox_frame(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "is_empty": src.get("is_empty", False),
                },
                width,
                height,
            )
            dst["index"] = i
            frames_out.append(dst)

        bbox_out = {
            "version": "std_bbox_v1",
            "source": bbox.get("source", "bbox"),
            "orig_size": [int(width), int(height)],
            "frames": frames_out,
            "rescale": {
                "left": int(left),
                "right": int(right),
                "top": int(top),
                "bottom": int(bottom),
            },
        }

        batch = len(frames_out)
        masks, preview = self._make_mask_and_preview(
            frames_out,
            width,
            height,
            batch,
            color_rgb,
            image=image,
            inverted=bool(inverted),
        )

        return (bbox_out, masks, preview)


NODE_CLASS_MAPPINGS = {
    "MaskToBBox": MaskToBBox,
    "BBoxRescale": BBoxRescale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToBBox": "Mask To BBox",
    "BBoxRescale": "BBox Rescale",
}