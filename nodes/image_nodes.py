import re
import torch
import torch.nn.functional as F


class ReplaceImagesInBatchIndexed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "original_images": ("IMAGE",),
                "replacement_images": ("IMAGE",),
                "original_masks": ("MASK",),
                "replacement_masks": ("MASK",),
                "indices_text": ("STRING", {"multiline": False, "default": "0, 1-3, 5"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "replace_batches"
    CATEGORY = "StDismas/Image"

    def parse_indices(self, text):
        if text is None or str(text).strip() == "":
            return []

        parts = str(text).split(",")
        indices = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            range_match = re.match(r"^(\d+)\s*-\s*(\d+)$", part)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                if start > end:
                    raise ValueError(
                        f"Invalid range '{part}': range start must be less than or equal to range end"
                    )
                indices.extend(range(start, end + 1))
                continue

            if re.match(r"^\d+$", part):
                indices.append(int(part))
                continue

            raise ValueError(
                f"Invalid index entry: '{part}'. Use comma-separated integers and/or ranges like 0, 3, 5-8"
            )

        return indices

    def resize_image_to_match(self, source_img, target_img):
        if source_img.shape == target_img.shape:
            return source_img

        return F.interpolate(
            source_img.unsqueeze(0).permute(0, 3, 1, 2),
            size=(target_img.shape[0], target_img.shape[1]),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).squeeze(0)

    def resize_mask_to_match(self, source_mask, target_mask):
        if source_mask.shape == target_mask.shape:
            return source_mask

        return F.interpolate(
            source_mask.unsqueeze(0).unsqueeze(0),
            size=(target_mask.shape[0], target_mask.shape[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    def validate_inputs(self, original_images, replacement_images, original_masks, replacement_masks, indices):
        has_any_data = any(x is not None for x in [original_images, replacement_images, original_masks, replacement_masks])
        if not has_any_data:
            raise ValueError("At least one input batch must be connected.")

        if replacement_images is not None and original_images is None:
            raise ValueError("replacement_images is connected, but original_images is missing.")

        if replacement_masks is not None and original_masks is None:
            raise ValueError("replacement_masks is connected, but original_masks is missing.")

        if replacement_images is not None and len(indices) != replacement_images.shape[0]:
            raise ValueError(
                f"Number of parsed indices ({len(indices)}) must match number of replacement images ({replacement_images.shape[0]}). "
                f"Parsed indices: {indices}"
            )

        if replacement_masks is not None and len(indices) != replacement_masks.shape[0]:
            raise ValueError(
                f"Number of parsed indices ({len(indices)}) must match number of replacement masks ({replacement_masks.shape[0]}). "
                f"Parsed indices: {indices}"
            )

        if original_images is not None:
            for idx in indices:
                if idx < 0 or idx >= original_images.shape[0]:
                    raise IndexError(
                        f"Image index {idx} is out of bounds for original_images batch size {original_images.shape[0]}"
                    )

        if original_masks is not None:
            for idx in indices:
                if idx < 0 or idx >= original_masks.shape[0]:
                    raise IndexError(
                        f"Mask index {idx} is out of bounds for original_masks batch size {original_masks.shape[0]}"
                    )

    def replace_batches(
        self,
        original_images=None,
        replacement_images=None,
        original_masks=None,
        replacement_masks=None,
        indices_text="",
    ):
        indices = self.parse_indices(indices_text)

        self.validate_inputs(
            original_images,
            replacement_images,
            original_masks,
            replacement_masks,
            indices,
        )

        output_images = original_images.clone() if original_images is not None else None
        output_masks = original_masks.clone() if original_masks is not None else None

        if replacement_images is not None:
            for i, idx in enumerate(indices):
                rep_img = self.resize_image_to_match(replacement_images[i], output_images[idx])
                output_images[idx] = rep_img

        if replacement_masks is not None:
            for i, idx in enumerate(indices):
                rep_mask = self.resize_mask_to_match(replacement_masks[i], output_masks[idx])
                output_masks[idx] = rep_mask

        return (output_images, output_masks)


NODE_CLASS_MAPPINGS = {
    "ReplaceImagesInBatchIndexed": ReplaceImagesInBatchIndexed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplaceImagesInBatchIndexed": "Replace Images In Batch Indexed",
}
