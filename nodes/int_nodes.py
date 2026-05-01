class IntDivisibleBy_StDismas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "divisible_by": ("INT", {"default": 1, "min": 1, "max": 2147483647, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "Comfyui-StDismas/Utils/Int"

    def run(self, value, divisible_by):
        divisible_by = max(1, int(divisible_by))
        value = max(0, int(value))
        value = round(value / divisible_by) * divisible_by
        return (int(value),)


class SetDimension_StDismas:
    FORMATS = ["none", "LTXV", "WAN", "F.Klein"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": 0, "max": 30720, "step": 1}),
                "format": (cls.FORMATS, {"default": "none"}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "Comfyui-StDismas/Utils/Int"

    @staticmethod
    def _rules(format_name):
        if format_name == "LTXV":
            return 32, 0, 0, 30720
        if format_name == "WAN":
            return 8, 0, 0, 30720
        if format_name == "F.Klein":
            return 16, 0, 0, 30720
        return 1, 0, 0, 30720

    def run(self, value, format):
        step, mod, min_v, max_v = self._rules(format)
        value = max(min_v, min(max_v, int(value)))
        value = int(round((value - mod) / step) * step + mod)
        value = max(min_v, min(max_v, value))
        return (value,)


class SetDuration_StDismas:
    FORMATS = ["none", "LTXV", "WAN"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 1, "min": 0, "max": 2147483647, "step": 1}),
                "format": (cls.FORMATS, {"default": "none"}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frame_count",)
    FUNCTION = "run"
    CATEGORY = "Comfyui-StDismas/Utils/Int"

    @staticmethod
    def _rules(format_name):
        if format_name == "LTXV":
            return 8, 1, 1, 2147483647
        if format_name == "WAN":
            return 4, 1, 1, 2147483647
        return 1, 0, 0, 2147483647

    def run(self, value, format):
        step, mod, min_v, max_v = self._rules(format)
        value = max(min_v, min(max_v, int(value)))
        if step == 1 and mod == 0:
            return (value,)
        value = int(round((value - mod) / step) * step + mod)
        value = max(min_v, min(max_v, value))
        return (value,)


NODE_CLASS_MAPPINGS = {
    "IntDivisibleBy_StDismas": IntDivisibleBy_StDismas,
    "SetDimension_StDismas": SetDimension_StDismas,
    "SetDuration_StDismas": SetDuration_StDismas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntDivisibleBy_StDismas": "Int Divisible By (StDismas)",
    "SetDimension_StDismas": "Set Dimension (StDismas)",
    "SetDuration_StDismas": "Set Duration (StDismas)",
}
