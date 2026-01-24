class CSVIndexPicker_StDismas:
    """CSV Index Picker (StDismas)

    Picks element by index from a delimiter-separated string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "values": ("STRING", {"default": "a, b, c", "multiline": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": ",", "multiline": False}),
                "strip_spaces": (["yes", "no"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "pick"
    CATEGORY = "Comfyui-StDismas/text"
    def pick(self, values, index, delimiter=",", strip_spaces="yes"):
        if not delimiter:
            delimiter = ","
        parts = values.split(delimiter)
        if strip_spaces == "yes":
            parts = [p.strip() for p in parts]
        parts = [p for p in parts if p != ""]
        if not parts:
            return ("",)
        if index < 0:
            index = 0
        if index >= len(parts):
            index = len(parts) - 1
        return (parts[index],)


NODE_CLASS_MAPPINGS = {"CSVIndexPicker_StDismas": CSVIndexPicker_StDismas}
NODE_DISPLAY_NAME_MAPPINGS = {"CSVIndexPicker_StDismas": "CSV Index Picker (StDismas)"}
