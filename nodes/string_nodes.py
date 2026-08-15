import json


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


class MultiStringSelector_StDismas:
    """
    Multi String Selector (StDismasNodes)

    - Внутри ноды отображаются динамические string_1, string_2, ... (чистые поля, без сокетов).
    - На Python-стороне нода видит только:
        index       – какой элемент выбрать (0-базовый)
        values_json – JSON-список строк из UI
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                }),
                # Скрытое поле, в котором хранится JSON-массив строк.
                "values_json": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "select"
    CATEGORY = "Comfyui-StDismas/text"

    def select(self, index: int, values_json: str):
        try:
            values = json.loads(values_json)
            if not isinstance(values, list):
                values = [str(values)]
        except Exception:
            # Fallback: если вдруг там не JSON, попробуем парсить как CSV.
            values = [v.strip() for v in str(values_json).split(",") if v.strip()]

        if not values:
            return ("",)

        if index < 0:
            index = 0
        if index >= len(values):
            index = len(values) - 1

        return (str(values[index]),)


NODE_CLASS_MAPPINGS = {
    "CSVIndexPicker_StDismas": CSVIndexPicker_StDismas,
    "MultiStringSelector_StDismas": MultiStringSelector_StDismas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CSVIndexPicker_StDismas": "CSV Index Picker (StDismas)",
    "MultiStringSelector_StDismas": "Multi String Selector (StDismas)",
}
