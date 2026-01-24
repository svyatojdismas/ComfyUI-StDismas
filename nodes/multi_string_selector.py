import json



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
    "MultiStringSelector_StDismas": MultiStringSelector_StDismas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiStringSelector_StDismas": "Multi String Selector (StDismas)",
}
