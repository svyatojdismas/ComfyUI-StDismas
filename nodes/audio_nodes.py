import torch


class ExtendEmptyAudio:
    """
    Adds silence to the beginning or end of a ComfyUI AUDIO object.

    ComfyUI AUDIO format:
        {
            "waveform": torch.Tensor,  # [..., samples]
            "sample_rate": int
        }
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "add duration": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": 0.00,
                        "max": 1000.00,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "position": (
                    ["before", "after"],
                    {"default": "after"},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "extend_audio"
    CATEGORY = "StDismas/Audio"

    def extend_audio(self, audio, position="after", **kwargs):
        add_duration = float(kwargs.get("add duration", 0.0))

        if audio is None:
            raise ValueError("Extend Empty Audio: input audio is None.")

        if not isinstance(audio, dict):
            raise TypeError(
                "Extend Empty Audio: AUDIO input must be a dictionary containing "
                "'waveform' and 'sample_rate'."
            )

        if "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError(
                "Extend Empty Audio: AUDIO input must contain both "
                "'waveform' and 'sample_rate'."
            )

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if not torch.is_tensor(waveform):
            raise TypeError(
                "Extend Empty Audio: audio['waveform'] must be a torch.Tensor."
            )

        if waveform.ndim < 1:
            raise ValueError(
                "Extend Empty Audio: audio waveform must have a samples dimension."
            )

        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Extend Empty Audio: audio['sample_rate'] must be an integer."
            ) from error

        if sample_rate <= 0:
            raise ValueError(
                "Extend Empty Audio: audio['sample_rate'] must be greater than zero."
            )

        if add_duration < 0:
            raise ValueError(
                "Extend Empty Audio: add duration cannot be negative."
            )

        if position not in {"before", "after"}:
            raise ValueError(
                "Extend Empty Audio: position must be either 'before' or 'after'."
            )

        silence_samples = int(round(add_duration * sample_rate))

        # Preserve the original AUDIO object and any extra metadata it may contain.
        output_audio = dict(audio)
        output_audio["sample_rate"] = sample_rate

        if silence_samples == 0:
            output_audio["waveform"] = waveform
            return (output_audio,)

        silence_shape = (*waveform.shape[:-1], silence_samples)
        silence = torch.zeros(
            silence_shape,
            dtype=waveform.dtype,
            device=waveform.device,
        )

        if position == "before":
            extended_waveform = torch.cat((silence, waveform), dim=-1)
        else:
            extended_waveform = torch.cat((waveform, silence), dim=-1)

        output_audio["waveform"] = extended_waveform
        return (output_audio,)


NODE_CLASS_MAPPINGS = {
    "ExtendEmptyAudio": ExtendEmptyAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtendEmptyAudio": "Extend Empty Audio",
}
