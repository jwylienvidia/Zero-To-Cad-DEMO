"""Qwen3-VL model wrapper for CadQuery code generation."""

from __future__ import annotations

from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from zero_to_cad.config import MAX_NEW_TOKENS, MODEL_ID
from zero_to_cad.inference.prompts import build_messages


class CadModel:
    """Loads and runs ADSKAILab/Zero-To-CAD-Qwen3-VL-2B."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device_map: str | dict = "auto",
    ) -> None:
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device_map,
        )
        self.model.eval()

    def generate(
        self,
        views: list[Image.Image],
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        messages = build_messages(views)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=text,
            images=views,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        decoded = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        return decoded[0].strip()
