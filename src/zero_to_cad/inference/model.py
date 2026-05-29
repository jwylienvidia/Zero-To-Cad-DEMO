"""Qwen3-VL model wrapper for CadQuery code generation."""

from __future__ import annotations

import gc
from typing import overload

from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from zero_to_cad.config import MAX_NEW_TOKENS, MODEL_ID, ModelEntry, get_model_entry
from zero_to_cad.inference.prompts import build_messages


class CadModel:
    """Loads and runs a Qwen3-VL-compatible model for CadQuery generation."""

    @overload
    def __init__(
        self,
        entry: ModelEntry,
        *,
        device_map: str | dict = "auto",
    ) -> None: ...

    @overload
    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        device_map: str | dict = "auto",
    ) -> None: ...

    def __init__(
        self,
        entry_or_id: ModelEntry | str | None = None,
        *,
        entry: ModelEntry | None = None,
        model_id: str = MODEL_ID,
        device_map: str | dict = "auto",
    ) -> None:
        resolved = entry or entry_or_id or model_id
        if isinstance(resolved, ModelEntry):
            self.entry = resolved
        else:
            from zero_to_cad.inference.prompts import SYSTEM_PROMPT, USER_TEXT

            known = get_model_entry(resolved)
            self.entry = known or ModelEntry(
                id=resolved,
                label=resolved,
                system_prompt=SYSTEM_PROMPT,
                user_text=USER_TEXT,
            )

        self.model_id = self.entry.id
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=device_map,
        )
        self.model.eval()

    def release(self) -> None:
        """Drop model weights and free GPU memory."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(
        self,
        views: list[Image.Image],
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        messages = build_messages(
            views,
            system_prompt=self.entry.system_prompt,
            user_text=self.entry.user_text,
        )
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
