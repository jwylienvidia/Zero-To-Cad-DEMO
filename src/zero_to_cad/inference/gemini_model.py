"""Google Gemini API model wrapper for CadQuery code generation."""

from __future__ import annotations

import os

from PIL import Image

from zero_to_cad.config import MAX_NEW_TOKENS, ModelEntry


class GeminiModel:
    """Runs a Gemini model through the Google GenAI API."""

    def __init__(self, entry: ModelEntry) -> None:
        from google import genai

        self.entry = entry
        self.model_id = entry.id
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini requires an API key. Set it in Settings (GEMINI_API_KEY) "
                "or the GEMINI_API_KEY / GOOGLE_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        views: list[Image.Image],
        max_new_tokens: int = MAX_NEW_TOKENS,
        *,
        system_prompt: str | None = None,
        user_text: str | None = None,
    ) -> str:
        from google.genai import types

        contents: list = [view.convert("RGB") for view in views]
        contents.append(user_text or self.entry.user_text)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt or self.entry.system_prompt,
            max_output_tokens=max_new_tokens,
            temperature=0.0,
        )
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config,
        )
        return (response.text or "").strip()

    def release(self) -> None:
        """No persistent resources to free for an API client."""
        return
