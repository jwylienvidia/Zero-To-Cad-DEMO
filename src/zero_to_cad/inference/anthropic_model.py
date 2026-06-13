"""Anthropic API model wrapper for CadQuery code generation."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from zero_to_cad.config import MAX_NEW_TOKENS, ModelEntry


def _image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ClaudeModel:
    """Runs a Claude model through the Anthropic Messages API."""

    def __init__(self, entry: ModelEntry) -> None:
        import anthropic

        self.entry = entry
        self.model_id = entry.id
        self.client = anthropic.Anthropic()

    def generate(
        self,
        views: list[Image.Image],
        max_new_tokens: int = MAX_NEW_TOKENS,
        *,
        system_prompt: str | None = None,
        user_text: str | None = None,
    ) -> str:
        content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _image_to_base64(view),
                },
            }
            for view in views
        ]
        content.append({"type": "text", "text": user_text or self.entry.user_text})

        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_new_tokens,
            system=system_prompt or self.entry.system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "\n".join(parts).strip()

    def release(self) -> None:
        """No persistent resources to free for an API client."""
        return
