"""Remote OpenAI-compatible backend for CadQuery code generation.

Talks to a separately-run vLLM server (e.g. NVIDIA's ``vllm/vllm-omni:cosmos3``
image serving ``nvidia/Cosmos3-Nano``) over its OpenAI-compatible HTTP API. Used
for models whose architecture stock in-process vLLM cannot load.
"""

from __future__ import annotations

import os

from PIL import Image

from zero_to_cad.config import MAX_NEW_TOKENS, ModelEntry
from zero_to_cad.inference.vllm_model import _build_chat_messages

DEFAULT_BASE_URL = "http://localhost:8000/v1"


class RemoteVLLMModel:
    """Runs a model on a remote OpenAI-compatible (vLLM) server."""

    def __init__(self, entry: ModelEntry) -> None:
        from openai import OpenAI

        self.entry = entry
        self.model_id = entry.id
        self.base_url = (
            entry.base_url
            or os.environ.get("VLLM_REMOTE_BASE_URL")
            or DEFAULT_BASE_URL
        )
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        )
        try:
            self.client.models.list()
        except Exception as exc:
            raise RuntimeError(
                f"Could not reach an OpenAI-compatible vLLM server at "
                f"{self.base_url!r} for {entry.label!r}. Start the server (e.g. the "
                f"`vllm/vllm-omni:cosmos3` image) and/or set VLLM_REMOTE_BASE_URL. "
                f"Original error:\n\n{exc}"
            ) from exc

    def generate(
        self,
        views: list[Image.Image],
        max_new_tokens: int = MAX_NEW_TOKENS,
        *,
        system_prompt: str | None = None,
        user_text: str | None = None,
    ) -> str:
        messages = _build_chat_messages(
            views,
            system_prompt or self.entry.system_prompt,
            user_text or self.entry.user_text,
        )
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()

    def release(self) -> None:
        """No persistent local resources to free for a remote client."""
        return
