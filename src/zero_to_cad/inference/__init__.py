"""Inference: model backends, factory, and prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from PIL import Image

__all__ = ["InferenceModel", "load_model"]

if TYPE_CHECKING:
    from zero_to_cad.config import ModelEntry


@runtime_checkable
class InferenceModel(Protocol):
    """Common interface shared by every inference backend."""

    entry: "ModelEntry"

    def generate(
        self,
        views: list[Image.Image],
        max_new_tokens: int = ...,
        *,
        system_prompt: str | None = ...,
        user_text: str | None = ...,
    ) -> str: ...

    def release(self) -> None: ...


def load_model(entry: "ModelEntry") -> InferenceModel:
    """Instantiate the backend implementation for ``entry``."""
    if entry.backend == "anthropic":
        from zero_to_cad.inference.anthropic_model import ClaudeModel

        return ClaudeModel(entry)
    if entry.backend == "gemini":
        from zero_to_cad.inference.gemini_model import GeminiModel

        return GeminiModel(entry)
    if entry.backend == "openai":
        from zero_to_cad.inference.remote_vllm_model import RemoteVLLMModel

        return RemoteVLLMModel(entry)
    if entry.backend == "vllm":
        from zero_to_cad.inference.vllm_model import VLLMModel

        return VLLMModel(entry)
    raise ValueError(f"Unknown model backend: {entry.backend!r}")
