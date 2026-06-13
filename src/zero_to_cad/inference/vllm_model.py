"""In-process vLLM engine wrapper for CadQuery code generation.

Runs models locally via ``vllm.LLM`` (offline inference, no HTTP server). The
engine is built when a model is loaded and torn down on ``release`` so GPU
memory is reclaimed before the next model is loaded.
"""

from __future__ import annotations

import base64
import gc
import os
from io import BytesIO

from PIL import Image

from zero_to_cad.config import (
    MAX_NEW_TOKENS,
    NUM_VIEWS,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
    ModelEntry,
)


def _image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_chat_messages(
    views: list[Image.Image],
    system_prompt: str,
    user_text: str,
) -> list[dict]:
    user_content: list[dict] = [
        {"type": "image_url", "image_url": {"url": _image_to_data_uri(view)}}
        for view in views
    ]
    user_content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


class VLLMModel:
    """Loads and runs a model with an in-process vLLM engine."""

    def __init__(self, entry: ModelEntry) -> None:
        # The flashinfer sampler's JIT arch probe fails on some GPUs (e.g.
        # Blackwell sm_120); fall back to the native Torch sampler.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        from vllm import LLM

        self.entry = entry
        self.model_id = entry.id
        engine_kwargs: dict = dict(
            model=entry.id,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=VLLM_MAX_MODEL_LEN,
            limit_mm_per_prompt={"image": NUM_VIEWS},
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        )
        if entry.hf_overrides:
            engine_kwargs["hf_overrides"] = entry.hf_overrides
        try:
            self.engine = LLM(**engine_kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "memory" in msg:
                raise RuntimeError(
                    f"{exc}\n\nNot enough free GPU memory to load "
                    f"{entry.label!r}. Free VRAM (close other GPU apps) or lower "
                    f"the engine budget, e.g. VLLM_GPU_MEMORY_UTILIZATION=0.6 "
                    f"(currently {VLLM_GPU_MEMORY_UTILIZATION})."
                ) from exc
            if "cosmos3" in msg or "does not recognize this architecture" in msg:
                from vllm import __version__ as vllm_version

                raise RuntimeError(
                    f"{entry.label!r} uses NVIDIA's Cosmos3 architecture, which the "
                    f"installed vLLM ({vllm_version}) / Transformers build does not "
                    f"support. Run it with a Cosmos3-capable vLLM (e.g. the "
                    f"`vllm/vllm-omni:cosmos3` image or vLLM-from-main with the Cosmos3 "
                    f"reasoner) — see the README. Original error:\n\n{exc}"
                ) from exc
            raise

    def generate(
        self,
        views: list[Image.Image],
        max_new_tokens: int = MAX_NEW_TOKENS,
        *,
        system_prompt: str | None = None,
        user_text: str | None = None,
    ) -> str:
        from vllm import SamplingParams

        messages = _build_chat_messages(
            views,
            system_prompt or self.entry.system_prompt,
            user_text or self.entry.user_text,
        )
        sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = self.engine.chat([messages], sampling, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def release(self) -> None:
        """Drop the engine and free GPU memory."""
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass
        if hasattr(self, "engine"):
            del self.engine
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
