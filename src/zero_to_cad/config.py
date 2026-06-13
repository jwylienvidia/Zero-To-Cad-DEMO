"""Application configuration and cache paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from zero_to_cad.inference.prompts import (
    CADQUERY_DOCS_SYSTEM_PROMPT,
    COSMOS_DOCS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    USER_TEXT,
)

DATASET_REPO = "ADSKAILab/Zero-To-CAD-1m"
DATASET_TEST_GLOB = "data/test/*.parquet"
MAX_NEW_TOKENS = 4096
SANDBOX_TIMEOUT_SEC = 60
NUM_VIEWS = 8

# 3D viewer backend: "image" (offscreen screenshot, reliable) or "vtk" (embedded VTK).
VIEWER_BACKEND = os.environ.get("ZERO_TO_CAD_VIEWER", "image").strip().lower()

# In-process vLLM engine tuning (env-overridable).
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))

_COSMOS_USER_TEXT = (
    "From these 8 views, write a complete CadQuery Python script "
    "that defines a variable `result`."
)


@dataclass(frozen=True)
class ModelEntry:
    """A selectable model with per-model prompt overrides and a backend."""

    id: str
    label: str
    system_prompt: str
    user_text: str
    backend: str = "vllm"
    gated: bool = False
    notes: str = ""
    # Passed to vLLM's ``hf_overrides`` (e.g. to select a checkpoint's reasoner
    # architecture). Ignored by non-vLLM backends.
    hf_overrides: dict | None = None
    # Base URL for the "openai" remote backend (OpenAI-compatible vLLM server).
    base_url: str | None = None


MODELS: list[ModelEntry] = [
    ModelEntry(
        id="ADSKAILab/Zero-To-CAD-Qwen3-VL-2B",
        label="Zero-To-CAD 2B (CadQuery fine-tune)",
        system_prompt=SYSTEM_PROMPT,
        user_text=USER_TEXT,
        backend="vllm",
    ),
    ModelEntry(
        id="nvidia/Cosmos-Reason2-8B",
        label="Cosmos-Reason2 8B + CadQuery docs (baseline, gated)",
        system_prompt=COSMOS_DOCS_SYSTEM_PROMPT,
        user_text=_COSMOS_USER_TEXT,
        backend="vllm",
        gated=True,
        notes=(
            "Baseline Cosmos-Reason2 with condensed CadQuery docs in the prompt. "
            "Needs `huggingface-cli login` + model gate; requires ~32 GB GPU memory."
        ),
    ),
    ModelEntry(
        id=os.environ.get(
            "COSMOS3_MODEL",
            "/home/jwylie/Dev/Cosmos2Cad/outputs/eval_model/iter_000091458",
        ),
        label="Cosmos3 8B (Zero-To-CAD reasoning fine-tune)",
        system_prompt=COSMOS_DOCS_SYSTEM_PROMPT,
        user_text=_COSMOS_USER_TEXT,
        backend="vllm",
        notes=(
            "Local Cosmos3 8B reasoning fine-tune. Override the weights path with "
            "the COSMOS3_MODEL env var."
        ),
    ),
    ModelEntry(
        id=os.environ.get("COSMOS3_NANO_MODEL", "nvidia/Cosmos3-Nano"),
        label="Cosmos3-Nano + CadQuery docs (baseline, gated)",
        system_prompt=COSMOS_DOCS_SYSTEM_PROMPT,
        user_text=_COSMOS_USER_TEXT,
        backend="openai",
        gated=True,
        base_url=os.environ.get("COSMOS3_NANO_BASE_URL", "http://localhost:8000/v1"),
        notes=(
            "Cosmos3-Nano baseline served by a separate Cosmos3-capable vLLM "
            "server (e.g. the `vllm/vllm-omni:cosmos3` image). Set "
            "COSMOS3_NANO_BASE_URL to point at it; stock in-process vLLM cannot "
            "load the cosmos3_omni architecture."
        ),
    ),
    ModelEntry(
        id=os.environ.get("CLAUDE_FABLE_MODEL", "claude-fable-5"),
        label="Claude Fable 5 (API)",
        system_prompt=CADQUERY_DOCS_SYSTEM_PROMPT,
        user_text=USER_TEXT,
        backend="anthropic",
        notes="Anthropic API model. Set ANTHROPIC_API_KEY; override id with CLAUDE_FABLE_MODEL.",
    ),
]

MODEL_ID = MODELS[0].id


def get_model_entry(model_id: str) -> ModelEntry | None:
    """Return the registry entry for ``model_id``, or ``None`` if unknown."""
    for entry in MODELS:
        if entry.id == model_id:
            return entry
    return None


def cache_root() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "zero-to-cad"
    return Path.home() / ".cache" / "zero-to-cad"


def dataset_test_dir() -> Path:
    return cache_root() / "dataset" / "test"


def exports_dir() -> Path:
    return cache_root() / "exports"


def assets_dir() -> Path:
    return cache_root() / "assets"
