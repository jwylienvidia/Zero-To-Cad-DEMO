"""Application configuration and cache paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from zero_to_cad.inference.prompts import (
    COSMOS_DOCS_SYSTEM_PROMPT,
    COSMOS_REASON_SYSTEM_PROMPT,
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


@dataclass(frozen=True)
class ModelEntry:
    """A selectable Hugging Face model with per-model prompt overrides."""

    id: str
    label: str
    system_prompt: str
    user_text: str
    gated: bool = False
    notes: str = ""


MODELS: list[ModelEntry] = [
    ModelEntry(
        id="ADSKAILab/Zero-To-CAD-Qwen3-VL-2B",
        label="Zero-To-CAD 2B (CadQuery fine-tune)",
        system_prompt=SYSTEM_PROMPT,
        user_text=USER_TEXT,
    ),
    ModelEntry(
        id="nvidia/Cosmos-Reason2-8B",
        label="Cosmos-Reason2 8B (baseline, gated)",
        system_prompt=COSMOS_REASON_SYSTEM_PROMPT,
        user_text=(
            "From these 8 views, write a complete CadQuery Python script "
            "that defines a variable `result`."
        ),
        gated=True,
        notes="Needs `huggingface-cli login` + model gate; requires ~32 GB GPU memory.",
    ),
    ModelEntry(
        id="nvidia/Cosmos-Reason2-8B",
        label="Cosmos-Reason2 8B + CadQuery docs (baseline, gated)",
        system_prompt=COSMOS_DOCS_SYSTEM_PROMPT,
        user_text=(
            "From these 8 views, write a complete CadQuery Python script "
            "that defines a variable `result`."
        ),
        gated=True,
        notes=(
            "Baseline Cosmos with condensed CadQuery docs in the prompt. "
            "Same gate/GPU needs as Cosmos-Reason2 8B."
        ),
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
