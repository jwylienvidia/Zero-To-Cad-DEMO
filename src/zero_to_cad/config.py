"""Application configuration and cache paths."""

from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "ADSKAILab/Zero-To-CAD-Qwen3-VL-2B"
DATASET_REPO = "ADSKAILab/Zero-To-CAD-1m"
DATASET_TEST_GLOB = "data/test/*.parquet"
MAX_NEW_TOKENS = 4096
SANDBOX_TIMEOUT_SEC = 60
NUM_VIEWS = 8

# 3D viewer backend: "image" (offscreen screenshot, reliable) or "vtk" (embedded VTK).
VIEWER_BACKEND = os.environ.get("ZERO_TO_CAD_VIEWER", "image").strip().lower()


def cache_root() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "zero-to-cad"
    return Path.home() / ".cache" / "zero-to-cad"


def dataset_test_dir() -> Path:
    return cache_root() / "dataset" / "test"


def exports_dir() -> Path:
    return cache_root() / "exports"
