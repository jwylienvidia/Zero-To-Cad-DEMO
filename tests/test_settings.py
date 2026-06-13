"""Tests for persistent settings (settings.json)."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


def _reload_settings(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ZERO_TO_CAD_SETTINGS", str(tmp_path / "settings.json"))
    import zero_to_cad.settings as settings

    return importlib.reload(settings)


def test_save_and_load_roundtrip(tmp_path, monkeypatch) -> None:
    settings = _reload_settings(tmp_path, monkeypatch)
    path = settings.save_settings(
        {"ANTHROPIC_API_KEY": "sk-test", "GEMINI_API_KEY": "  ", "COSMOS3_NANO_BASE_URL": "http://x/v1"}
    )
    assert path.exists()
    loaded = settings.load_settings()
    # Empty/whitespace values are dropped.
    assert loaded == {"ANTHROPIC_API_KEY": "sk-test", "COSMOS3_NANO_BASE_URL": "http://x/v1"}


def test_load_missing_returns_empty(tmp_path, monkeypatch) -> None:
    settings = _reload_settings(tmp_path, monkeypatch)
    assert settings.load_settings() == {}


def test_apply_settings_does_not_override_env(tmp_path, monkeypatch) -> None:
    settings = _reload_settings(tmp_path, monkeypatch)
    settings.save_settings({"ANTHROPIC_API_KEY": "from-file", "GEMINI_API_KEY": "g-file"})

    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings.apply_settings_to_env()

    assert os.environ["ANTHROPIC_API_KEY"] == "from-env"
    assert os.environ["GEMINI_API_KEY"] == "g-file"
