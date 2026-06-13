"""Persistent app settings (API keys and endpoints) stored in settings.json.

The file lives at the repository root in development checkouts (and is
gitignored) so secrets never get committed. Environment variables always take
precedence over the file, so CI / shell exports keep working unchanged.

This module deliberately avoids importing ``zero_to_cad.config`` so that
``apply_settings_to_env`` can run before config-time environment reads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# (env var name, human label, is_secret)
SETTINGS_FIELDS: list[tuple[str, str, bool]] = [
    ("ANTHROPIC_API_KEY", "Anthropic API key (Claude)", True),
    ("GEMINI_API_KEY", "Google Gemini API key", True),
    ("COSMOS3_NANO_BASE_URL", "Cosmos3-Nano vLLM server URL", False),
    ("VLLM_REMOTE_BASE_URL", "Default remote vLLM server URL", False),
]
SETTINGS_KEYS: list[str] = [key for key, _, _ in SETTINGS_FIELDS]


def settings_path() -> Path:
    """Resolve the settings.json location.

    Honors ``ZERO_TO_CAD_SETTINGS`` if set; otherwise uses the repo root in
    editable/dev checkouts, falling back to the user config dir.
    """
    override = os.environ.get("ZERO_TO_CAD_SETTINGS")
    if override:
        return Path(override).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").exists():
        return repo_root / "settings.json"
    return Path.home() / ".config" / "zero-to-cad" / "settings.json"


def load_settings() -> dict[str, str]:
    """Return the saved settings, or an empty dict if none/invalid."""
    path = settings_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def save_settings(values: dict[str, str]) -> Path:
    """Persist non-empty settings to settings.json (mode 600) and return the path."""
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {k: v.strip() for k, v in values.items() if isinstance(v, str) and v.strip()}
    path.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def apply_settings_to_env() -> None:
    """Populate environment variables from settings.json.

    Uses ``setdefault`` so any value already present in the environment wins.
    """
    for key, value in load_settings().items():
        if value.strip():
            os.environ.setdefault(key, value)
