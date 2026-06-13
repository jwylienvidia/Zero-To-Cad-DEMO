"""Tests for the hardcoded model registry."""

from __future__ import annotations

from zero_to_cad.config import MODELS, MODEL_ID, get_model_entry


def test_models_non_empty() -> None:
    assert len(MODELS) >= 2
    for entry in MODELS:
        assert entry.id
        assert entry.label
        assert entry.system_prompt
        assert entry.user_text
        assert entry.backend in {"vllm", "openai", "anthropic", "gemini"}
        assert isinstance(entry.is_reasoning, bool)
        if entry.backend == "openai":
            assert entry.base_url


def test_reasoning_models_present() -> None:
    reasoning = [e for e in MODELS if e.is_reasoning]
    assert len(reasoning) >= 3
    assert any("Cosmos-Reason2" in e.label for e in reasoning)
    assert any("Cosmos3 8B" in e.label for e in reasoning)
    assert any("Cosmos3-Nano" in e.label for e in reasoning)


def test_registry_contains_expected_backends() -> None:
    backends = {e.backend for e in MODELS}
    assert "vllm" in backends
    assert "anthropic" in backends
    assert "openai" in backends
    assert "gemini" in backends
    labels = [e.label for e in MODELS]
    assert any("Cosmos3 8B" in label for label in labels)
    assert any("Cosmos3-Nano" in label for label in labels)
    assert any("Claude" in label for label in labels)
    assert any("Gemini" in label for label in labels)


def test_model_labels_unique() -> None:
    labels = [e.label for e in MODELS]
    assert len(labels) == len(set(labels))


def test_default_model_id_in_registry() -> None:
    assert get_model_entry(MODEL_ID) is not None
    assert MODEL_ID == MODELS[0].id


def test_get_model_entry_unknown() -> None:
    assert get_model_entry("does/not/exist") is None
