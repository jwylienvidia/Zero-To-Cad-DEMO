"""Tests for inference prompts."""

from __future__ import annotations

from PIL import Image

from zero_to_cad.inference.cadquery_reference import CADQUERY_REFERENCE
from zero_to_cad.inference.prompts import (
    COSMOS_DOCS_SYSTEM_PROMPT,
    COSMOS_REASON_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    USER_TEXT,
    build_doc_augmented_system_prompt,
    build_messages,
    build_reasoning_test_user_text,
)


def test_build_messages_structure() -> None:
    views = [Image.new("RGB", (64, 64)) for _ in range(8)]
    messages = build_messages(views)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert SYSTEM_PROMPT in messages[0]["content"]
    user_content = messages[1]["content"]
    assert len(user_content) == 9
    assert user_content[-1]["text"] == USER_TEXT
    assert sum(1 for c in user_content if c["type"] == "image") == 8


def test_build_messages_prompt_overrides() -> None:
    views = [Image.new("RGB", (64, 64)) for _ in range(8)]
    messages = build_messages(views, system_prompt="X", user_text="Y")
    assert messages[0]["content"] == "X"
    assert messages[1]["content"][-1]["text"] == "Y"


def test_build_reasoning_test_user_text_includes_ground_truth() -> None:
    prompt = build_reasoning_test_user_text(
        """
        import cadquery as cq
        result = cq.Workplane("XY").box(1, 2, 3)
        """
    )
    assert "Ground-truth CadQuery script" in prompt
    assert 'result = cq.Workplane("XY").box(1, 2, 3)' in prompt
    assert "Do not rewrite the script" in prompt
    assert "from the images" in prompt


def test_cadquery_reference_is_non_empty() -> None:
    assert CADQUERY_REFERENCE.strip()
    assert "import cadquery as cq" in CADQUERY_REFERENCE
    assert "result" in CADQUERY_REFERENCE
    assert "mirrorX" in CADQUERY_REFERENCE
    assert ".sketch()" in CADQUERY_REFERENCE
    assert "sweep" in CADQUERY_REFERENCE


def test_cosmos_docs_system_prompt_includes_reference() -> None:
    assert COSMOS_REASON_SYSTEM_PROMPT in COSMOS_DOCS_SYSTEM_PROMPT
    assert CADQUERY_REFERENCE in COSMOS_DOCS_SYSTEM_PROMPT
    assert "condensed CadQuery reference" in COSMOS_DOCS_SYSTEM_PROMPT


def test_build_doc_augmented_system_prompt() -> None:
    augmented = build_doc_augmented_system_prompt("Base prompt.")
    assert augmented.startswith("Base prompt.")
    assert CADQUERY_REFERENCE in augmented


def test_build_messages_with_docs_system_prompt() -> None:
    views = [Image.new("RGB", (64, 64)) for _ in range(8)]
    messages = build_messages(views, system_prompt=COSMOS_DOCS_SYSTEM_PROMPT)
    assert CADQUERY_REFERENCE in messages[0]["content"]
    assert COSMOS_REASON_SYSTEM_PROMPT in messages[0]["content"]
