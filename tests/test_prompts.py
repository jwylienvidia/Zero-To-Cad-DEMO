"""Tests for inference prompts."""

from __future__ import annotations

from PIL import Image

from zero_to_cad.inference.prompts import (
    SYSTEM_PROMPT,
    USER_TEXT,
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
