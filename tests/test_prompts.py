"""Tests for inference prompts."""

from __future__ import annotations

from PIL import Image

from zero_to_cad.inference.cadquery_reference import CADQUERY_REFERENCE
from zero_to_cad.inference.prompts import (
    CADQUERY_DOCS_SYSTEM_PROMPT,
    COSMOS_DOCS_SYSTEM_PROMPT,
    COSMOS_REASON_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    USER_TEXT,
    build_doc_augmented_system_prompt,
    build_messages,
    build_reasoning_test_user_text,
    build_refine_fix_user_text,
    build_refine_user_text,
    extract_cadquery_code,
    extract_reasoning,
    extract_refine_code,
    format_execution_error,
    format_refine_display,
    looks_like_cadquery_code,
    parse_refine_output,
    prepare_refine_images,
    refine_code_is_truncated,
    refine_code_unchanged,
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


def test_cadquery_docs_system_prompt_includes_reference() -> None:
    assert CADQUERY_DOCS_SYSTEM_PROMPT.startswith(SYSTEM_PROMPT)
    assert CADQUERY_REFERENCE in CADQUERY_DOCS_SYSTEM_PROMPT
    assert COSMOS_REASON_SYSTEM_PROMPT not in CADQUERY_DOCS_SYSTEM_PROMPT


def test_extract_cadquery_code_plain_passthrough() -> None:
    code = 'import cadquery as cq\nresult = cq.Workplane("XY").box(1, 2, 3)'
    assert extract_cadquery_code(code) == code


def test_extract_cadquery_code_strips_fences() -> None:
    text = "Here is the script:\n```python\nimport cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)\n```\n"
    extracted = extract_cadquery_code(text)
    assert extracted == "import cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)"


def test_extract_cadquery_code_from_answer_block() -> None:
    text = (
        "<think>\nFirst make a base box.\n</think>\n"
        "<answer>\n```python\nimport cadquery as cq\nresult = cq.Workplane().box(2, 2, 2)\n```\n</answer>"
    )
    extracted = extract_cadquery_code(text)
    assert extracted == "import cadquery as cq\nresult = cq.Workplane().box(2, 2, 2)"
    assert "<think>" not in extracted


def test_extract_cadquery_code_empty() -> None:
    assert extract_cadquery_code("") == ""


def test_extract_reasoning_from_think_block() -> None:
    text = (
        "<think>\nIdentify the base box, then cut a hole.\n</think>\n"
        "<answer>\n```python\nresult = 1\n```\n</answer>"
    )
    reasoning = extract_reasoning(text)
    assert reasoning == "Identify the base box, then cut a hole."
    assert "result = 1" not in reasoning


def test_extract_reasoning_before_answer() -> None:
    text = "First build the base.\n<answer>\nresult = 1\n</answer>"
    assert extract_reasoning(text) == "First build the base."


def test_extract_reasoning_plain_code_is_empty() -> None:
    assert extract_reasoning("import cadquery as cq\nresult = cq.Workplane()") == ""
    assert extract_reasoning("") == ""


def test_extract_cadquery_code_prefers_last_fence() -> None:
    text = (
        "Broken script:\n```python\nresult = broken(\n```\n"
        "Fixed script:\n```python\nimport cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)\n```"
    )
    assert "broken(" in extract_cadquery_code(text)
    assert extract_cadquery_code(text, prefer_last=True) == (
        "import cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)"
    )


def test_extract_refine_code_rejects_diagnosis_prose() -> None:
    text = (
        "### Diagnosis\n\n"
        "The script fails with a `StdFail_NotDone` error during `.chamfer()`.\n"
        "The selector `.faces(\">Z\")` selects the entire top face."
    )
    assert extract_refine_code(text, mode="fix") == ""
    assert looks_like_cadquery_code(text) is False


def test_extract_refine_code_uses_answer_fence_not_full_response() -> None:
    text = (
        "<score_reasoning>\nThe hex shape is correct.\n</score_reasoning>\n"
        "Score: 25%\n"
        "<answer>\n"
        "```python\n"
        "import cadquery as cq\n"
        "result = cq.Workplane('XY').box(1, 1, 1)\n"
        ".rect(bottom"
    )
    code = extract_refine_code(text, mode="visual")
    assert code.startswith("import cadquery as cq")
    assert "<score_reasoning>" not in code
    assert "Score: 25%" not in code
    assert code.endswith(".rect(bottom")
    assert refine_code_is_truncated(text, code) is True


def test_refine_fix_system_prompt_includes_cadquery_reference() -> None:
    from zero_to_cad.inference.prompts import REFINE_FIX_SYSTEM_PROMPT

    assert CADQUERY_REFERENCE in REFINE_FIX_SYSTEM_PROMPT
    assert "condensed CadQuery reference" in REFINE_FIX_SYSTEM_PROMPT


def test_extract_refine_code_fix_mode_ignores_perfect_score() -> None:
    text = (
        "Score: 100%\n"
        "```python\nimport cadquery as cq\nresult = cq.Workplane().box(2, 2, 2)\n```"
    )
    assert extract_refine_code(text, mode="visual") == ""
    assert extract_refine_code(text, mode="fix") == (
        "import cadquery as cq\nresult = cq.Workplane().box(2, 2, 2)"
    )


def test_parse_refine_output_fix_mode_with_score() -> None:
    text = (
        "Diagnosis: missing parenthesis.\n"
        "Score: 100%\n"
        "```python\nimport cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)\n```"
    )
    visual = parse_refine_output(text, mode="visual")
    fix = parse_refine_output(text, mode="fix")
    assert visual.code == ""
    assert fix.code == "import cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)"


def test_prepare_refine_images_caps_at_num_views() -> None:
    targets = [Image.new("RGB", (8, 8)) for _ in range(8)]
    renders = [Image.new("RGB", (8, 8)) for _ in range(4)]
    sel_targets, sel_renders = prepare_refine_images(targets, renders)
    assert len(sel_targets) == 4
    assert len(sel_renders) == 4
    assert len(sel_targets) + len(sel_renders) == 8


def test_build_refine_fix_user_text_includes_code_and_error() -> None:
    prompt = build_refine_fix_user_text(
        "import cadquery as cq\nresult = cq.Workplane().box(1, 1",
        "SyntaxError: unexpected EOF while parsing",
    )
    assert "failed to execute" in prompt
    assert "result = cq.Workplane().box(1, 1" in prompt
    assert "SyntaxError: unexpected EOF while parsing" in prompt
    assert "define a variable `result`" in prompt
    assert "```python" in prompt


def test_format_execution_error() -> None:
    assert format_execution_error("boom", "Traceback...") == "boom\n\nTraceback..."
    assert format_execution_error("boom", None) == "boom"


def test_parse_refine_output_fix_response_extracts_code() -> None:
    text = (
        "<think>\nMissing closing parenthesis.\n</think>\n"
        "<answer>\n```python\nimport cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)\n```\n</answer>"
    )
    result = parse_refine_output(text, mode="fix")
    assert result.score_reasoning == ""
    assert result.critique == ""
    assert result.score is None
    assert result.code == "import cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)"


def test_build_refine_user_text_includes_counts_and_code() -> None:
    prompt = build_refine_user_text(
        'import cadquery as cq\nresult = cq.Workplane("XY").box(1, 2, 3)',
        num_target=8,
        num_render=4,
    )
    assert "8 reference views" in prompt
    assert "4 rendered views" in prompt
    assert "score_reasoning" in prompt
    assert 'result = cq.Workplane("XY").box(1, 2, 3)' in prompt


def test_parse_refine_output_full() -> None:
    text = (
        "<score_reasoning>\nBase width is too large vs target.\n</score_reasoning>\n"
        "<critique>\nThe base is too wide.\n</critique>\n"
        "Score: 80%\n"
        "<change_reasoning>\nNarrowing the box width should match the target profile.\n"
        "</change_reasoning>\n"
        "<suggestions>\nReduce the box width.\n</suggestions>\n"
        "<answer>\n```python\nimport cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)\n```\n</answer>"
    )
    result = parse_refine_output(text)
    assert result.score_reasoning == "Base width is too large vs target."
    assert result.critique == "The base is too wide."
    assert result.score == 80
    assert "Narrowing the box width" in result.change_reasoning
    assert result.suggestions == "Reduce the box width."
    assert result.code == "import cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)"


def test_parse_refine_output_perfect_score_no_code() -> None:
    text = (
        "<score_reasoning>\nAll dimensions and features match.\n</score_reasoning>\n"
        "<critique>\nPerfect match.\n</critique>\n"
        "Score: 100%\n"
    )
    result = parse_refine_output(text)
    assert result.score_reasoning == "All dimensions and features match."
    assert result.critique == "Perfect match."
    assert result.score == 100
    assert result.change_reasoning == ""
    assert result.suggestions == ""
    assert result.code == ""


def test_format_refine_display() -> None:
    from zero_to_cad.inference.prompts import RefineResult

    display = format_refine_display(
        RefineResult(
            score_reasoning="Most features align.",
            critique="Close match.",
            score=92,
            change_reasoning="Fillet radius is slightly off.",
            suggestions="Tweak fillet radius.",
            code="",
        )
    )
    assert "Most features align." in display
    assert "Score: 92%" in display
    assert "Fillet radius is slightly off." in display
    assert "Tweak fillet radius." in display


def test_refine_code_unchanged() -> None:
    code = "import cadquery as cq\nresult = cq.Workplane().box(1, 1, 1)"
    assert refine_code_unchanged(code, code) is True
    assert refine_code_unchanged(code, code + "\n") is True
    assert refine_code_unchanged(code, code.replace("1, 1, 1", "2, 2, 2")) is False
    assert refine_code_unchanged("", "") is False
