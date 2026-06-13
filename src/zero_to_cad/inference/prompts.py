"""Chat prompts for Zero-To-CAD inference."""

from __future__ import annotations

import re

from PIL import Image

from zero_to_cad.inference.cadquery_reference import CADQUERY_REFERENCE

SYSTEM_PROMPT = (
    "You are a CAD code assistant. Given multiple rendered views of a 3D shape, "
    "generate clean, well-structured CadQuery Python code that accurately "
    "reproduces the geometry."
)

USER_TEXT = "Generate CadQuery code for this shape."

COSMOS_REASON_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the question in the following format: "
    "<think>\nyour reasoning\n</think>\n\n"
    "<answer>\nyour answer\n</answer>."
)


def build_doc_augmented_system_prompt(base_system_prompt: str) -> str:
    """Append the condensed CadQuery reference to a base system prompt."""
    return (
        f"{base_system_prompt}\n\n"
        "Use the following condensed CadQuery reference to write correct code:\n\n"
        f"{CADQUERY_REFERENCE}"
    )


COSMOS_DOCS_SYSTEM_PROMPT = build_doc_augmented_system_prompt(COSMOS_REASON_SYSTEM_PROMPT)

# Docs-augmented prompt without the <think>/<answer> wrapper (used for the Claude API model).
CADQUERY_DOCS_SYSTEM_PROMPT = build_doc_augmented_system_prompt(SYSTEM_PROMPT)


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def extract_cadquery_code(text: str) -> str:
    """Pull executable CadQuery code out of a model response.

    Handles reasoning-style ``<answer>...</answer>`` blocks and Markdown code
    fences. Falls back to the original text (stripped) when neither is present,
    so plain code passes through unchanged.
    """
    if not text:
        return ""

    answer_match = _ANSWER_RE.search(text)
    if answer_match:
        text = answer_match.group(1)

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    return text.strip()


def extract_reasoning(text: str) -> str:
    """Return the model's reasoning trace, if any.

    Prefers an explicit ``<think>...</think>`` block; otherwise returns any text
    that precedes an ``<answer>`` block. Returns an empty string when the output
    is plain code (no reasoning to show).
    """
    if not text:
        return ""

    think_match = _THINK_RE.search(text)
    if think_match:
        return think_match.group(1).strip()

    answer_match = _ANSWER_RE.search(text)
    if answer_match:
        return text[: answer_match.start()].strip()

    return ""

REASONING_TEST_USER_TEMPLATE = """\
You are given 8 rendered views of a CAD object and its ground-truth CadQuery script.

Ground-truth CadQuery script:
```python
{cadquery_code}
```

Generate concise reasoning that explains how the visible geometry maps to the
operations and parameters in the ground-truth script. Do not rewrite the script.
Focus on observations that would help train another model to produce this code
from the images. This is meant for training a model that will generate python code
from images. Provide the reasoning of the steps and order of operations involved in generating 
the parametric model. From the training perspective, reasoning is the input and the script 
is the output, so the reasoning should be a description of the steps and order of operations 
involved and CAN NEVER know, infer or mention anything about the code or script. """


def build_messages(
    views: list[Image.Image],
    *,
    system_prompt: str = SYSTEM_PROMPT,
    user_text: str = USER_TEXT,
) -> list[dict]:
    """Build the chat message list expected by the model processor."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": view} for view in views],
                {"type": "text", "text": user_text},
            ],
        },
    ]


def build_reasoning_test_user_text(cadquery_code: str) -> str:
    """Build a Cosmos-Reason prompt using a row's ground-truth CadQuery code."""
    return REASONING_TEST_USER_TEMPLATE.format(cadquery_code=cadquery_code.strip())
