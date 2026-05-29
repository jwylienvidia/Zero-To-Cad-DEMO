"""Chat prompts for Zero-To-CAD inference."""

from __future__ import annotations

from PIL import Image

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
