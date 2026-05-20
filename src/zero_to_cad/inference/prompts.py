"""Chat prompts for Zero-To-CAD inference."""

from __future__ import annotations

from PIL import Image

SYSTEM_PROMPT = (
    "You are a CAD code assistant. Given multiple rendered views of a 3D shape, "
    "generate clean, well-structured CadQuery Python code that accurately "
    "reproduces the geometry."
)

USER_TEXT = "Generate CadQuery code for this shape."


def build_messages(views: list[Image.Image]) -> list[dict]:
    """Build the chat message list expected by the model processor."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": view} for view in views],
                {"type": "text", "text": USER_TEXT},
            ],
        },
    ]
