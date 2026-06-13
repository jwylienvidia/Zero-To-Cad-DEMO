"""Chat prompts for Zero-To-CAD inference."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
_OPEN_ANSWER_RE = re.compile(r"<answer>\s*(.*)", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_UNCLOSED_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*)\Z", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_CRITIQUE_RE = re.compile(r"<critique>\s*(.*?)\s*</critique>", re.DOTALL | re.IGNORECASE)
_SUGGESTIONS_RE = re.compile(r"<suggestions>\s*(.*?)\s*</suggestions>", re.DOTALL | re.IGNORECASE)
_SCORE_REASONING_RE = re.compile(
    r"<score_reasoning>\s*(.*?)\s*</score_reasoning>", re.DOTALL | re.IGNORECASE
)
_CHANGE_REASONING_RE = re.compile(
    r"<change_reasoning>\s*(.*?)\s*</change_reasoning>", re.DOTALL | re.IGNORECASE
)
_SCORE_RE = re.compile(r"Score:\s*(\d{1,3})\s*%", re.IGNORECASE)
_CADQUERY_CODE_MARKERS = (
    "import cadquery",
    "import cq",
    "cadquery as cq",
    "from cadquery",
    "result =",
    "cq.Workplane",
)
_REFINE_PROSE_MARKERS = (
    "<score_reasoning>",
    "<critique>",
    "<change_reasoning>",
    "<suggestions>",
    "<answer>",
)


def looks_like_cadquery_code(text: str) -> bool:
    """Return True when text plausibly contains executable CadQuery code."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    lowered = stripped.lower()
    if any(marker in lowered for marker in _REFINE_PROSE_MARKERS):
        return False
    return any(marker in stripped for marker in _CADQUERY_CODE_MARKERS)


def _refine_answer_content(text: str) -> str:
    """Return the content inside <answer>, including unclosed tags when truncated."""
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    match = _OPEN_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_python_fence(content: str) -> tuple[str, bool]:
    """Return fenced Python and whether the opening fence was not closed."""
    fences = list(_FENCE_RE.finditer(content))
    if fences:
        return fences[-1].group(1).strip(), False
    match = _UNCLOSED_FENCE_RE.search(content)
    if match:
        return match.group(1).strip(), True
    return "", False


def refine_code_is_truncated(text: str, code: str) -> bool:
    """Return True when refine output opened a python fence but did not close it."""
    if not code:
        return False
    content = _refine_answer_content(text)
    if _FENCE_RE.search(content):
        return False
    return bool(_UNCLOSED_FENCE_RE.search(content))


def extract_cadquery_code(text: str, *, prefer_last: bool = False) -> str:
    """Pull executable CadQuery code out of a model response.

    Handles reasoning-style ``<answer>...</answer>`` blocks and Markdown code
    fences. Falls back to the original text (stripped) when neither is present,
    so plain code passes through unchanged.

    When ``prefer_last`` is True, uses the last fenced block (refine responses
    often quote the broken script before the corrected one).
    """
    if not text:
        return ""

    answer_match = _ANSWER_RE.search(text)
    if answer_match:
        text = answer_match.group(1)

    fences = list(_FENCE_RE.finditer(text))
    if fences:
        match = fences[-1] if prefer_last else fences[0]
        return match.group(1).strip()

    return text.strip()


def extract_refine_code(text: str, *, mode: str = "visual") -> str:
    """Extract CadQuery code from a refine model response.

    Only accepts content inside ``<answer>`` (or an unclosed ``<answer>`` when
    truncated) within a python code fence. Never treats critique prose as code.
    """
    if not text:
        return ""

    working = _refine_answer_content(text)
    code, _ = _extract_python_fence(working)
    if not code or not looks_like_cadquery_code(code):
        return ""

    if mode != "fix":
        score_match = _SCORE_RE.search(text)
        if score_match and int(score_match.group(1)) == 100:
            return ""
    return code


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

    open_answer = re.search(r"<answer>", text, re.IGNORECASE)
    if open_answer:
        return text[: open_answer.start()].strip()

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


REFINE_SYSTEM_PROMPT = (
    "You are a CAD quality critic. You compare a target 3D object (reference views) "
    "against a generated CadQuery model (rendered views) and its script.\n\n"
    "You MUST respond with ALL of the following sections. Never return only a "
    "percentage score.\n\n"
    "Optional (for chain-of-thought models):\n"
    "<think>\n"
    "step-by-step comparison notes\n"
    "</think>\n\n"
    "Required:\n"
    "<score_reasoning>\n"
    "Detailed reasoning for the accuracy score — what matches the target, what "
    "diverges, and why you chose this score\n"
    "</score_reasoning>\n\n"
    "Score: NN%\n\n"
    "Optional summary:\n"
    "<critique>\n"
    "brief overall quality summary\n"
    "</critique>\n\n"
    "If the score is less than 100%, you MUST also include:\n"
    "<change_reasoning>\n"
    "Reasoning behind each proposed fix — explain what to change and why it would "
    "improve accuracy\n"
    "</change_reasoning>\n\n"
    "<suggestions>\n"
    "specific, actionable changes derived from the reasoning above\n"
    "</suggestions>\n\n"
    "<answer>\n"
    "```python\n"
    "complete improved CadQuery script that defines a variable `result`\n"
    "```\n"
    "</answer>\n\n"
    "When the score is below 100%, the script in <answer> MUST address the "
    "identified issues and MUST NOT be identical to the input script.\n"
    "If the score is 100%, omit <change_reasoning>, <suggestions>, and <answer>."
)


REFINE_USER_TEMPLATE = """\
You are given {num_target} reference views of the target 3D object, followed by \
{num_render} rendered views of the current generated model produced by the script below.

Current CadQuery script:
```python
{cadquery_code}
```

Compare the generated model renders against the target views. You MUST provide:
1. <score_reasoning> explaining why you assign the score
2. Score: NN% on its own line
3. If the score is not 100%: <change_reasoning> for why each change is needed, \
<suggestions> with concrete fixes, and a complete updated CadQuery script in <answer> \
that differs from the input script."""


REFINE_FIX_SYSTEM_BASE = (
    "You are a CadQuery debugging assistant. A script failed to execute due to a "
    "syntax or runtime error. Fix the script so it runs successfully and defines a "
    "variable named `result`.\n\n"
    "Write your diagnosis as prose BEFORE the code block. Then provide the full "
    "corrected script in a ```python code block containing ONLY executable Python — "
    "no markdown headings or commentary inside the fence.\n\n"
    "The corrected script MUST fix the reported error and MUST NOT be identical "
    "to the broken input script."
)

REFINE_FIX_SYSTEM_PROMPT = build_doc_augmented_system_prompt(REFINE_FIX_SYSTEM_BASE)


REFINE_FIX_USER_TEMPLATE = """\
The following CadQuery script failed to execute:

```python
{cadquery_code}
```

Error:
{error}

Diagnose the error in prose, then return the full corrected CadQuery script in a \
```python code block with ONLY executable code (no headings inside the fence). \
The script must run successfully, define a variable `result`, and differ from \
the broken input script."""


def format_execution_error(error: str | None, traceback: str | None = None) -> str:
    """Combine an execution error message and traceback for refine prompts."""
    msg = (error or "Execution failed").strip()
    if traceback and traceback.strip():
        return f"{msg}\n\n{traceback.strip()}"
    return msg


def build_refine_fix_user_text(cadquery_code: str, error: str) -> str:
    """Build the user prompt for refining a script that failed to execute."""
    return REFINE_FIX_USER_TEMPLATE.format(
        cadquery_code=cadquery_code.strip(),
        error=error.strip(),
    )


def build_refine_user_text(
    cadquery_code: str,
    *,
    num_target: int,
    num_render: int,
) -> str:
    """Build the user prompt for the Refine self-critique workflow."""
    return REFINE_USER_TEMPLATE.format(
        num_target=num_target,
        num_render=num_render,
        cadquery_code=cadquery_code.strip(),
    )


def _subsample_evenly(items: list, count: int) -> list:
    if count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def prepare_refine_images(
    target_views: list[Image.Image],
    render_views: list[Image.Image],
    *,
    max_images: int | None = None,
) -> tuple[list[Image.Image], list[Image.Image]]:
    """Select target and render images that fit the per-prompt image limit."""
    from zero_to_cad.config import NUM_VIEWS

    limit = max_images if max_images is not None else NUM_VIEWS
    max_render = limit // 2
    renders = list(render_views[:max_render])
    num_target = limit - len(renders)
    targets = _subsample_evenly(target_views, num_target)
    return targets, renders


@dataclass
class RefineResult:
    """Parsed output from a Refine model response."""

    score_reasoning: str
    critique: str
    score: int | None
    change_reasoning: str
    suggestions: str
    code: str


def refine_code_unchanged(before: str, after: str) -> bool:
    """Return True when refine returned a non-empty script identical to the input."""
    before_stripped = before.strip()
    after_stripped = after.strip()
    return bool(before_stripped) and before_stripped == after_stripped


def parse_refine_output(text: str, *, mode: str = "visual") -> RefineResult:
    """Parse refine critique output: score reasoning, score, changes, and optional code."""
    if not text:
        return RefineResult(
            score_reasoning="",
            critique="",
            score=None,
            change_reasoning="",
            suggestions="",
            code="",
        )

    score_reasoning_match = _SCORE_REASONING_RE.search(text)
    score_reasoning = (
        score_reasoning_match.group(1).strip() if score_reasoning_match else ""
    )

    critique_match = _CRITIQUE_RE.search(text)
    critique = critique_match.group(1).strip() if critique_match else ""

    score_match = _SCORE_RE.search(text)
    score = int(score_match.group(1)) if score_match else None
    if score is not None:
        score = min(100, max(0, score))

    change_reasoning_match = _CHANGE_REASONING_RE.search(text)
    change_reasoning = (
        change_reasoning_match.group(1).strip() if change_reasoning_match else ""
    )

    suggestions_match = _SUGGESTIONS_RE.search(text)
    suggestions = suggestions_match.group(1).strip() if suggestions_match else ""

    code = extract_refine_code(text, mode=mode)

    return RefineResult(
        score_reasoning=score_reasoning,
        critique=critique,
        score=score,
        change_reasoning=change_reasoning,
        suggestions=suggestions,
        code=code,
    )


def format_refine_display(result: RefineResult) -> str:
    """Format refine critique output for display in the Reasoning tab."""
    parts: list[str] = []
    if result.score_reasoning:
        parts.append(f"Score reasoning:\n{result.score_reasoning}")
    if result.score is not None:
        parts.append(f"Score: {result.score}%")
    if result.critique:
        parts.append(f"Summary:\n{result.critique}")
    if result.change_reasoning:
        parts.append(f"Change reasoning:\n{result.change_reasoning}")
    if result.suggestions:
        parts.append(f"Suggestions:\n{result.suggestions}")
    return "\n\n".join(parts).strip()
