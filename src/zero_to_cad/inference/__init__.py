"""Inference: Qwen3-VL model wrapper and prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["CadModel"]

if TYPE_CHECKING:
    from zero_to_cad.inference.model import CadModel


def __getattr__(name: str):
    if name == "CadModel":
        from zero_to_cad.inference.model import CadModel

        return CadModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
