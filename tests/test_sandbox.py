"""Tests for CadQuery sandbox execution."""

from __future__ import annotations

import pytest

from zero_to_cad.execute.sandbox import execute_cadquery

BOX_CODE = """
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 10)
"""


def test_execute_box() -> None:
    try:
        import cadquery  # noqa: F401
    except ImportError:
        pytest.skip("cadquery not installed")

    result = execute_cadquery(BOX_CODE)
    assert result.ok, result.error
    assert result.step_path is not None
    assert result.step_path.exists()
    # STL export depends on CadQuery/OCP build; STEP is always required
    if result.stl_path is not None:
        assert result.stl_path.exists()
