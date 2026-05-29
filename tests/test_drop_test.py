"""Tests for the drop-test multi-instance assembly wrapper."""

from __future__ import annotations

import pytest

from zero_to_cad.execute.drop_test import DropTestParams, build_drop_test_code
from zero_to_cad.execute.sandbox import execute_cadquery

BOX_CODE = """
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 10)
""".strip()


def test_build_drop_test_code_includes_base_and_assembly() -> None:
    wrapped = build_drop_test_code(BOX_CODE, DropTestParams(count=4))
    assert BOX_CODE in wrapped
    assert "cq.Assembly" in wrapped or "_cq_dt.Assembly" in wrapped
    assert "toCompound" in wrapped


def test_build_drop_test_code_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        build_drop_test_code("", DropTestParams(count=4))
    with pytest.raises(ValueError):
        build_drop_test_code(BOX_CODE, DropTestParams(count=0))


def test_drop_test_grid_cols_defaults_to_sqrt() -> None:
    assert DropTestParams(count=1).resolved_cols() == 1
    assert DropTestParams(count=4).resolved_cols() == 2
    assert DropTestParams(count=9).resolved_cols() == 3
    assert DropTestParams(count=8).resolved_cols() == 3
    assert DropTestParams(count=8, grid_cols=4).resolved_cols() == 4


def test_drop_test_executes_and_grows_with_count() -> None:
    try:
        import cadquery  # noqa: F401
    except ImportError:
        pytest.skip("cadquery not installed")

    single = execute_cadquery(BOX_CODE)
    assert single.ok, single.error
    assert single.step_path is not None and single.step_path.exists()
    single_size = single.step_path.stat().st_size

    dropped_code = build_drop_test_code(BOX_CODE, DropTestParams(count=4, spacing=15.0))
    dropped = execute_cadquery(dropped_code)
    assert dropped.ok, dropped.error
    assert dropped.step_path is not None and dropped.step_path.exists()

    assert dropped.step_path.stat().st_size > single_size
    if dropped.stl_path is not None:
        assert dropped.stl_path.exists()
        assert dropped.stl_path.stat().st_size > 0
