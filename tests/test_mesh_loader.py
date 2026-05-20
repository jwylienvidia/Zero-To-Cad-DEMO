"""Tests for mesh loading utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from zero_to_cad.ui.mesh_loader import load_mesh_path


def test_load_stl_from_cadquery_export() -> None:
    try:
        import cadquery as cq
    except ImportError:
        pytest.skip("cadquery not installed")

    result = cq.Workplane("XY").box(5, 5, 5)
    with tempfile.TemporaryDirectory() as tmp:
        stl = Path(tmp) / "box.stl"
        step = Path(tmp) / "box.step"
        cq.exporters.export(result, str(stl), exportType="STL")
        cq.exporters.export(result, str(step), exportType="STEP")

        mesh_stl = load_mesh_path(stl)
        assert mesh_stl.n_cells > 0

        mesh_step = load_mesh_path(step)
        assert mesh_step.n_cells > 0
