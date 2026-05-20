"""Tests for offscreen mesh rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from zero_to_cad.ui.viewer_render import CameraState, render_mesh_to_array


def test_render_mesh_with_and_without_edges() -> None:
    try:
        import cadquery as cq
        import pyvista as pv
    except ImportError:
        pytest.skip("cadquery/pyvista not installed")

    result = cq.Workplane("XY").box(5, 5, 5)
    with tempfile.TemporaryDirectory() as tmp:
        stl = Path(tmp) / "b.stl"
        cq.exporters.export(result, str(stl), exportType="STL")
        mesh = pv.read(stl)

    img = render_mesh_to_array(mesh, show_edges=False, camera=CameraState())
    assert img.ndim == 3
    assert img.shape[2] in (3, 4)

    img_edges = render_mesh_to_array(mesh, show_edges=True, camera=CameraState())
    assert img_edges.shape == img.shape
