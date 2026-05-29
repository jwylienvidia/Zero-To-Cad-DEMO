"""Tests for the local asset-folder exporter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from zero_to_cad.export.asset import save_asset


def _make_box_stl_step(tmp: Path) -> tuple[Path, Path]:
    import cadquery as cq

    result = cq.Workplane("XY").box(8, 6, 4)
    step = tmp / "box.step"
    stl = tmp / "box.stl"
    cq.exporters.export(result, str(step), exportType="STEP")
    cq.exporters.export(result, str(stl), exportType="STL")
    return step, stl


def test_save_asset_writes_full_layout(tmp_path: Path) -> None:
    try:
        import cadquery  # noqa: F401
        import pyvista  # noqa: F401
    except ImportError:
        pytest.skip("cadquery/pyvista not installed")

    with tempfile.TemporaryDirectory() as work:
        step, stl = _make_box_stl_step(Path(work))
        views = [Image.new("RGB", (32, 32), color=(i * 30, 50, 200)) for i in range(8)]
        code = "import cadquery as cq\nresult = cq.Workplane('XY').box(8, 6, 4)\n"

        paths = save_asset(
            tmp_path / "asset_box",
            name="box",
            code=code,
            step_path=step,
            stl_path=stl,
            views=views,
        )

    assert paths.asset_dir.is_dir()
    assert paths.manifest_path.exists()

    manifest = json.loads(paths.manifest_path.read_text())
    assert manifest["name"] == "box"
    comps = manifest["components"]
    assert comps["step"] == "box.step"
    assert comps["stl"] == "box.stl"
    assert comps["obj"] == "box.obj"
    assert comps["mtl"] == "box.mtl"
    assert comps["albedo"] == "textures/albedo.png"
    assert len(comps["views"]) == 8

    assert (paths.asset_dir / "box.step").exists()
    assert (paths.asset_dir / "box.stl").exists()
    assert (paths.asset_dir / "box.obj").exists()
    assert (paths.asset_dir / "box.mtl").exists()
    assert (paths.asset_dir / "textures" / "albedo.png").exists()
    assert (paths.asset_dir / "code.py").exists()
    for i in range(8):
        assert (paths.asset_dir / "views" / f"view_{i}.png").exists()
        assert (paths.asset_dir / "textures" / f"view_{i}.png").exists()

    obj_text = (paths.asset_dir / "box.obj").read_text()
    assert "mtllib box.mtl" in obj_text
    assert "usemtl ZeroToCadMat" in obj_text
    assert obj_text.count("\nv ") >= 8  # cube has at least 8 unique vertices

    mtl_text = (paths.asset_dir / "box.mtl").read_text()
    assert "newmtl ZeroToCadMat" in mtl_text
    assert "map_Kd textures/albedo.png" in mtl_text


def test_save_asset_minimal_without_stl(tmp_path: Path) -> None:
    paths = save_asset(
        tmp_path / "asset_min",
        name="empty",
        code="result = None  # placeholder",
        step_path=None,
        stl_path=None,
        views=None,
    )

    assert paths.manifest_path.exists()
    manifest = json.loads(paths.manifest_path.read_text())
    assert manifest["components"]["stl"] is None
    assert manifest["components"]["obj"] is None
    assert manifest["components"]["albedo"] is None
    assert manifest["components"]["views"] == []
    assert (paths.asset_dir / "code.py").exists()
