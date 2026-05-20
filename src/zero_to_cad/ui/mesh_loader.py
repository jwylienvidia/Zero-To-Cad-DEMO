"""Load CAD meshes for the VTK viewer (STL native; STEP via CadQuery)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyvista as pv

STEP_SUFFIXES = {".step", ".stp"}
MESH_SUFFIXES = {".stl", ".vtk", ".vtu", ".obj", ".ply"}


def load_mesh_path(path: str | Path) -> pv.DataSet:
    """Load a mesh file into a PyVista dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in MESH_SUFFIXES:
        return pv.read(path)
    if suffix in STEP_SUFFIXES:
        return _step_to_pyvista(path)
    raise ValueError(f"Unsupported mesh format: {suffix}")


def load_mesh_bytes(data: bytes, suffix: str = ".stl") -> pv.DataSet:
    """Load mesh from in-memory bytes (writes a temporary file)."""
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        tmp = Path(f.name)
    try:
        return load_mesh_path(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def _step_to_pyvista(step_path: Path) -> pv.DataSet:
    """Tessellate STEP to STL via CadQuery, then read with PyVista."""
    import cadquery as cq

    shape = cq.importers.importStep(str(step_path))
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        stl_path = Path(f.name)
    try:
        cq.exporters.export(shape, str(stl_path), exportType="STL")
        if not stl_path.exists() or stl_path.stat().st_size == 0:
            raise RuntimeError("CadQuery produced an empty STL from STEP")
        return pv.read(stl_path)
    finally:
        stl_path.unlink(missing_ok=True)
