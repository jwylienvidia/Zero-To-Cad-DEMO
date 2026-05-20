"""3D mesh viewer factory (image or VTK backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PySide6.QtWidgets import QWidget

from zero_to_cad.config import VIEWER_BACKEND
from zero_to_cad.ui.viewer_image import ImageViewer3D
from zero_to_cad.ui.viewer_vtk import VtkViewer3D


class MeshViewer(Protocol):
    def clear(self) -> None: ...
    def load_mesh(self, path: str | Path) -> None: ...
    def load_mesh_bytes(self, data: bytes, suffix: str = ".stl") -> None: ...
    def load_stl(self, path: str | Path) -> None: ...


def create_viewer(title: str, parent: QWidget | None = None) -> MeshViewer:
    """
    Create a 3D preview widget.

    Default backend is ``image`` (offscreen PyVista screenshot) because embedded
    VTK often shows desktop compositor artifacts on Linux + Qt6 + NVIDIA.
    Set ``ZERO_TO_CAD_VIEWER=vtk`` for an interactive VTK widget.
    """
    if VIEWER_BACKEND == "vtk":
        return VtkViewer3D(title, parent)
    return ImageViewer3D(title, parent)


# Backward-compatible alias used by main_window
Viewer3D = create_viewer
