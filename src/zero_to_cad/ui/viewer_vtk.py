"""Embedded VTK viewer (interactive; requires working OpenGL + Qt compositor)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from zero_to_cad.ui.mesh_loader import load_mesh_bytes, load_mesh_path

_MESH_COLOR = "lightsteelblue"


class VtkViewer3D(QWidget):
    """pyvistaqt QtInteractor embedded in the layout."""

    def __init__(self, title: str = "3D View", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._mesh = None
        self._show_edges = False

        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #333;")
        header.addWidget(title_label)
        header.addStretch()

        self._edges_check = QCheckBox("Wireframe edges")
        self._edges_check.setChecked(False)
        self._edges_check.toggled.connect(self._on_edges_toggled)
        header.addWidget(self._edges_check)

        self._reset_btn = QPushButton("Reset view")
        self._reset_btn.clicked.connect(self._on_reset_view)
        header.addWidget(self._reset_btn)
        layout.addLayout(header)

        hint = QLabel("Drag to orbit · Scroll to zoom · Right-drag to pan")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        self.plotter = QtInteractor(
            self,
            off_screen=False,
            auto_update=False,
        )
        self.plotter.setMinimumSize(260, 260)
        layout.addWidget(self.plotter, stretch=1)

        self._setup_scene()

    def _setup_scene(self) -> None:
        self.plotter.set_background("white")
        self.plotter.show_axes()
        self.plotter.show_grid()
        self.plotter.add_text(self._title, position="upper_edge", font_size=10)
        if self.plotter.iren is not None:
            self.plotter.enable_trackball_style()

    def _on_edges_toggled(self, checked: bool) -> None:
        self._show_edges = checked
        if self._mesh is not None:
            self._redraw_mesh()

    def _on_reset_view(self) -> None:
        if self._mesh is None:
            return
        self.plotter.view_isometric()
        self._sync_and_render()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._sync_and_render)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_and_render)

    def _sync_and_render(self) -> None:
        w = max(self.plotter.width(), 1)
        h = max(self.plotter.height(), 1)
        try:
            self.plotter.ren_win.SetSize(w, h)
            if self.plotter.iren is not None:
                self.plotter.iren.SetSize(w, h)
                self.plotter.iren.ConfigureEvent()
            self.plotter.ren_win.Render()
            self.plotter.render()
            self.plotter.update()
        except Exception:
            pass

    def clear(self) -> None:
        self._mesh = None
        self.plotter.clear()
        self._setup_scene()
        self._sync_and_render()

    def load_mesh(self, path: str | Path) -> None:
        self._display_mesh(load_mesh_path(path))

    def load_mesh_bytes(self, data: bytes, suffix: str = ".stl") -> None:
        self._display_mesh(load_mesh_bytes(data, suffix=suffix))

    def load_stl(self, path: str | Path) -> None:
        self.load_mesh(path)

    def _display_mesh(self, mesh) -> None:
        self._mesh = mesh
        self._redraw_mesh()

    def _redraw_mesh(self) -> None:
        self.plotter.clear()
        self._setup_scene()
        if self._mesh is None:
            return
        self.plotter.add_mesh(
            self._mesh,
            show_edges=self._show_edges,
            color=_MESH_COLOR,
            smooth_shading=True,
        )
        self.plotter.reset_camera()
        self.plotter.view_isometric()
        QTimer.singleShot(50, self._sync_and_render)
