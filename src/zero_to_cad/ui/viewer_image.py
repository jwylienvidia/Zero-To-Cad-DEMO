"""Offscreen-rendered mesh preview with orbit controls."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from zero_to_cad.ui.orbit_viewport import OrbitViewport


class ImageViewer3D(QWidget):
    """Shows a PyVista offscreen render with orbit + zoom and optional edge lines."""

    def __init__(self, title: str = "3D View", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title

        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet("font-weight: bold; color: #333;")
        header.addWidget(self._title_label)
        header.addStretch()

        self._edges_check = QCheckBox("Wireframe edges")
        self._edges_check.setChecked(False)
        self._edges_check.toggled.connect(self._on_edges_toggled)
        header.addWidget(self._edges_check)

        self._reset_btn = QPushButton("Reset view")
        self._reset_btn.setToolTip("Restore default isometric camera")
        self._reset_btn.clicked.connect(self._on_reset_view)
        header.addWidget(self._reset_btn)

        layout.addLayout(header)

        hint = QLabel("Drag to orbit · Scroll to zoom")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        self._viewport = OrbitViewport()
        layout.addWidget(self._viewport, stretch=1)

    def _on_edges_toggled(self, checked: bool) -> None:
        self._viewport.set_show_edges(checked)

    def _on_reset_view(self) -> None:
        self._viewport.reset_camera()

    def clear(self) -> None:
        self._viewport.clear_mesh()

    def load_mesh(self, path: str | Path) -> None:
        from zero_to_cad.ui.mesh_loader import load_mesh_path

        self._viewport.set_mesh(load_mesh_path(path))

    def load_mesh_bytes(self, data: bytes, suffix: str = ".stl") -> None:
        from zero_to_cad.ui.mesh_loader import load_mesh_bytes

        self._viewport.set_mesh(load_mesh_bytes(data, suffix=suffix))

    def load_stl(self, path: str | Path) -> None:
        self.load_mesh(path)
