"""Mouse-driven orbit viewport (offscreen re-render on interaction)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QLabel

from zero_to_cad.ui.qt_image import numpy_to_pixmap
from zero_to_cad.ui.viewer_render import CameraState, render_mesh_to_array

_ORBIT_SENSITIVITY = 0.4
_RENDER_DEBOUNCE_MS = 40


class OrbitViewport(QLabel):
    """
    Displays an offscreen mesh render and supports orbit (drag) and zoom (wheel).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(260, 260)
        self.setText("No mesh loaded")
        self.setStyleSheet(
            "background-color: #f4f4f4; border: 1px solid #bbb; color: #666;"
        )

        self._mesh = None
        self._camera = CameraState()
        self._show_edges = False
        self._pixmap: QPixmap | None = None
        self._dragging = False
        self._last_mouse: tuple[float, float] | None = None

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_now)

    def set_show_edges(self, enabled: bool) -> None:
        self._show_edges = enabled
        if self._mesh is not None:
            self._schedule_render()

    def show_edges(self) -> bool:
        return self._show_edges

    def set_mesh(self, mesh) -> None:
        self._mesh = mesh
        self._camera.reset()
        self._schedule_render(immediate=True)

    def clear_mesh(self) -> None:
        self._mesh = None
        self._pixmap = None
        self.setText("No mesh loaded")
        self.setPixmap(QPixmap())

    def reset_camera(self) -> None:
        self._camera.reset()
        if self._mesh is not None:
            self._schedule_render(immediate=True)

    def _schedule_render(self, *, immediate: bool = False) -> None:
        if self._mesh is None:
            return
        if immediate:
            self._render_timer.stop()
            self._render_now()
        else:
            self._render_timer.start(_RENDER_DEBOUNCE_MS)

    def _render_now(self) -> None:
        if self._mesh is None:
            return
        try:
            image = render_mesh_to_array(
                self._mesh,
                show_edges=self._show_edges,
                camera=self._camera,
            )
        except Exception:
            return

        self._pixmap = numpy_to_pixmap(image)
        self.setText("")
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        target = self.size()
        if target.width() < 8 or target.height() < 8:
            return
        scaled = self._pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_pixmap()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._mesh is not None:
            self._dragging = True
            self._last_mouse = (event.position().x(), event.position().y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and self._last_mouse is not None and self._mesh is not None:
            x, y = event.position().x(), event.position().y()
            dx = x - self._last_mouse[0]
            dy = y - self._last_mouse[1]
            self._camera.azimuth += dx * _ORBIT_SENSITIVITY
            self._camera.elevation = float(
                np.clip(self._camera.elevation - dy * _ORBIT_SENSITIVITY, -89.0, 89.0)
            )
            self._last_mouse = (x, y)
            self._schedule_render()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_mouse = None
            if self._mesh is not None:
                self._schedule_render(immediate=True)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._mesh is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self._camera.zoom = min(5.0, self._camera.zoom * 1.12)
        elif delta < 0:
            self._camera.zoom = max(0.15, self._camera.zoom / 1.12)
        self._schedule_render(immediate=True)
        event.accept()
