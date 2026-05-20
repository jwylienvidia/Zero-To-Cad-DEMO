"""Eight-view input grid with drag-and-drop."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zero_to_cad.config import NUM_VIEWS


class _ImageTile(QLabel):
    """Single view slot accepting image drops."""

    image_dropped = Signal(int, str)

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.index = index
        self._path: str | None = None
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(120, 120)
        self.setStyleSheet(
            "QLabel { border: 1px dashed #888; background: #2a2a2a; color: #aaa; }"
        )
        self.setText(f"View {index}\n(drop PNG)")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp", ".bmp")
            ):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        self.set_image_path(path)
        self.image_dropped.emit(self.index, path)
        event.acceptProposedAction()

    def set_image_path(self, path: str) -> None:
        self._path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
            self.setText("")

    def get_image(self) -> Image.Image | None:
        if not self._path:
            return None
        return Image.open(self._path).convert("RGB")

    def clear_slot(self) -> None:
        self._path = None
        self.clear()
        self.setText(f"View {self.index}\n(drop PNG)")


class InputPanel(QGroupBox):
    """Grid of 8 rendered views for model input."""

    views_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Input views (8 × 256×256)", parent)
        self._tiles: list[_ImageTile] = []

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setSpacing(4)

        for i in range(NUM_VIEWS):
            tile = _ImageTile(i)
            tile.image_dropped.connect(self._on_tile_changed)
            self._tiles.append(tile)
            row, col = divmod(i, 4)
            grid.addWidget(tile, row, col)

        layout.addLayout(grid)

        btn_row = QWidget()
        btn_layout = QGridLayout(btn_row)
        load_btn = QPushButton("Load 8 PNGs…")
        load_btn.clicked.connect(self._load_from_files)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_views)
        btn_layout.addWidget(load_btn, 0, 0)
        btn_layout.addWidget(clear_btn, 0, 1)
        layout.addWidget(btn_row)

    def _on_tile_changed(self, _index: int, _path: str) -> None:
        self.views_changed.emit()

    def _load_from_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select 8 view images",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not paths:
            return
        for i, path in enumerate(paths[:NUM_VIEWS]):
            self._tiles[i].set_image_path(path)
        self.views_changed.emit()

    def set_views(self, views: list[Image.Image]) -> None:
        import tempfile

        for i, img in enumerate(views[:NUM_VIEWS]):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f.name)
                self._tiles[i].set_image_path(f.name)
        self.views_changed.emit()

    def get_views(self) -> list[Image.Image] | None:
        images = [t.get_image() for t in self._tiles]
        if any(img is None for img in images):
            return None
        return images  # type: ignore[return-value]

    def clear_views(self) -> None:
        for tile in self._tiles:
            tile.clear_slot()
        self.views_changed.emit()

    def is_complete(self) -> bool:
        return self.get_views() is not None
