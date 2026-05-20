"""Dataset UUID browser with filter."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DatasetBrowser(QGroupBox):
    """Browse test-split samples by UUID."""

    row_selected = Signal(str)
    download_requested = Signal()
    refresh_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Test dataset", parent)
        self._all_uuids: list[str] = []

        layout = QVBoxLayout(self)

        self.status_label = QLabel("No dataset loaded")
        layout.addWidget(self.status_label)

        filter_row = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter UUID…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_edit)
        layout.addLayout(filter_row)

        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.download_btn = QPushButton("Download test split…")
        self.download_btn.clicked.connect(self.download_requested.emit)
        self.refresh_btn = QPushButton("Refresh index")
        self.refresh_btn.clicked.connect(self.refresh_requested.emit)
        btn_row.addWidget(self.download_btn)
        btn_row.addWidget(self.refresh_btn)
        layout.addLayout(btn_row)

    def set_uuids(self, uuids: list[str], status: str | None = None) -> None:
        self._all_uuids = list(uuids)
        self._apply_filter()
        if status:
            self.status_label.setText(status)
        else:
            self.status_label.setText(f"{len(uuids)} samples")

    def _apply_filter(self) -> None:
        needle = self.filter_edit.text().strip().lower()
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for uuid in self._all_uuids:
            if not needle or needle in uuid.lower():
                self.list_widget.addItem(uuid)
        self.list_widget.blockSignals(False)

    def _on_item_changed(self, current, _previous) -> None:
        if current is not None:
            self.row_selected.emit(current.text())

    def current_uuid(self) -> str | None:
        item = self.list_widget.currentItem()
        return item.text() if item else None

    def set_busy(self, busy: bool, message: str = "") -> None:
        self.download_btn.setEnabled(not busy)
        self.refresh_btn.setEnabled(not busy)
        self.list_widget.setEnabled(not busy)
        if message:
            self.status_label.setText(message)
