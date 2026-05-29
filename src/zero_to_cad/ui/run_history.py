"""Per-session run history for model comparisons."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGroupBox, QListWidget, QListWidgetItem, QVBoxLayout

from zero_to_cad.execute.sandbox import ExecutionResult


@dataclass
class RunRecord:
    """One generation (and optional execution) from a specific model."""

    run_id: str
    timestamp: datetime
    model_id: str
    model_label: str
    source: str
    code: str
    result: ExecutionResult | None = None
    views: list[Image.Image] = field(default_factory=list)


def _format_row(rec: RunRecord) -> str:
    ts = rec.timestamp.strftime("%H:%M:%S")
    source = rec.source if len(rec.source) <= 16 else f"{rec.source[:16]}…"
    status = "ok" if rec.result and rec.result.ok else "pending"
    return f"{ts}  {rec.model_label}  {source}  {status}"


class RunHistoryPanel(QGroupBox):
    """List of past runs; click an entry to restore code, views, and mesh."""

    record_selected = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__("Run history", parent)
        self._records: dict[str, RunRecord] = {}

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

    def add_run(self, rec: RunRecord) -> None:
        self._records[rec.run_id] = rec
        item = QListWidgetItem(_format_row(rec))
        item.setData(Qt.ItemDataRole.UserRole, rec.run_id)
        self.list_widget.insertItem(0, item)

    def update_run(self, run_id: str, result: ExecutionResult) -> None:
        rec = self._records.get(run_id)
        if rec is None:
            return
        rec.result = result
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == run_id:
                item.setText(_format_row(rec))
                break

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        run_id = item.data(Qt.ItemDataRole.UserRole)
        if not run_id:
            return
        rec = self._records.get(run_id)
        if rec is not None:
            self.record_selected.emit(rec)


def new_run_record(
    *,
    model_id: str,
    model_label: str,
    source: str,
    code: str,
    views: list[Image.Image],
) -> RunRecord:
    return RunRecord(
        run_id=uuid.uuid4().hex[:8],
        timestamp=datetime.now(),
        model_id=model_id,
        model_label=model_label,
        source=source,
        code=code,
        views=[v.copy() for v in views],
    )
