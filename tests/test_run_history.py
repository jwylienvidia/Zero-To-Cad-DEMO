"""Tests for run history panel."""

from __future__ import annotations

import sys

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from zero_to_cad.ui.run_history import RunHistoryPanel, RunRecord, new_run_record


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_new_run_record_copies_views() -> None:
    views = [Image.new("RGB", (8, 8), color=(i, 0, 0)) for i in range(8)]
    rec = new_run_record(
        model_id="test/model",
        model_label="Test",
        source="custom",
        code="result = 1",
        views=views,
    )
    assert rec.run_id
    assert len(rec.views) == 8
    assert rec.views is not views
    rec.views[0].putpixel((0, 0), (255, 255, 255))
    assert views[0].getpixel((0, 0)) != (255, 255, 255)


def test_run_history_add_run(qapp) -> None:
    panel = RunHistoryPanel()
    rec = new_run_record(
        model_id="test/model",
        model_label="Test Model",
        source="sample-uuid",
        code="import cadquery as cq",
        views=[],
    )
    panel.add_run(rec)
    assert panel.list_widget.count() == 1
    item = panel.list_widget.item(0)
    assert item is not None
    assert "Test Model" in item.text()
    assert "sample-uuid" in item.text()


def test_run_history_update_run(qapp) -> None:
    from zero_to_cad.execute.sandbox import ExecutionResult

    panel = RunHistoryPanel()
    rec = new_run_record(
        model_id="test/model",
        model_label="Test",
        source="custom",
        code="x = 1",
        views=[],
    )
    panel.add_run(rec)
    result = ExecutionResult(ok=True)
    panel.update_run(rec.run_id, result)
    item = panel.list_widget.item(0)
    assert item is not None
    assert item.text().endswith("ok")
