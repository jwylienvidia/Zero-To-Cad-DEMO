"""Background workers for download, inference, and execution."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QThread, Signal

from zero_to_cad.dataset.downloader import download_test_split
from zero_to_cad.execute.sandbox import ExecutionResult, execute_cadquery
from zero_to_cad.config import ModelEntry
from zero_to_cad.inference.model import CadModel


class DownloadWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)
    progress = Signal(str)

    def run(self) -> None:
        try:
            self.progress.emit("Listing remote shards…")
            dest = download_test_split(
                progress=lambda path, i, n: self.progress.emit(
                    f"Downloading {i + 1}/{n}: {Path(path).name}"
                ),
            )
            self.finished_ok.emit(str(dest))
        except Exception as e:
            self.failed.emit(str(e))


class LoadModelWorker(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, entry: ModelEntry, parent=None) -> None:
        super().__init__(parent)
        self.entry = entry

    def run(self) -> None:
        try:
            self.progress.emit(
                f"Loading {self.entry.label} (first run downloads weights)…"
            )
            model = CadModel(entry=self.entry)
            self.finished_ok.emit(model)
        except Exception as e:
            self.failed.emit(str(e))


class GenerateWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        model: CadModel,
        views: list[Image.Image],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.views = views

    def run(self) -> None:
        try:
            self.progress.emit("Generating CadQuery code…")
            code = self.model.generate(self.views)
            self.finished_ok.emit(code)
        except Exception as e:
            self.failed.emit(str(e))


class ExecuteWorker(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, code: str, parent=None) -> None:
        super().__init__(parent)
        self.code = code

    def run(self) -> None:
        try:
            self.progress.emit("Executing CadQuery in sandbox…")
            result: ExecutionResult = execute_cadquery(self.code)
            if result.ok:
                self.finished_ok.emit(result)
            else:
                msg = result.error or "Execution failed"
                if result.traceback:
                    msg = f"{msg}\n\n{result.traceback}"
                self.failed.emit(msg)
        except Exception as e:
            self.failed.emit(str(e))
