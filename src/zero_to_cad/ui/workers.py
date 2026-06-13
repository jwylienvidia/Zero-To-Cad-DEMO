"""Background workers for download, inference, and execution."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtCore import QThread, Signal

from zero_to_cad.dataset.downloader import download_test_split
from zero_to_cad.execute.sandbox import ExecutionResult, execute_cadquery
from zero_to_cad.config import ModelEntry
from zero_to_cad.inference import InferenceModel, load_model
from zero_to_cad.inference.prompts import (
    REFINE_FIX_SYSTEM_PROMPT,
    REFINE_SYSTEM_PROMPT,
    build_refine_fix_user_text,
    build_refine_user_text,
    format_execution_error,
    prepare_refine_images,
)
from zero_to_cad.ui.mesh_loader import load_mesh_path
from zero_to_cad.ui.viewer_render import render_mesh_views


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
            model = load_model(self.entry)
            self.finished_ok.emit(model)
        except Exception as e:
            self.failed.emit(str(e))


class GenerateWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        model: InferenceModel,
        views: list[Image.Image],
        *,
        user_text: str | None = None,
        progress_text: str = "Generating CadQuery code…",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.views = views
        self.user_text = user_text
        self.progress_text = progress_text

    def run(self) -> None:
        try:
            self.progress.emit(self.progress_text)
            code = self.model.generate(self.views, user_text=self.user_text)
            self.finished_ok.emit(code)
        except Exception as e:
            self.failed.emit(str(e))


def _resolve_mesh_path(
    stl_path: Path | None,
    step_path: Path | None,
) -> Path | None:
    if stl_path and stl_path.exists():
        return stl_path
    if step_path and step_path.exists():
        return step_path
    return None


class RefineWorker(QThread):
    finished_ok = Signal(str, str)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        model: InferenceModel,
        views: list[Image.Image],
        code: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.model = model
        self.views = views
        self.code = code

    def run(self) -> None:
        try:
            self.progress.emit("Checking CadQuery script…")
            result = execute_cadquery(self.code)

            if not result.ok:
                error_msg = format_execution_error(result.error, result.traceback)
                self.progress.emit("Fixing script error…")
                user_text = build_refine_fix_user_text(self.code, error_msg)
                raw = self.model.generate(
                    [],
                    max_new_tokens=8192,
                    system_prompt=REFINE_FIX_SYSTEM_PROMPT,
                    user_text=user_text,
                )
                self.finished_ok.emit(raw, "fix")
                return

            mesh_path = _resolve_mesh_path(result.stl_path, result.step_path)
            if mesh_path is None:
                self.failed.emit("Execution succeeded but no mesh file was produced.")
                return

            if not self.views:
                self.failed.emit(
                    "Provide all 8 view images to refine geometry against the target."
                )
                return

            self.progress.emit("Rendering generated model views…")
            mesh = load_mesh_path(mesh_path)
            renders = render_mesh_views(mesh)
            target_views, render_views = prepare_refine_images(self.views, renders)

            combined = target_views + render_views
            user_text = build_refine_user_text(
                self.code,
                num_target=len(target_views),
                num_render=len(render_views),
            )

            self.progress.emit("Refining CadQuery code…")
            raw = self.model.generate(
                combined,
                max_new_tokens=8192,
                system_prompt=REFINE_SYSTEM_PROMPT,
                user_text=user_text,
            )
            self.finished_ok.emit(raw, "visual")
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
