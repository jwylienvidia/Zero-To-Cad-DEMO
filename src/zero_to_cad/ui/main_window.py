"""Main application window."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from zero_to_cad.config import MODEL_ID, exports_dir
from zero_to_cad.dataset.downloader import is_test_split_downloaded
from zero_to_cad.dataset.parquet_store import ParquetStore
from zero_to_cad.execute.sandbox import ExecutionResult
from zero_to_cad.inference.model import CadModel
from zero_to_cad.ui.code_panel import CodePanel
from zero_to_cad.ui.dataset_browser import DatasetBrowser
from zero_to_cad.ui.input_panel import InputPanel
from zero_to_cad.config import VIEWER_BACKEND
from zero_to_cad.ui.viewer_3d import create_viewer
from zero_to_cad.ui.workers import (
    DownloadWorker,
    ExecuteWorker,
    GenerateWorker,
    LoadModelWorker,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Zero-To-CAD")
        self.resize(1600, 900)

        self._store = ParquetStore()
        self._model: CadModel | None = None
        self._current_uuid: str | None = None
        self._gt_stl_bytes: bytes | None = None

        self._download_worker: DownloadWorker | None = None
        self._model_worker: LoadModelWorker | None = None
        self._generate_worker: GenerateWorker | None = None
        self._execute_worker: ExecuteWorker | None = None

        self._build_ui()
        self._build_toolbar()
        self._connect_signals()
        self._refresh_dataset_index()
        backend = "screenshot" if VIEWER_BACKEND != "vtk" else "VTK"
        self._set_status(
            f"Ready ({backend} 3D viewer). Download test split or load 8 PNGs, then load the model."
        )

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        main_split = QSplitter(Qt.Orientation.Horizontal)

        self.dataset_browser = DatasetBrowser()
        main_split.addWidget(self.dataset_browser)

        center_split = QSplitter(Qt.Orientation.Vertical)
        self.input_panel = InputPanel()
        self.code_panel = CodePanel()
        center_split.addWidget(self.input_panel)
        center_split.addWidget(self.code_panel)
        center_split.setStretchFactor(0, 2)
        center_split.setStretchFactor(1, 3)
        main_split.addWidget(center_split)

        viewer_split = QSplitter(Qt.Orientation.Vertical)
        self.viewer_predicted = create_viewer("Predicted")
        self.viewer_ground_truth = create_viewer("Ground truth")
        viewer_split.addWidget(self.viewer_predicted)
        viewer_split.addWidget(self.viewer_ground_truth)
        viewer_split.setStretchFactor(0, 1)
        viewer_split.setStretchFactor(1, 1)
        viewer_split.setSizes([450, 450])
        viewer_split.setMinimumWidth(300)
        main_split.addWidget(viewer_split)

        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 2)
        main_split.setStretchFactor(2, 2)

        root.addWidget(main_split)

        self.setStatusBar(QStatusBar())

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        self.act_load_pngs = QAction("Load 8 PNGs…", self)
        self.act_load_model = QAction("Load model", self)
        self.act_reload_model = QAction("Reload model", self)
        self.act_generate = QAction("Generate", self)
        self.act_execute = QAction("Execute", self)
        self.act_export = QAction("Export row…", self)
        self.act_execute_gt = QAction("View GT mesh", self)

        toolbar.addAction(self.act_load_pngs)
        toolbar.addAction(self.act_load_model)
        toolbar.addAction(self.act_reload_model)
        toolbar.addSeparator()
        toolbar.addAction(self.act_generate)
        toolbar.addAction(self.act_execute)
        toolbar.addSeparator()
        toolbar.addAction(self.act_export)
        toolbar.addAction(self.act_execute_gt)

        self.act_generate.setEnabled(False)
        self.act_execute.setEnabled(False)

    def _connect_signals(self) -> None:
        self.act_load_pngs.triggered.connect(self.input_panel._load_from_files)
        self.act_load_model.triggered.connect(self._load_model)
        self.act_reload_model.triggered.connect(self._load_model)
        self.act_generate.triggered.connect(self._generate)
        self.act_execute.triggered.connect(self._execute_predicted)
        self.act_export.triggered.connect(self._export_row)
        self.act_execute_gt.triggered.connect(self._view_gt_mesh)

        self.dataset_browser.download_requested.connect(self._download_dataset)
        self.dataset_browser.refresh_requested.connect(self._refresh_dataset_index)
        self.dataset_browser.row_selected.connect(self._on_row_selected)

        self.input_panel.views_changed.connect(self._update_generate_enabled)

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _refresh_dataset_index(self) -> None:
        if not is_test_split_downloaded(self._store.data_dir):
            self.dataset_browser.set_uuids([], "Test split not downloaded")
            return
        try:
            count = self._store.refresh_index()
            uuids = self._store.list_uuids()
            self.dataset_browser.set_uuids(uuids, f"{count} samples indexed")
        except Exception as e:
            self.dataset_browser.set_uuids([], f"Index error: {e}")

    def _download_dataset(self) -> None:
        if self._download_worker and self._download_worker.isRunning():
            return
        self.dataset_browser.set_busy(True, "Downloading test split…")
        self._download_worker = DownloadWorker()
        self._download_worker.progress.connect(self._set_status)
        self._download_worker.finished_ok.connect(self._on_download_done)
        self._download_worker.failed.connect(self._on_download_failed)
        self._download_worker.start()

    def _on_download_done(self, dest: str) -> None:
        self.dataset_browser.set_busy(False)
        self._set_status(f"Download complete: {dest}")
        self._refresh_dataset_index()

    def _on_download_failed(self, error: str) -> None:
        self.dataset_browser.set_busy(False)
        QMessageBox.critical(self, "Download failed", error)
        self._set_status("Download failed")

    def _on_row_selected(self, uuid: str) -> None:
        self._current_uuid = uuid
        self._set_status(f"Loading sample {uuid[:16]}…")
        try:
            row = self._store.get_row(uuid)
        except Exception as e:
            QMessageBox.warning(self, "Load row", str(e))
            return

        self.input_panel.set_views(row.views)
        self.code_panel.set_ground_truth(row.cadquery_code)
        self._gt_stl_bytes = row.stl_bytes

        try:
            if row.stl_bytes:
                self.viewer_ground_truth.load_mesh_bytes(row.stl_bytes, suffix=".stl")
            elif row.step_bytes:
                self.viewer_ground_truth.load_mesh_bytes(row.step_bytes, suffix=".step")
            else:
                self.viewer_ground_truth.clear()
        except Exception as e:
            self.viewer_ground_truth.clear()
            self._set_status(f"GT mesh load failed: {e}")

        meta = []
        if row.num_faces is not None:
            meta.append(f"faces={row.num_faces}")
        if row.cadquery_ops_count is not None:
            meta.append(f"ops={row.cadquery_ops_count}")
        meta_str = f" ({', '.join(meta)})" if meta else ""
        self._set_status(f"Loaded {uuid[:16]}…{meta_str}")
        self._update_generate_enabled()

    def _load_model(self) -> None:
        if self._model_worker and self._model_worker.isRunning():
            return
        self.act_load_model.setEnabled(False)
        self.act_reload_model.setEnabled(False)
        self._set_status(f"Loading model {MODEL_ID}…")

        self._model_worker = LoadModelWorker(MODEL_ID)
        self._model_worker.progress.connect(self._set_status)
        self._model_worker.finished_ok.connect(self._on_model_loaded)
        self._model_worker.failed.connect(self._on_model_failed)
        self._model_worker.start()

    def _on_model_loaded(self, model: CadModel) -> None:
        self._model = model
        self.act_load_model.setEnabled(True)
        self.act_reload_model.setEnabled(True)
        self._set_status("Model loaded.")
        self._update_generate_enabled()

    def _on_model_failed(self, error: str) -> None:
        self.act_load_model.setEnabled(True)
        self.act_reload_model.setEnabled(True)
        QMessageBox.critical(self, "Model load failed", error)
        self._set_status("Model load failed")

    def _update_generate_enabled(self) -> None:
        ready = self._model is not None and self.input_panel.is_complete()
        self.act_generate.setEnabled(ready)

    def _generate(self) -> None:
        if not self._model:
            QMessageBox.information(self, "Generate", "Load the model first.")
            return
        views = self.input_panel.get_views()
        if not views:
            QMessageBox.information(self, "Generate", "Provide all 8 view images.")
            return
        if self._generate_worker and self._generate_worker.isRunning():
            return

        self.act_generate.setEnabled(False)
        self._set_status("Generating…")
        self._generate_worker = GenerateWorker(self._model, views)
        self._generate_worker.progress.connect(self._set_status)
        self._generate_worker.finished_ok.connect(self._on_generate_done)
        self._generate_worker.failed.connect(self._on_generate_failed)
        self._generate_worker.start()

    def _on_generate_done(self, code: str) -> None:
        self.code_panel.set_predicted(code)
        self.act_execute.setEnabled(bool(code.strip()))
        self.act_generate.setEnabled(True)
        self._set_status("Generation complete.")

    def _on_generate_failed(self, error: str) -> None:
        self.act_generate.setEnabled(True)
        QMessageBox.critical(self, "Generation failed", error)
        self._set_status("Generation failed")

    def _execute_predicted(self) -> None:
        code = self.code_panel.get_predicted()
        if not code:
            QMessageBox.information(self, "Execute", "No predicted code to execute.")
            return
        self._run_execute(code, target="predicted")

    def _run_execute(self, code: str, target: str = "predicted") -> None:
        if self._execute_worker and self._execute_worker.isRunning():
            return
        self.act_execute.setEnabled(False)
        self._set_status("Executing CadQuery…")
        self._execute_worker = ExecuteWorker(code)
        self._execute_worker.progress.connect(self._set_status)
        self._execute_worker.finished_ok.connect(
            lambda r: self._on_execute_done(r, target)
        )
        self._execute_worker.failed.connect(self._on_execute_failed)
        self._execute_worker.start()

    def _on_execute_done(self, result: ExecutionResult, target: str) -> None:
        self.act_execute.setEnabled(True)
        viewer = (
            self.viewer_predicted if target == "predicted" else self.viewer_ground_truth
        )
        mesh_path = None
        if result.stl_path and result.stl_path.exists():
            mesh_path = result.stl_path
        elif result.step_path and result.step_path.exists():
            mesh_path = result.step_path
        if mesh_path:
            try:
                viewer.load_mesh(mesh_path)
                self._set_status(f"Execution OK — {mesh_path}")
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Viewer",
                    f"Mesh saved but viewer failed to load:\n{e}\n\nFile: {mesh_path}",
                )
        else:
            self._set_status("Execution OK but no mesh file produced.")

    def _on_execute_failed(self, error: str) -> None:
        self.act_execute.setEnabled(True)
        QMessageBox.critical(self, "Execution failed", error)
        self._set_status("Execution failed")

    def _view_gt_mesh(self) -> None:
        if self._gt_stl_bytes:
            try:
                self.viewer_ground_truth.load_mesh_bytes(self._gt_stl_bytes)
                self._set_status("Ground-truth mesh displayed.")
            except Exception as e:
                QMessageBox.warning(self, "GT mesh", str(e))
            return
        code = self.code_panel.ground_truth_edit.toPlainText().strip()
        if code:
            self._run_execute(code, target="ground_truth")
        else:
            QMessageBox.information(
                self,
                "GT mesh",
                "Select a dataset row or provide ground-truth code.",
            )

    def _export_row(self) -> None:
        uuid = self._current_uuid or self.dataset_browser.current_uuid()
        if not uuid:
            QMessageBox.information(self, "Export", "Select a dataset row first.")
            return
        out = QFileDialog.getExistingDirectory(
            self,
            "Export sample to folder",
            str(exports_dir()),
        )
        if not out:
            return
        try:
            path = self._store.export_row(uuid, Path(out))
            self._set_status(f"Exported to {path}")
            QMessageBox.information(self, "Export", f"Exported to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
