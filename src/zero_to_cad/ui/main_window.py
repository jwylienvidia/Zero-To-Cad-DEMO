"""Main application window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from zero_to_cad.config import MODELS, ModelEntry, VIEWER_BACKEND, assets_dir, exports_dir
from zero_to_cad.dataset.downloader import is_test_split_downloaded
from zero_to_cad.dataset.parquet_store import ParquetStore
from zero_to_cad.execute.drop_test import DropTestParams, build_drop_test_code
from zero_to_cad.execute.sandbox import ExecutionResult
from zero_to_cad.export.asset import save_asset
from zero_to_cad.inference import InferenceModel
from zero_to_cad.inference.prompts import (
    build_reasoning_test_user_text,
    extract_cadquery_code,
    extract_reasoning,
)
from zero_to_cad.ui.code_panel import CodePanel
from zero_to_cad.ui.dataset_browser import DatasetBrowser
from zero_to_cad.ui.input_panel import InputPanel
from zero_to_cad.ui.run_history import RunHistoryPanel, new_run_record
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
        self._model: InferenceModel | None = None
        self._current_uuid: str | None = None
        self._gt_stl_bytes: bytes | None = None
        self._current_ground_truth_code: str = ""

        self._last_predicted_code: str | None = None
        self._last_predicted_result: ExecutionResult | None = None
        self._current_run_id: str | None = None

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

        self.run_history = RunHistoryPanel()
        self.code_panel.tab_widget.addTab(self.run_history, "History")

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
        self.act_generate = QAction("Generate", self)
        self.act_reasoning_test = QAction("Reasoning test", self)
        self.act_execute = QAction("Execute", self)
        self.act_drop_test = QAction("Drop test…", self)
        self.act_save_asset = QAction("Save asset…", self)
        self.act_export = QAction("Export row…", self)
        self.act_execute_gt = QAction("View GT mesh", self)

        toolbar.addAction(self.act_load_pngs)

        self.model_combo = QComboBox()
        for entry in MODELS:
            self.model_combo.addItem(entry.label, entry)
            idx = self.model_combo.count() - 1
            if entry.notes:
                self.model_combo.setItemData(
                    idx, entry.notes, Qt.ItemDataRole.ToolTipRole
                )
        toolbar.addWidget(self.model_combo)

        toolbar.addAction(self.act_load_model)
        toolbar.addSeparator()
        toolbar.addAction(self.act_generate)
        toolbar.addAction(self.act_reasoning_test)
        toolbar.addAction(self.act_execute)
        toolbar.addAction(self.act_drop_test)
        toolbar.addSeparator()
        toolbar.addAction(self.act_save_asset)
        toolbar.addAction(self.act_export)
        toolbar.addAction(self.act_execute_gt)

        self.act_generate.setEnabled(False)
        self.act_reasoning_test.setEnabled(False)
        self.act_reasoning_test.setToolTip(
            "With Cosmos-Reason loaded, generate reasoning for the selected "
            "dataset row using its ground-truth CadQuery code."
        )
        self.act_execute.setEnabled(False)
        self.act_drop_test.setEnabled(False)
        self.act_save_asset.setEnabled(False)

    def _connect_signals(self) -> None:
        self.act_load_pngs.triggered.connect(self.input_panel._load_from_files)
        self.act_load_model.triggered.connect(self._load_model)
        self.act_generate.triggered.connect(self._generate)
        self.act_reasoning_test.triggered.connect(self._run_reasoning_test)
        self.act_execute.triggered.connect(self._execute_predicted)
        self.act_drop_test.triggered.connect(self._run_drop_test)
        self.act_save_asset.triggered.connect(self._save_asset)
        self.act_export.triggered.connect(self._export_row)
        self.act_execute_gt.triggered.connect(self._view_gt_mesh)

        self.dataset_browser.download_requested.connect(self._download_dataset)
        self.dataset_browser.refresh_requested.connect(self._refresh_dataset_index)
        self.dataset_browser.row_selected.connect(self._on_row_selected)

        self.input_panel.views_changed.connect(self._update_generate_enabled)

        self.run_history.record_selected.connect(self._on_history_selected)

    def _selected_model_entry(self) -> ModelEntry:
        data = self.model_combo.currentData()
        if isinstance(data, ModelEntry):
            return data
        return MODELS[0]

    def _loaded_model_supports_reasoning_test(self) -> bool:
        if self._model is None:
            return False
        return "cosmos-reason" in self._model.entry.id.lower()

    def _hf_token_present(self) -> bool:
        import os

        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            return True
        try:
            from huggingface_hub import HfFolder

            return bool(HfFolder.get_token())
        except Exception:
            return False

    def _anthropic_key_present(self) -> bool:
        import os

        return bool(os.environ.get("ANTHROPIC_API_KEY"))

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
        self.code_panel.clear_reasoning()
        self._current_ground_truth_code = row.cadquery_code
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

        entry = self._selected_model_entry()
        if entry.backend == "anthropic" and not self._anthropic_key_present():
            QMessageBox.warning(
                self,
                "Anthropic API key missing",
                f"{entry.label} requires the ANTHROPIC_API_KEY environment variable.\n\n"
                f"{entry.notes}",
            )
        elif entry.backend == "vllm" and entry.gated and not self._hf_token_present():
            QMessageBox.warning(
                self,
                "Gated model",
                f"{entry.label} requires Hugging Face authentication.\n\n"
                f"Run `huggingface-cli login` and accept the model gate at:\n"
                f"https://huggingface.co/{entry.id}\n\n"
                f"{entry.notes}",
            )

        if self._model is not None:
            self._model.release()
            self._model = None
            self.act_reasoning_test.setEnabled(False)

        self.act_load_model.setEnabled(False)
        self._set_status(f"Loading model {entry.label}…")

        self._model_worker = LoadModelWorker(entry)
        self._model_worker.progress.connect(self._set_status)
        self._model_worker.finished_ok.connect(self._on_model_loaded)
        self._model_worker.failed.connect(self._on_model_failed)
        self._model_worker.start()

    def _on_model_loaded(self, model: InferenceModel) -> None:
        self._model = model
        self.act_load_model.setEnabled(True)
        self._set_status(f"Model loaded: {model.entry.label}")
        self._update_generate_enabled()
        self._update_reasoning_enabled()

    def _on_model_failed(self, error: str) -> None:
        self.act_load_model.setEnabled(True)
        self.act_reasoning_test.setEnabled(False)
        QMessageBox.critical(self, "Model load failed", error)
        self._set_status("Model load failed")

    def _update_generate_enabled(self) -> None:
        ready = self._model is not None and self.input_panel.is_complete()
        self.act_generate.setEnabled(ready)
        self._update_reasoning_enabled()

    def _update_reasoning_enabled(self) -> None:
        ready = (
            self._loaded_model_supports_reasoning_test()
            and self.input_panel.is_complete()
            and bool(self._current_ground_truth_code.strip())
        )
        self.act_reasoning_test.setEnabled(ready)

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
        self.act_reasoning_test.setEnabled(False)
        self._set_status("Generating…")
        self._generate_worker = GenerateWorker(self._model, views)
        self._generate_worker.progress.connect(self._set_status)
        self._generate_worker.finished_ok.connect(self._on_generate_done)
        self._generate_worker.failed.connect(self._on_generate_failed)
        self._generate_worker.start()

    def _on_generate_done(self, raw_output: str) -> None:
        reasoning = extract_reasoning(raw_output)
        code = extract_cadquery_code(raw_output)
        if reasoning:
            self.code_panel.set_reasoning(reasoning)
        else:
            self.code_panel.clear_reasoning()
        # set_predicted last so the Predicted tab stays focused after generation.
        self.code_panel.set_predicted(code)
        self.act_execute.setEnabled(bool(code.strip()))
        self.act_generate.setEnabled(True)
        self._update_reasoning_enabled()
        self._set_status("Generation complete.")

        if self._model is None:
            return
        views = self.input_panel.get_views()
        source = self._current_uuid or "custom"
        rec = new_run_record(
            model_id=self._model.entry.id,
            model_label=self._model.entry.label,
            source=source,
            code=code,
            views=views or [],
        )
        self._current_run_id = rec.run_id
        self.run_history.add_run(rec)

    def _on_history_selected(self, rec: object) -> None:
        from zero_to_cad.ui.run_history import RunRecord

        if not isinstance(rec, RunRecord):
            return

        self._current_run_id = rec.run_id
        self.code_panel.set_predicted(rec.code)
        self._last_predicted_code = rec.code
        self._last_predicted_result = rec.result

        if rec.views:
            self.input_panel.set_views(rec.views)

        mesh_path = None
        if rec.result:
            if rec.result.stl_path and rec.result.stl_path.exists():
                mesh_path = rec.result.stl_path
            elif rec.result.step_path and rec.result.step_path.exists():
                mesh_path = rec.result.step_path

        if mesh_path:
            try:
                self.viewer_predicted.load_mesh(mesh_path)
            except Exception as e:
                self._set_status(f"History mesh load failed: {e}")
        else:
            self.viewer_predicted.clear()

        self.act_execute.setEnabled(bool(rec.code.strip()))
        self.act_drop_test.setEnabled(rec.result is not None and rec.result.ok)
        self.act_save_asset.setEnabled(rec.result is not None and rec.result.ok)
        self._set_status(f"Restored run {rec.run_id} ({rec.model_label})")

    def _on_generate_failed(self, error: str) -> None:
        self.act_generate.setEnabled(True)
        self._update_reasoning_enabled()
        QMessageBox.critical(self, "Generation failed", error)
        self._set_status("Generation failed")

    def _run_reasoning_test(self) -> None:
        if not self._model:
            QMessageBox.information(self, "Reasoning test", "Load Cosmos-Reason first.")
            return
        if not self._loaded_model_supports_reasoning_test():
            QMessageBox.information(
                self,
                "Reasoning test",
                "Reasoning test is intended for the Cosmos-Reason baseline model.",
            )
            return
        views = self.input_panel.get_views()
        if not views:
            QMessageBox.information(
                self,
                "Reasoning test",
                "Select a dataset row with all 8 views first.",
            )
            return
        ground_truth = self._current_ground_truth_code.strip()
        if not ground_truth:
            QMessageBox.information(
                self,
                "Reasoning test",
                "Select a dataset row with ground-truth CadQuery code first.",
            )
            return
        if self._generate_worker and self._generate_worker.isRunning():
            return

        prompt = build_reasoning_test_user_text(ground_truth)
        self.act_generate.setEnabled(False)
        self.act_reasoning_test.setEnabled(False)
        self._set_status("Generating reasoning test output…")
        self._generate_worker = GenerateWorker(
            self._model,
            views,
            user_text=prompt,
            progress_text="Generating reasoning from ground truth…",
        )
        self._generate_worker.progress.connect(self._set_status)
        self._generate_worker.finished_ok.connect(self._on_reasoning_done)
        self._generate_worker.failed.connect(self._on_reasoning_failed)
        self._generate_worker.start()

    def _on_reasoning_done(self, reasoning: str) -> None:
        self.code_panel.set_reasoning(reasoning)
        self.act_generate.setEnabled(True)
        self._update_reasoning_enabled()
        self._set_status("Reasoning test complete.")

    def _on_reasoning_failed(self, error: str) -> None:
        self.act_generate.setEnabled(True)
        self._update_reasoning_enabled()
        QMessageBox.critical(self, "Reasoning test failed", error)
        self._set_status("Reasoning test failed")

    def _execute_predicted(self) -> None:
        code = self.code_panel.get_predicted()
        if not code:
            QMessageBox.information(self, "Execute", "No predicted code to execute.")
            return
        self._run_execute(code, target="predicted")

    def _run_execute(
        self, code: str, target: str = "predicted", *, label: str = "Execution"
    ) -> None:
        if self._execute_worker and self._execute_worker.isRunning():
            return
        self.act_execute.setEnabled(False)
        self.act_drop_test.setEnabled(False)
        self._set_status(f"{label}: running CadQuery…")
        self._execute_worker = ExecuteWorker(code)
        self._execute_worker.progress.connect(self._set_status)
        self._execute_worker.finished_ok.connect(
            lambda r: self._on_execute_done(r, target, code, label)
        )
        self._execute_worker.failed.connect(self._on_execute_failed)
        self._execute_worker.start()

    def _on_execute_done(
        self,
        result: ExecutionResult,
        target: str,
        code: str,
        label: str = "Execution",
    ) -> None:
        self.act_execute.setEnabled(True)
        viewer = (
            self.viewer_predicted if target == "predicted" else self.viewer_ground_truth
        )
        mesh_path = None
        if result.stl_path and result.stl_path.exists():
            mesh_path = result.stl_path
        elif result.step_path and result.step_path.exists():
            mesh_path = result.step_path

        if target == "predicted":
            self._last_predicted_code = code
            self._last_predicted_result = result
            self.act_drop_test.setEnabled(bool(mesh_path))
            self.act_save_asset.setEnabled(bool(mesh_path))
            if self._current_run_id:
                self.run_history.update_run(self._current_run_id, result)

        if mesh_path:
            try:
                viewer.load_mesh(mesh_path)
                self._set_status(f"{label} OK — {mesh_path}")
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Viewer",
                    f"Mesh saved but viewer failed to load:\n{e}\n\nFile: {mesh_path}",
                )
        else:
            self._set_status(f"{label} OK but no mesh file produced.")

    def _on_execute_failed(self, error: str) -> None:
        self.act_execute.setEnabled(True)
        self.act_drop_test.setEnabled(self._last_predicted_result is not None)
        QMessageBox.critical(self, "Execution failed", error)
        self._set_status("Execution failed")

    def _run_drop_test(self) -> None:
        code = self._last_predicted_code or self.code_panel.get_predicted()
        if not code:
            QMessageBox.information(
                self,
                "Drop test",
                "Execute a prediction first — drop test needs a successfully "
                "executed CadQuery script.",
            )
            return
        count, accepted = QInputDialog.getInt(
            self, "Drop test", "Number of copies:", 8, 1, 200, 1
        )
        if not accepted:
            return
        try:
            drop_code = build_drop_test_code(code, DropTestParams(count=count))
        except Exception as e:
            QMessageBox.critical(self, "Drop test", f"Failed to build script:\n{e}")
            return
        self.code_panel.set_predicted(drop_code)
        self._run_execute(drop_code, target="predicted", label=f"Drop test x{count}")

    def _save_asset(self) -> None:
        result = self._last_predicted_result
        if result is None or not result.ok:
            QMessageBox.information(
                self,
                "Save asset",
                "Execute a prediction first — saving needs a successful run.",
            )
            return

        default_dir = assets_dir()
        default_dir.mkdir(parents=True, exist_ok=True)
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose asset folder location", str(default_dir)
        )
        if not chosen:
            return

        suggested_name, accepted = QInputDialog.getText(
            self, "Save asset", "Asset name (subfolder):", text="prediction"
        )
        if not accepted or not suggested_name.strip():
            return
        out_dir = Path(chosen) / suggested_name.strip()

        views = self.input_panel.get_views()
        try:
            paths = save_asset(
                out_dir,
                name="model",
                code=self._last_predicted_code,
                step_path=result.step_path,
                stl_path=result.stl_path,
                views=views,
            )
        except Exception as e:
            QMessageBox.critical(self, "Save asset failed", str(e))
            return

        self._set_status(f"Asset saved to {paths.asset_dir}")
        QMessageBox.information(
            self,
            "Save asset",
            f"Asset folder written:\n{paths.asset_dir}\n\n"
            "Contains STEP, STL, OBJ + MTL, textures/ (albedo + view PNGs), "
            "views/, code.py and manifest.json.",
        )

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
