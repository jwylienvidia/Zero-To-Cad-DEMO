"""Code editor panel with predicted and ground-truth tabs."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QPlainTextEdit, QTabWidget, QVBoxLayout


class CodePanel(QGroupBox):
    """Displays predicted and ground-truth CadQuery code."""

    def __init__(self, parent=None) -> None:
        super().__init__("CadQuery code", parent)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.predicted_edit = QPlainTextEdit()
        self.predicted_edit.setPlaceholderText(
            "Generated CadQuery code will appear here…"
        )
        self.predicted_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        self.ground_truth_edit = QPlainTextEdit()
        self.ground_truth_edit.setReadOnly(True)
        self.ground_truth_edit.setPlaceholderText(
            "Ground-truth code from dataset (read-only)…"
        )
        self.ground_truth_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        self.reasoning_edit = QPlainTextEdit()
        self.reasoning_edit.setReadOnly(True)
        self.reasoning_edit.setPlaceholderText(
            "Cosmos reasoning test output will appear here…"
        )
        self.reasoning_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)

        self.tabs.addTab(self.predicted_edit, "Predicted")
        self.tabs.addTab(self.ground_truth_edit, "Ground truth")
        self.tabs.addTab(self.reasoning_edit, "Reasoning")
        layout.addWidget(self.tabs)

    @property
    def tab_widget(self) -> QTabWidget:
        """Expose tabs so callers can add extra panels (e.g. run history)."""
        return self.tabs

    def set_predicted(self, code: str) -> None:
        self.predicted_edit.setPlainText(code)
        self.tabs.setCurrentIndex(0)

    def set_ground_truth(self, code: str) -> None:
        self.ground_truth_edit.setPlainText(code)

    def set_reasoning(self, reasoning: str) -> None:
        self.reasoning_edit.setPlainText(reasoning)
        self.tabs.setCurrentWidget(self.reasoning_edit)

    def get_predicted(self) -> str:
        return self.predicted_edit.toPlainText().strip()

    def clear_predicted(self) -> None:
        self.predicted_edit.clear()

    def clear_ground_truth(self) -> None:
        self.ground_truth_edit.clear()

    def clear_reasoning(self) -> None:
        self.reasoning_edit.clear()
