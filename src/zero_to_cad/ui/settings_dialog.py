"""Settings dialog for API keys and endpoints (stored in settings.json)."""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from zero_to_cad.settings import SETTINGS_FIELDS, load_settings, save_settings, settings_path


class SettingsDialog(QDialog):
    """Edit API keys / endpoints and persist them to settings.json."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        info = QLabel(
            "API keys and endpoints are stored in "
            f"<code>{settings_path()}</code> (gitignored). Environment variables, "
            "if set, take precedence over these values."
        )
        info.setWordWrap(True)
        info.setTextFormat(info.textFormat().RichText)
        layout.addWidget(info)

        form = QFormLayout()
        self._edits: dict[str, QLineEdit] = {}
        current = load_settings()
        for key, label, is_secret in SETTINGS_FIELDS:
            edit = QLineEdit(current.get(key, ""))
            if is_secret:
                edit.setEchoMode(QLineEdit.EchoMode.Password)
            if os.environ.get(key) and not current.get(key):
                edit.setPlaceholderText(f"{key} (currently set via environment)")
            else:
                edit.setPlaceholderText(key)
            self._edits[key] = edit
            form.addRow(label, edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_save(self) -> None:
        values = {key: edit.text().strip() for key, edit in self._edits.items()}
        save_settings(values)
        # Apply non-empty values immediately so they take effect this session
        # (cleared fields are dropped from the file and won't reload next launch).
        for key, value in values.items():
            if value:
                os.environ[key] = value
        self.accept()
