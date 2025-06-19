from __future__ import annotations

"""Widget representing a single scheduled job."""

from PyQt5.QtCore import pyqtSignal
import json
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
    QPlainTextEdit,
)


class JobItemWidget(QWidget):
    """Display job metadata and allow basic control."""

    request_pause = pyqtSignal(int)
    request_resume = pyqtSignal(int)
    request_cancel = pyqtSignal(int)
    request_remove = pyqtSignal(int)

    def __init__(self, job_id: int, name: str, config: dict) -> None:
        super().__init__()
        self.job_id = job_id
        self.config = config
        self._paused = False

        layout = QVBoxLayout(self)
        self.title = QGroupBox(name)
        inner = QVBoxLayout(self.title)

        self.status = QLabel("queued")
        inner.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        inner.addWidget(self.progress)

        self.meta = QPlainTextEdit(json.dumps(config, indent=2))
        self.meta.setReadOnly(True)
        self.meta.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.meta.setStyleSheet("font-family: monospace")
        self.meta.setMaximumHeight(100)
        inner.addWidget(self.meta)

        btn_row = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self.pause_btn)

        cancel_btn = QPushButton("Stop")
        cancel_btn.clicked.connect(lambda: self.request_cancel.emit(self.job_id))
        btn_row.addWidget(cancel_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(
            lambda: self.request_remove.emit(self.job_id)
        )
        btn_row.addWidget(self.remove_btn)
        inner.addLayout(btn_row)

        layout.addWidget(self.title)

    # ------------------------------------------------------------------
    def _toggle_pause(self) -> None:
        if self._paused:
            self.request_resume.emit(self.job_id)
            self.pause_btn.setText("Pause")
        else:
            self.request_pause.emit(self.job_id)
            self.pause_btn.setText("Resume")
        self._paused = not self._paused

    def update_progress(self, processed: int, total: int) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(processed)
        self.status.setText(f"{processed}/{total} running")

    def set_status(self, text: str) -> None:
        self.status.setText(text)
        if text in {"completed", "failed", "canceled", "paused"}:
            self.pause_btn.setEnabled(False)
        self.remove_btn.setEnabled(text != "running")

