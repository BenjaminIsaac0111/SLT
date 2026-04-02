from __future__ import annotations

"""Widget representing a single scheduled job with controls."""

from typing import Any, Dict

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
)


class JobItemWidget(QGroupBox):
    """Display progress and controls for a scheduled job."""

    request_pause = pyqtSignal(int)
    request_resume = pyqtSignal(int)
    request_cancel = pyqtSignal(int)
    request_delete = pyqtSignal(int)

    def __init__(self, job_id: int, name: str, config: Dict[str, Any]):
        super().__init__(name)
        self.job_id = job_id
        self.config = config
        self._paused = False
        layout = QVBoxLayout(self)

        self.meta = QLabel(str(config))
        layout.addWidget(self.meta)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)

        btn_row = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self.pause_btn)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(lambda: self.request_delete.emit(self.job_id))
        btn_row.addWidget(del_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    def _toggle_pause(self) -> None:
        if self._paused:
            self.request_resume.emit(self.job_id)
            self.pause_btn.setText("Pause")
        else:
            self.request_pause.emit(self.job_id)
            self.pause_btn.setText("Resume")
        self._paused = not self._paused

    # ------------------------------------------------------------------
    def update_status(self, status: str, start: str = "", end: str = "") -> None:
        """Update displayed metadata."""
        text = f"{status}"
        if start:
            text += f"\nStarted: {start}"
        if end:
            text += f"\nFinished: {end}"
        self.meta.setText(text)

    # ------------------------------------------------------------------
    def update_progress(self, processed: int, total: int) -> None:
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(processed)
        else:
            self.bar.setRange(0, 0)


