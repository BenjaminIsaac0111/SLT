from __future__ import annotations

"""Widget showing progress of the MC banker worker with pause/cancel controls."""

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QHBoxLayout,
)


class MCBankerProgressWidget(QGroupBox):
    """Progress view for MC banker jobs."""

    request_pause = pyqtSignal()
    request_resume = pyqtSignal()
    request_cancel = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("MC Banker Progress", parent)
        self._paused = False
        self._total = 0
        layout = QVBoxLayout(self)

        self.label = QLabel("Idle")
        layout.addWidget(self.label)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)

        btn_row = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self.pause_btn)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.request_cancel.emit)
        btn_row.addWidget(stop_btn)
        layout.addLayout(btn_row)

    def _toggle_pause(self):
        if self._paused:
            self.request_resume.emit()
            self.pause_btn.setText("Pause")
        else:
            self.request_pause.emit()
            self.pause_btn.setText("Resume")
        self._paused = not self._paused

    def start(self, output_file: str, total: int) -> None:
        self._paused = False
        self.pause_btn.setText("Pause")
        self.pause_btn.setEnabled(True)
        self._total = total
        self.bar.setRange(0, total)
        self.bar.setValue(0)
        self.label.setText(Path(output_file).name)

    def update_progress(self, processed: int, total: int) -> None:
        if total != self._total:
            self.bar.setRange(0, total)
            self._total = total
        self.bar.setValue(processed)
        self.label.setText(f"{processed}/{total} samples")

    def finish(self):
        self._paused = False
        self.pause_btn.setText("Pause")
        self.pause_btn.setEnabled(False)
        self.label.setText("Done")

