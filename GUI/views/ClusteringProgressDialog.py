from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton
)


class ClusteringProgressDialog(QDialog):
    """
    A small modal window that tracks “Generate Annotations” and
    lets the user abort the job.
    """
    cancel_requested = pyqtSignal()  # emitted when the Cancel button is pressed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generating Annotations…")
        self.setModal(True)
        self.setFixedWidth(360)

        lay = QVBoxLayout(self)

        self.phase_lbl = QLabel("Preparing…")
        lay.addWidget(self.phase_lbl)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        lay.addWidget(self.bar)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancel_requested.emit)
        lay.addWidget(cancel_btn)

    # ------------------------------------------------------------------ API
    def update_phase(self, phase: str, value: int):
        """
        phase ∈ {"Clustering", "Extracting"} ;  value ∈ [0,100] or -1 for indeterminate
        """
        self.phase_lbl.setText(f"{phase} annotations…")
        if value == -1:
            self.bar.setRange(0, 0)  # Qt «busy» bar
        else:
            if self.bar.maximum() == 0:
                self.bar.setRange(0, 100)
            self.bar.setValue(value)
