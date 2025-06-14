from __future__ import annotations

"""Dialog for configuring cross-validation fold creation."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class CrossValidationDialog(QDialog):
    """Gather parameters for cross-validation fold creation."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build Cross-Validation Folds")

        layout = QVBoxLayout(self)
        layout.addLayout(self._directory_row("Image Directory", True))
        layout.addLayout(self._directory_row("Output Directory", False))

        fold_row = QHBoxLayout()
        fold_row.addWidget(QLabel("Number of folds:"))
        self.sp_folds = QSpinBox()
        self.sp_folds.setRange(2, 20)
        self.sp_folds.setValue(3)
        fold_row.addWidget(self.sp_folds)
        fold_row.addStretch(1)
        layout.addLayout(fold_row)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            orientation=Qt.Horizontal,
            parent=self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ------------------------------------------------------------------ helpers
    def _directory_row(self, label: str, input_dir: bool) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{label}:"))
        edit = QLineEdit()
        button = QPushButton("Browse…")
        if input_dir:
            self.img_edit = edit
            button.clicked.connect(self._choose_img)
        else:
            self.out_edit = edit
            button.clicked.connect(self._choose_out)
        row.addWidget(edit, 1)
        row.addWidget(button)
        return row

    def _choose_img(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if path:
            self.img_edit.setText(path)

    def _choose_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.out_edit.setText(path)

    # ------------------------------------------------------------------ public
    def get_config(self) -> tuple[str, str, int] | None:
        """Return user selections or ``None`` if cancelled."""
        if self.exec_() != QDialog.Accepted:
            return None
        return (
            self.img_edit.text(),
            self.out_edit.text(),
            self.sp_folds.value(),
        )
