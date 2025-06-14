from __future__ import annotations

"""Wizard dialog for configuring MC banker inference."""

import tempfile
from typing import Optional

import yaml
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)


class _SelectPage(QWizardPage):
    """Choose between loading or creating a configuration."""

    def __init__(self, wizard: "MCBankerWizard") -> None:
        super().__init__(wizard)
        self.setTitle("Configuration Source")
        layout = QVBoxLayout(self)
        self.rb_load = QRadioButton("Load existing YAML configuration")
        self.rb_create = QRadioButton("Create new configuration")
        self.rb_load.setChecked(True)
        layout.addWidget(self.rb_load)
        layout.addWidget(self.rb_create)

    def nextId(self) -> int:  # pragma: no cover - wizard navigation
        if self.rb_load.isChecked():
            return MCBankerWizard.PAGE_LOAD
        return MCBankerWizard.PAGE_CREATE


class _LoadPage(QWizardPage):
    """Page for selecting a YAML configuration file."""

    def __init__(self, wizard: "MCBankerWizard") -> None:
        super().__init__(wizard)
        self.setTitle("Load Configuration")
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.edit = QLineEdit()
        row.addWidget(self.edit, 1)
        btn = wizard._make_browse_button(self.edit, select_dir=False)
        row.addWidget(btn)
        layout.addLayout(row)


class _CreatePage(QWizardPage):
    """Gather required parameters for MC banker inference."""

    def __init__(self, wizard: "MCBankerWizard") -> None:
        super().__init__(wizard)
        self.setTitle("Create Configuration")
        form = QFormLayout(self)

        self.model_dir = QLineEdit()
        form.addRow("Model directory:", wizard._make_browse_row(self.model_dir, True))

        self.model_name = QLineEdit()
        form.addRow("Model name:", self.model_name)

        self.data_dir = QLineEdit()
        form.addRow("Data directory:", wizard._make_browse_row(self.data_dir, True))

        self.train_list = QLineEdit()
        form.addRow("Training list:", wizard._make_browse_row(self.train_list, False))

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 512)
        self.batch_size.setValue(1)
        form.addRow("Batch size:", self.batch_size)

        self.shuffle_buf = QSpinBox()
        self.shuffle_buf.setRange(1, 8192)
        self.shuffle_buf.setValue(256)
        form.addRow("Shuffle buffer:", self.shuffle_buf)

        self.out_channels = QSpinBox()
        self.out_channels.setRange(1, 20)
        self.out_channels.setValue(2)
        form.addRow("Output channels:", self.out_channels)

        self.in_h = QSpinBox(); self.in_h.setRange(1, 4096)
        self.in_w = QSpinBox(); self.in_w.setRange(1, 4096)
        self.in_c = QSpinBox(); self.in_c.setRange(1, 4)
        row = QHBoxLayout()
        row.addWidget(self.in_h)
        row.addWidget(self.in_w)
        row.addWidget(self.in_c)
        form.addRow("Input size (H W C):", row)

        self.mc_iter = QSpinBox()
        self.mc_iter.setRange(1, 100)
        self.mc_iter.setValue(8)
        form.addRow("MC iterations:", self.mc_iter)


class MCBankerWizard(QWizard):
    """Wizard guiding the user to run MC banker inference."""

    PAGE_SELECT = 0
    PAGE_LOAD = 1
    PAGE_CREATE = 2

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build MC Inference HDF5")

        self.select_page = _SelectPage(self)
        self.load_page = _LoadPage(self)
        self.create_page = _CreatePage(self)

        self.setPage(self.PAGE_SELECT, self.select_page)
        self.setPage(self.PAGE_LOAD, self.load_page)
        self.setPage(self.PAGE_CREATE, self.create_page)
        self.setStartId(self.PAGE_SELECT)

    # ------------------------------------------------------------------ helpers
    def _make_browse_row(self, edit: QLineEdit, select_dir: bool) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(self._make_browse_button(edit, select_dir))
        return row

    def _make_browse_button(self, edit: QLineEdit, select_dir: bool):
        button = QFileDialog.getExistingDirectory if select_dir else QFileDialog.getOpenFileName
        btn = _BrowseButton(edit, button, select_dir)
        return btn

    # ------------------------------------------------------------------ API
    def get_config_path(self) -> Optional[str]:
        """Return path to configuration YAML or ``None`` if cancelled."""
        if self.exec_() != QWizard.Accepted:
            return None
        if self.select_page.rb_load.isChecked():
            path = self.load_page.edit.text()
            return path if path else None

        cfg = {
            "MODEL_DIR": self.create_page.model_dir.text(),
            "MODEL_NAME": self.create_page.model_name.text(),
            "DATA_DIR": self.create_page.data_dir.text(),
            "TRAINING_LIST": self.create_page.train_list.text(),
            "BATCH_SIZE": self.create_page.batch_size.value(),
            "SHUFFLE_BUFFER_SIZE": self.create_page.shuffle_buf.value(),
            "OUT_CHANNELS": self.create_page.out_channels.value(),
            "INPUT_SIZE": [
                self.create_page.in_h.value(),
                self.create_page.in_w.value(),
                self.create_page.in_c.value(),
            ],
            "MC_N_ITER": self.create_page.mc_iter.value(),
        }
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        with open(tmp.name, "w") as fh:
            yaml.safe_dump(cfg, fh)
        return tmp.name


class _BrowseButton(QPushButton):
    """Button helper for file/directory selection."""

    def __init__(self, edit: QLineEdit, chooser, select_dir: bool) -> None:
        super().__init__("Browse…")
        self.edit = edit
        self._chooser = chooser
        self._dir = select_dir
        self.clicked.connect(self._choose)

    def _choose(self) -> None:  # pragma: no cover - UI
        if self._dir:
            path = QFileDialog.getExistingDirectory(None, "Select Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(None, "Select File")
        if path:
            self.edit.setText(path)

