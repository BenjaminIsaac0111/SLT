from __future__ import annotations

"""Wizard dialog for configuring MC banker inference."""

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import yaml
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QComboBox,
    QLabel,
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
    """Gather parameters for MC banker inference."""

    def __init__(self, wizard: "MCBankerWizard") -> None:
        super().__init__(wizard)
        self.setTitle("Create Configuration")
        form = QFormLayout(self)

        self.model_path = QLineEdit()
        self.model_select = wizard._make_model_combo(self.model_path)
        row = QHBoxLayout()
        row.addWidget(self.model_path, 1)
        row.addWidget(self.model_select)
        row.addWidget(wizard._make_browse_button(self.model_path, False))
        form.addRow("Model file:", row)
        self.model_path.textChanged.connect(wizard._update_model_info)

        self.data_dir = QLineEdit()
        form.addRow("Data directory:", wizard._make_browse_row(self.data_dir, True))

        self.file_list = QLineEdit()
        form.addRow("Data list:", wizard._make_browse_row(self.file_list, False))

        self.in_size = QLabel("?")
        form.addRow("Input size:", self.in_size)

        self.out_channels = QLabel("?")
        form.addRow("Output channels:", self.out_channels)

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

    def _make_model_combo(self, edit: QLineEdit):
        from GUI.models.MCConfigDB import get_recent_model_paths

        combo = QComboBox()
        combo.setEditable(False)
        combo.addItems(get_recent_model_paths())
        combo.currentTextChanged.connect(edit.setText)
        return combo

    def _make_browse_button(self, edit: QLineEdit, select_dir: bool):
        button = QFileDialog.getExistingDirectory if select_dir else QFileDialog.getOpenFileName
        btn = _BrowseButton(edit, button, select_dir)
        return btn

    def _update_model_info(self, path: str) -> None:  # pragma: no cover - UI
        if not path:
            self.create_page.in_size.setText("?")
            self.create_page.out_channels.setText("?")
            return
        try:
            input_size, outc = self._infer_model_spec(path)
        except Exception:
            input_size, outc = ("?", "?")
        if input_size == "?":
            self.create_page.in_size.setText("?")
            self.create_page.out_channels.setText("?")
        else:
            self.create_page.in_size.setText("x".join(str(x) for x in input_size))
            self.create_page.out_channels.setText(str(outc))

    @staticmethod
    def _infer_model_spec(path: str) -> Tuple[list[int], int]:
        from tensorflow.keras.models import load_model
        from DeepLearning.models.custom_layers import (
            DropoutAttentionBlock,
            GroupNormalization,
            SpatialConcreteDropout,
        )

        model = load_model(
            path,
            custom_objects={
                "DropoutAttentionBlock": DropoutAttentionBlock,
                "GroupNormalization": GroupNormalization,
                "SpatialConcreteDropout": SpatialConcreteDropout,
            },
            compile=False,
        )
        input_shape = list(model.input_shape[1:])
        out_channels = int(model.layers[-1].output_shape[-1])
        return input_shape, out_channels

    # ------------------------------------------------------------------ API
    def get_config_path(self) -> Optional[str]:
        """Return path to configuration YAML or ``None`` if cancelled."""
        if self.exec_() != QWizard.Accepted:
            return None
        if self.select_page.rb_load.isChecked():
            path = self.load_page.edit.text()
            return path if path else None

        model_path = self.create_page.model_path.text()
        if not model_path:
            return None
        outc_txt = self.create_page.out_channels.text()
        in_size_txt = self.create_page.in_size.text()
        if "?" in outc_txt or "?" in in_size_txt:
            try:
                in_size, outc = self._infer_model_spec(model_path)
            except Exception:
                return None
        else:
            outc = int(outc_txt)
            in_size = [int(v) for v in in_size_txt.split("x")]

        cfg = {
            "MODEL_DIR": str(Path(model_path).parent),
            "MODEL_NAME": Path(model_path).name,
            "DATA_DIR": self.create_page.data_dir.text(),
            "FILE_LIST": self.create_page.file_list.text(),
            "BATCH_SIZE": 1,
            "SHUFFLE_BUFFER_SIZE": 256,
            "OUT_CHANNELS": outc,
            "INPUT_SIZE": in_size,
            "MC_N_ITER": self.create_page.mc_iter.value(),
        }
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        with open(tmp.name, "w") as fh:
            yaml.safe_dump(cfg, fh)
        from GUI.models.MCConfigDB import save_config

        save_config(cfg)
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

