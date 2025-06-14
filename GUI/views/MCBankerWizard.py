from __future__ import annotations

"""Wizard dialog for configuring MC banker inference."""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QComboBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)




class _CreatePage(QWizardPage):
    """Gather parameters for MC banker inference."""

    def __init__(self, wizard: "MCBankerWizard") -> None:
        super().__init__(wizard)
        self.setTitle("Create Configuration")
        form = QFormLayout(self)

        self.model_path = wizard._make_model_combo()
        self.model_path.setEditable(True)
        row = QHBoxLayout()
        row.addWidget(self.model_path, 1)
        row.addWidget(
            wizard._make_browse_button(
                self.model_path,
                select_dir=False,
                file_filter=
                "TensorFlow models (*.h5 *.hdf5 *.keras *.pb *.ckpt *.index);;All files (*)",
            )
        )
        form.addRow("Model file:", row)
        self.model_path.currentTextChanged.connect(wizard._update_model_info)
        self.model_path.editTextChanged.connect(wizard._update_model_info)

        self.data_dir = QLineEdit()
        form.addRow(
            "Data directory:",
            wizard._make_browse_row(
                self.data_dir,
                True,
                file_filter="Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            ),
        )

        self.file_list = QLineEdit()
        form.addRow(
            "Data list:",
            wizard._make_browse_row(
                self.file_list,
                False,
                file_filter=
                "List files (*.txt *.csv);;Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
            ),
        )

        self.output_file = QLineEdit()
        form.addRow(
            "Output file:",
            wizard._make_browse_row(
                self.output_file,
                False,
                file_filter="HDF5 files (*.h5 *.hdf5);;All files (*)",
                save_file=True,
            ),
        )

        self.in_size = QLabel("?")
        form.addRow("Input size:", self.in_size)

        self.out_channels = QLabel("?")
        form.addRow("Output channels:", self.out_channels)

        self.unc_type = QComboBox()
        self.unc_type.addItems(["BALD", "variance", "entropy"])
        form.addRow("Uncertainty:", self.unc_type)

        self.mc_iter = QSpinBox()
        self.mc_iter.setRange(1, 100)
        self.mc_iter.setValue(8)
        form.addRow("MC iterations:", self.mc_iter)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.1, 10.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(1.0)
        form.addRow("Temperature:", self.temperature)

        self.unc_type.currentTextChanged.connect(
            lambda text: self.mc_iter.setEnabled(text.lower() != "entropy")
        )


class MCBankerWizard(QWizard):
    """Wizard guiding the user to run MC banker inference."""

    PAGE_CREATE = 0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Build MC Inference HDF5")

        self.create_page = _CreatePage(self)

        self.setPage(self.PAGE_CREATE, self.create_page)
        self.setStartId(self.PAGE_CREATE)

    # ------------------------------------------------------------------ helpers
    def _make_browse_row(
        self,
        widget,
        select_dir: bool,
        *,
        file_filter: str = "",
        save_file: bool = False,
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(widget, 1)
        row.addWidget(
            self._make_browse_button(
                widget,
                select_dir=select_dir,
                file_filter=file_filter,
                save_file=save_file,
            )
        )
        return row

    def _make_model_combo(self) -> QComboBox:
        """Return editable combo box populated with recent models."""
        from GUI.models.MCConfigDB import get_recent_model_paths

        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(get_recent_model_paths())
        return combo

    def _make_browse_button(
        self,
        widget,
        *,
        select_dir: bool = False,
        file_filter: str = "",
        save_file: bool = False,
    ):
        return _BrowseButton(
            widget,
            select_dir=select_dir,
            file_filter=file_filter,
            save_file=save_file,
        )

    def _update_model_info(self, path: str) -> None:  # pragma: no cover - UI
        if not path:
            self.create_page.in_size.setText("?")
            self.create_page.out_channels.setText("?")
            return

        from PyQt5.QtWidgets import QApplication, QProgressDialog
        from PyQt5.QtCore import Qt

        dlg = QProgressDialog("Loading model…", None, 0, 0, self)
        dlg.setWindowTitle("Please Wait")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.show()
        QApplication.processEvents()

        try:
            input_size, outc = self._infer_model_spec(path)
        except Exception:
            input_size, outc = ("?", "?")
        finally:
            dlg.close()

        if input_size == "?":
            self.create_page.in_size.setText("?")
            self.create_page.out_channels.setText("?")
        else:
            self.create_page.in_size.setText("x".join(str(x) for x in input_size))
            self.create_page.out_channels.setText(str(outc))

    @staticmethod
    def _infer_model_spec(path: str) -> Tuple[list[int], int]:
        from tensorflow.keras.models import load_model
        from tensorflow.keras import mixed_precision
        from DeepLearning.models.custom_layers import (
            DropoutAttentionBlock,
            GroupNormalization,
            SpatialConcreteDropout,
        )

        mixed_precision.set_global_policy("mixed_float16")

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
    def get_config(self) -> Optional[Dict[str, Any]]:
        """Return configuration dict or ``None`` if cancelled."""
        if self.exec_() != QWizard.Accepted:
            return None

        model_path = self.create_page.model_path.currentText()
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
            "OUTPUT_FILE": self.create_page.output_file.text(),
            "BATCH_SIZE": 1,
            "SHUFFLE_BUFFER_SIZE": 256,
            "OUT_CHANNELS": outc,
            "INPUT_SIZE": in_size,
            "UNCERTAINTY_TYPE": self.create_page.unc_type.currentText().lower(),
            "MC_N_ITER": (
                1
                if self.create_page.unc_type.currentText().lower() == "entropy"
                else self.create_page.mc_iter.value()
            ),
            "TEMPERATURE": float(self.create_page.temperature.value()),
        }
        from GUI.models.MCConfigDB import save_config

        save_config(cfg)
        return cfg


class _BrowseButton(QPushButton):
    """Button helper for file/directory selection."""

    def __init__(
        self,
        widget,
        *,
        select_dir: bool = False,
        file_filter: str = "",
        save_file: bool = False,
    ) -> None:
        super().__init__("Browse…")
        self.widget = widget
        self._dir = select_dir
        self._filter = file_filter
        self._save = save_file
        self.clicked.connect(self._choose)

    def _choose(self) -> None:  # pragma: no cover - UI
        if self._dir:
            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.ShowDirsOnly, False)
            if self._filter:
                dlg.setNameFilter(self._filter)
            if not dlg.exec_():
                return
            path = dlg.selectedFiles()[0]
        elif self._save:
            path, _ = QFileDialog.getSaveFileName(
                None, "Select File", "", self._filter
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                None, "Select File", "", self._filter
            )
        if path:
            if hasattr(self.widget, "setText"):
                self.widget.setText(path)
            elif hasattr(self.widget, "setEditText"):
                self.widget.setEditText(path)

