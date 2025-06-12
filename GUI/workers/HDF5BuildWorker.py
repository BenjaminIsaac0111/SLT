"""Worker for building training HDF5 datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from GUI.models.hdf5_builder import build_training_hdf5

__all__ = ["HDF5BuildWorker", "HDF5BuildWorkerSignals"]


class HDF5BuildWorkerSignals(QObject):
    """Signals emitted by :class:`HDF5BuildWorker`."""

    finished = pyqtSignal()
    error = pyqtSignal(str)


class HDF5BuildWorker(QRunnable):
    """QRunnable wrapper around :func:`build_training_hdf5`."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        training_csv: Union[str, Path],
        model_path: Union[str, Path],
        output_file: Union[str, Path],
        *,
        sample_size: Optional[int] = None,
        mc_iter: int = 1,
    ) -> None:
        super().__init__()
        self.signals = HDF5BuildWorkerSignals()
        self.data_dir = Path(data_dir)
        self.training_csv = Path(training_csv)
        self.model_path = Path(model_path)
        self.output_file = Path(output_file)
        self.sample_size = sample_size
        self.mc_iter = mc_iter

    def run(self) -> None:
        try:
            build_training_hdf5(
                self.data_dir,
                self.training_csv,
                self.model_path,
                self.output_file,
                sample_size=self.sample_size,
                mc_iter=self.mc_iter,
            )
            self.signals.finished.emit()
        except Exception as err:  # pragma: no cover - rarely triggered
            logging.exception("HDF5 build failed")
            self.signals.error.emit(str(err))
