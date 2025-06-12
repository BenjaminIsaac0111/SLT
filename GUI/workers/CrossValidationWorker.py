"""Worker for creating grouped cross-validation folds asynchronously."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from GUI.models.cv_folds import create_grouped_folds

__all__ = ["CrossValidationWorker", "CrossValidationWorkerSignals"]


class CrossValidationWorkerSignals(QObject):
    """Signals emitted by :class:`CrossValidationWorker`."""

    finished = pyqtSignal()
    error = pyqtSignal(str)


class CrossValidationWorker(QRunnable):
    """QRunnable wrapper around :func:`create_grouped_folds`."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        *,
        n_splits: int = 3,
        sample_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.signals = CrossValidationWorkerSignals()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.n_splits = n_splits
        self.sample_size = sample_size

    def run(self) -> None:
        try:
            create_grouped_folds(
                self.data_dir,
                self.output_dir,
                n_splits=self.n_splits,
                sample_size=self.sample_size,
            )
            self.signals.finished.emit()
        except Exception as err:  # pragma: no cover - rarely triggered
            logging.exception("Cross validation fold generation failed")
            self.signals.error.emit(str(err))
