from __future__ import annotations

"""Worker that generates stratified group cross-validation folds."""

import logging
from pathlib import Path

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from GUI.models.cross_validation import write_cv_folds


class CrossValidationSignals(QObject):
    """Signals for :class:`CrossValidationWorker`."""

    finished = pyqtSignal(str)


class CrossValidationWorker(QRunnable):
    """Background task that writes cross-validation folds to disk."""

    def __init__(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        *,
        n_splits: int = 3,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.signals = CrossValidationSignals()

    def run(self) -> None:
        """Execute the cross-validation fold creation."""
        try:
            write_cv_folds(
                self.image_dir,
                self.output_dir,
                n_splits=self.n_splits,
                shuffle=self.shuffle,
            )
            self.signals.finished.emit(str(self.output_dir))
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("Cross-validation worker failed: %s", exc)
            self.signals.finished.emit("")

