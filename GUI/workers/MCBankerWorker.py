from __future__ import annotations

"""Worker to run MC banker inference in the background."""

import logging
from typing import Optional

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from DeepLearning.inference.mc_banker_gui import run_from_file


class MCBankerSignals(QObject):
    """Signals used by :class:`MCBankerWorker`."""

    finished = pyqtSignal(bool)


class MCBankerWorker(QRunnable):
    """Background task executing MC banker inference."""

    def __init__(self, config_path: str, *, output_dir: Optional[str] = None, resume: bool = False) -> None:
        super().__init__()
        self.config_path = config_path
        self.output_dir = output_dir
        self.resume = resume
        self.signals = MCBankerSignals()

    def run(self) -> None:  # pragma: no cover - background execution
        try:
            run_from_file(self.config_path, output_dir=self.output_dir, resume=self.resume)
            self.signals.finished.emit(True)
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("MC banker worker failed: %s", exc)
            self.signals.finished.emit(False)
