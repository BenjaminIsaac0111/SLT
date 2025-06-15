from __future__ import annotations

"""Worker to run MC banker inference in the background."""

import logging
from typing import Optional, Dict, Any

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from DeepLearning.inference.mc_banker_gui import run_config


class MCBankerSignals(QObject):
    """Signals used by :class:`MCBankerWorker`."""

    finished = pyqtSignal(bool)
    progress = pyqtSignal(int, int)


class MCBankerWorker(QRunnable):
    """Background task executing MC banker inference."""

    def __init__(self, config: Dict[str, Any], *, output_dir: Optional[str] = None, resume: bool = False) -> None:
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.resume = resume
        self.signals = MCBankerSignals()
        self._abort = False
        self._pause = False

    # ------------------------- control ----------------------------
    def pause(self) -> None:  # pragma: no cover - invoked from GUI
        self._pause = True

    def resume_task(self) -> None:  # pragma: no cover - invoked from GUI
        self._pause = False

    def cancel(self) -> None:  # pragma: no cover - invoked from GUI
        self._abort = True

    def run(self) -> None:  # pragma: no cover - background execution
        try:
            run_config(
                self.config,
                output_dir=self.output_dir,
                resume=self.resume,
                progress_cb=self.signals.progress.emit,
                should_abort=lambda: self._abort,
                should_pause=lambda: self._pause,
            )
            self.signals.finished.emit(True)
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("MC banker worker failed: %s", exc)
            self.signals.finished.emit(False)
