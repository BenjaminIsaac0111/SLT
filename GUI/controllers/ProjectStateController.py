#!/usr/bin/env python3
# project_state_controller.py
# --------------------------------------------------------------------
#  Qt‑based façade around the pure‑Python persistence layer.
# --------------------------------------------------------------------
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import List, Optional, Union

from PyQt5.QtCore import QObject, pyqtSignal

from GUI.configuration.configuration import PROJECT_EXT
from GUI.models.ImageDataModel import BaseImageDataModel
from GUI.models.StatePersistance import ProjectState, save_state, load_state

# --------------------------------------------------------------------
#  constants
# --------------------------------------------------------------------
TEMP_DIR = Path(gettempdir()) / "SLT_Temp"
TEMP_DIR.mkdir(exist_ok=True)

AUTOSAVE_BASENAME = f"project_autosave{PROJECT_EXT}"


# --------------------------------------------------------------------
#  controller
# --------------------------------------------------------------------
class ProjectStateController(QObject):
    """
    Thin Qt wrapper that persists and restores ProjectState objects.
    All heavy I/O is delegated to persistence.py and executed on a
    single‑thread ThreadPoolExecutor so the GUI never blocks.
    """

    # --------------------- Qt signals --------------------------------
    autosave_finished = pyqtSignal(bool)
    project_loaded = pyqtSignal(ProjectState)
    project_saved = pyqtSignal(str)
    save_failed = pyqtSignal(str)
    load_failed = pyqtSignal(str)

    # --------------------- ctor --------------------------------------
    def __init__(self, model: BaseImageDataModel | None):
        super().__init__()
        self.model = model
        self._current_save_path: Optional[str] = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="save")
        logging.debug("ProjectStateController initialised (temp = %s)", TEMP_DIR)

    # -----------------------------------------------------------------
    #  save‑path helpers
    # -----------------------------------------------------------------
    def set_current_save_path(self, file_path: str) -> None:
        self._current_save_path = file_path
        logging.info("Current save path set to %s", file_path)

    def get_current_save_path(self) -> Optional[str]:
        return self._current_save_path

    # -----------------------------------------------------------------
    #  AUTOSAVE  (never blocks GUI thread)
    # -----------------------------------------------------------------
    def autosave_project_state(self, state: ProjectState) -> None:
        """Write *state* to a timestamped file under TEMP_DIR."""
        if not state.clusters:  # nothing worth saving
            logging.debug("Autosave skipped: clusters empty.")
            return

        target = self._rotate_backups(TEMP_DIR / AUTOSAVE_BASENAME)
        fut = self._pool.submit(
            partial(save_state, state, target, level=1)  # fast zstd
        )
        fut.add_done_callback(
            lambda f: self.autosave_finished.emit(f.exception() is None)
        )

    # -----------------------------------------------------------------
    #  EXPLICIT SAVE  (Save / Save As…)
    # -----------------------------------------------------------------
    def save_project_state(
            self,
            state: ProjectState,
            file_path: Union[str, Path],
            *,
            level: int = 3,
    ) -> None:
        """Persist *state* to *file_path* in the background."""
        path = Path(file_path).expanduser()
        fut = self._pool.submit(
            partial(save_state, state, path, level=level)
        )

        def _done(f):
            exc = f.exception()
            if exc is None:
                self.project_saved.emit(str(path))
                logging.info("Saved project → %s (%.1f kB)",
                             path, path.stat().st_size / 1024)
            else:
                logging.exception("Save failed for %s", path)
                self.save_failed.emit(str(exc))

        fut.add_done_callback(_done)

    # -----------------------------------------------------------------
    #  LOAD
    # -----------------------------------------------------------------
    def load_project_state(self, file_path: Union[str, Path]) -> None:
        """Load file synchronously (fast) and emit signal."""
        path = Path(file_path).expanduser()
        if not path.exists():
            self.load_failed.emit(f"No project file at {path}")
            return
        try:
            state = load_state(path)
            self.project_loaded.emit(state)
            logging.info("Loaded project %s (schema v%d)",
                         path, state.schema_version)
        except Exception as err:
            logging.exception("Failed to load %s", path)
            self.load_failed.emit(str(err))

    # -----------------------------------------------------------------
    #  QUERY AUTOSAVE FILES
    # -----------------------------------------------------------------
    def get_latest_autosave_file(self) -> Optional[str]:
        files = self._autosave_files()
        return str(files[0]) if files else None

    def get_autosave_files(self) -> List[str]:
        return [str(p) for p in self._autosave_files()]

    # -----------------------------------------------------------------
    #  CLEAN‑UP
    # -----------------------------------------------------------------
    def cleanup(self) -> None:
        """Wait for background tasks before Qt exits."""
        logging.info("Waiting for save tasks to finish…")
        self._pool.shutdown(wait=True)
        logging.info("ProjectStateController cleanup done.")

    # -----------------------------------------------------------------
    #  INTERNAL UTILITIES
    # -----------------------------------------------------------------
    @staticmethod
    def _rotate_backups(base: Path, max_keep: int = 10) -> Path:
        """
        Return a new file name  <base.stem>_<timestamp><base.suffix>
        and delete oldest backups so at most *max_keep* remain.
        Works on Python 3.7+ (no Path.with_stem).
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{base.stem}_{ts}{base.suffix}"
        new_path = base.parent / new_name

        # clean up old backups
        pattern = f"{base.stem}_*{base.suffix}"
        backups = sorted(base.parent.glob(pattern), key=lambda p: p.stat().st_mtime)
        for old in backups[:-max_keep + 1]:
            old.unlink(missing_ok=True)
            logging.debug("Deleted old autosave %s", old.name)

        return new_path

    def _autosave_files(self) -> List[Path]:
        return sorted(
            TEMP_DIR.glob(f"{AUTOSAVE_BASENAME}_*.zst"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
