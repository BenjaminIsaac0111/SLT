from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional, Union

from PyQt5.QtCore import QObject, pyqtSignal

from GUI.models.io.Persistence import ProjectState
from GUI.models.io.Repository import StateRepository, AUTOSAVE_BASENAME
from GUI.models.io.Utils import TEMP_DIR, autosave_files, fingerprint

__all__ = ["ProjectIOService"]


class ProjectIOService(QObject):
    """Background I/O facilitator between the GUI and the repository."""

    # Qt signals ---------------------------------------------------------
    autosave_finished = pyqtSignal(bool)
    project_loaded = pyqtSignal(ProjectState)
    project_saved = pyqtSignal(str)
    save_failed = pyqtSignal(str)
    load_failed = pyqtSignal(str)

    # ------------------------------------------------------------------
    def __init__(self, *, data_anchor: Optional[Path] = None):
        super().__init__()
        self.repo = StateRepository()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="io")

        self._current_path: Optional[Path] = None
        self._tag = fingerprint(data_anchor)

    # ------------------------------------------------------------------
    #  Public path helpers
    # ------------------------------------------------------------------
    @property
    def current_path(self) -> Optional[Path]:
        return self._current_path

    def set_current_path(self, path: Union[str, Path]) -> None:
        path = Path(path).expanduser().resolve()
        if path == self._current_path:
            return
        self._current_path = path

    # ------------------------------------------------------------------
    #  Asynchronous operations
    # ------------------------------------------------------------------
    def save_async(self, state: ProjectState, path: Union[str, Path], *, level: int = 3):
        path = Path(path).expanduser()
        fut = self._pool.submit(partial(self.repo.save, state, path, level=level))
        fut.add_done_callback(lambda f: self._emit_save_result(f, path))

    def autosave_async(self, state: ProjectState):
        fut = self._pool.submit(self.repo.autosave, state)
        fut.add_done_callback(lambda f: self.autosave_finished.emit(f.result() is not None))

    def load_async(self, path: Union[str, Path]):
        path = Path(path).expanduser()
        if not path.exists():
            self.load_failed.emit(f"No project file at {path}")
            return
        self.set_current_path(path)

        fut = self._pool.submit(self.repo.load, path)
        fut.add_done_callback(self._emit_load_result)

    # ------------------------------------------------------------------
    #  Autosave listing
    # ------------------------------------------------------------------
    @staticmethod
    def latest_autosave() -> Optional[Path]:
        files = autosave_files(TEMP_DIR, AUTOSAVE_BASENAME)
        return files[0] if files else None

    @staticmethod
    def autosave_list():
        return autosave_files(TEMP_DIR, AUTOSAVE_BASENAME)

    # ------------------------------------------------------------------
    #  Shutdown
    # ------------------------------------------------------------------
    def shutdown(self):
        self._pool.shutdown(wait=True)

    # ------------------------------------------------------------------
    #  Internal callbacks
    # ------------------------------------------------------------------
    def _emit_save_result(self, fut, path: Path):
        if fut.exception() is None:
            self.project_saved.emit(str(path))
        else:
            self.save_failed.emit(str(fut.exception()))

    def _emit_load_result(self, fut):
        exc = fut.exception()
        if exc is None:
            self.project_loaded.emit(fut.result())
        else:
            self.load_failed.emit(str(exc))
