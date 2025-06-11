#!/usr/bin/env python3
# project_state_controller.py
# --------------------------------------------------------------------
#  Qt‑based façade around the pure‑Python persistence layer.
# --------------------------------------------------------------------
from __future__ import annotations

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from hashlib import blake2b
from pathlib import Path
from tempfile import gettempdir
from typing import List, Optional, Union

from GUI.models.StatePersistance import ProjectState, save_state, load_state
from PyQt5.QtCore import QObject, pyqtSignal

from GUI.configuration.configuration import PROJECT_EXT
from GUI.models.ImageDataModel import BaseImageDataModel

# --------------------------------------------------------------------
#  constants & helpers
# --------------------------------------------------------------------
TEMP_DIR = Path(gettempdir()) / "SLT_Temp"
TEMP_DIR.mkdir(exist_ok=True)

AUTOSAVE_BASENAME = f"project_autosave{PROJECT_EXT}"


def _hash_path(path: str, length: int = 8) -> str:
    """Case‑sensitive Blake2 hash of *path* for directory names."""
    return blake2b(path.encode("utf8"), digest_size=length).hexdigest()


def _fingerprint(anchor: Optional[Path]) -> str:
    """
    Return an 8-hex tag that is stable for *anchor* and ‘random’ if anchor is None.
    The same project file or data-backend file always yields the same tag.
    """
    if anchor is None:
        return uuid.uuid4().hex[:8]  # unattached session
    norm = os.path.normcase(str(anchor.resolve()))
    return _hash_path(norm, length=8)


# --------------------------------------------------------------------
#  controller
# --------------------------------------------------------------------
class ProjectStateController(QObject):
    """Thin Qt wrapper that persists and restores :class:`ProjectState`.

    Heavy I/O is delegated to the pure‑Python convenience functions in
    *StatePersistance* and executed on a dedicated thread‑pool so the GUI never
    blocks.  The controller additionally provides a **stable frames directory**
    that is unique per project file.
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

        # ------------- frames‑dir state -----------------------------
        self._frames_dir: Optional[Path] = None  # resolved & cached
        self._frames_tmp: Optional[Path] = None  # unsaved_<ts> folder (may be promoted)

        logging.debug("ProjectStateController initialised (temp = %s)", TEMP_DIR)

    # -----------------------------------------------------------------
    #  save‑path helpers
    # -----------------------------------------------------------------
    def set_current_save_path(self, file_path: str) -> None:
        """Register *file_path* as the definitive project location and promote
        any existing temporary frames."""
        if self._current_save_path == file_path:
            return
        self._current_save_path = file_path
        self._maybe_promote_unsaved_frames()
        logging.info("Current save path set to %s", file_path)

    def get_current_save_path(self) -> Optional[str]:
        return self._current_save_path

    # -----------------------------------------------------------------
    #  AUTOSAVE  (never blocks GUI thread)
    # -----------------------------------------------------------------
    def autosave_project_state(self, state: ProjectState) -> None:
        """Write *state* to a timestamped file under :pydata:`TEMP_DIR`."""
        if not state.clusters:  # nothing worth saving
            logging.debug("Autosave skipped: clusters empty.")
            return

        target = self._rotate_backups(TEMP_DIR / AUTOSAVE_BASENAME)
        fut = self._pool.submit(partial(save_state, state, target, level=1))  # fast zstd
        fut.add_done_callback(lambda f: self.autosave_finished.emit(f.exception() is None))

    # -----------------------------------------------------------------
    #  EXPLICIT SAVE  (Save / Save As…)
    # -----------------------------------------------------------------
    def save_project_state(self, state: ProjectState, file_path: Union[str, Path], *, level: int = 3) -> None:
        """Persist *state* to *file_path* in the background."""
        path = Path(file_path).expanduser()
        fut = self._pool.submit(partial(save_state, state, path, level=level))

        def _done(f):
            exc = f.exception()
            if exc is None:
                self.project_saved.emit(str(path))
                logging.info("Saved project → %s (%.1f kB)", path, path.stat().st_size / 1024)
            else:
                logging.exception("Save failed for %s", path)
                self.save_failed.emit(str(exc))

        fut.add_done_callback(_done)

    # -----------------------------------------------------------------
    #  LOAD
    # -----------------------------------------------------------------
    def load_project_state(self, file_path: Union[str, Path]) -> None:
        """Load file synchronously (fast) and emit :pydata:`project_loaded`."""
        path = Path(file_path).expanduser()
        if not path.exists():
            self.load_failed.emit(f"No project file at {path}")
            return

        self.set_current_save_path(str(path))

        try:
            state = load_state(path)
            self.project_loaded.emit(state)
            logging.info("Loaded project %s (schema v%d)", path, state.schema_version)
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
    #  FRAMES DIRECTORY LOGIC
    # -----------------------------------------------------------------
    def get_frames_dir(self) -> Path:
        """Return the directory where diagnostic frames should be written.

        * If the project has already been saved, create a sibling directory
          called ``<project‑stem>_frames`` next to the *.slt* file.
        * Otherwise create a temporary directory inside the OS temp folder.
        * When the project is first saved, any frames in the temporary
          directory are migrated into the definitive folder.
        """
        if self._frames_dir is not None:
            return self._frames_dir

        if self._current_save_path:
            anchor = Path(self._current_save_path).resolve()
            self._frames_dir = anchor.with_suffix("").with_name(anchor.stem + "_frames")
            self._frames_dir.mkdir(parents=True, exist_ok=True)
            return self._frames_dir

        root = Path(gettempdir()) / "SLT_Frames"
        root.mkdir(exist_ok=True)

        data_anchor = Path(self.model.data_path) if self.model else None
        tag = _fingerprint(data_anchor)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._frames_tmp = root / f"unsaved_{tag}_{ts}"
        self._frames_tmp.mkdir(parents=True, exist_ok=True)
        self._frames_dir = self._frames_tmp
        return self._frames_dir

    # -----------------------------------------------------------------
    #  INTERNAL UTILITIES
    # -----------------------------------------------------------------
    def _maybe_promote_unsaved_frames(self) -> None:
        """Move frames from the temp dir to the definitive sibling dir once
        the project gains a save‑path."""
        if not self._frames_tmp or not self._current_save_path:
            return

        anchor = Path(self._current_save_path).resolve()
        target = anchor.with_suffix("").with_name(anchor.stem + "_frames")
        target.mkdir(parents=True, exist_ok=True)

        for p in self._frames_tmp.glob("frame_*.png"):
            try:
                p.replace(target / p.name)
            except OSError:
                logging.warning("Could not move %s to %s", p, target)

        try:
            self._frames_tmp.rmdir()
        except OSError:
            logging.debug("Temp dir %s not empty after promotion", self._frames_tmp)

        self._frames_dir = target
        self._frames_tmp = None

    # ---------------------------- AUTOSAVE UTIL ----------------------
    @staticmethod
    def _rotate_backups(base: Path, max_keep: int = 10) -> Path:
        """Create a timestamped backup path and prune older ones."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{base.stem}_{ts}{base.suffix}"
        new_path = base.parent / new_name

        # delete oldest backups beyond *max_keep*
        pattern = f"{base.stem}_*{base.suffix}"
        backups = sorted(base.parent.glob(pattern), key=lambda p: p.stat().st_mtime)
        for old in backups[:-max_keep + 1]:
            old.unlink(missing_ok=True)
            logging.debug("Deleted old autosave %s", old.name)
        return new_path

    @staticmethod
    def _autosave_files() -> List[Path]:
        return sorted(
            TEMP_DIR.glob(f"{AUTOSAVE_BASENAME}_*.zst"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
