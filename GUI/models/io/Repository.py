from __future__ import annotations

"""project_io.repository
=======================

**Pure‑Python** persistence façade: provides a *minimal* synchronous API
around :pyfunc:`GUI.models.StatePersistance.save_state` / ``load_state`` plus a
convenience helper for timestamped autosaves.

Rationale
~~~~~~~~~
* Decouple *business logic* from any threading / Qt concerns.
* Centralise the snapshot‑rotation policy so tests do not need to monkey‑patch
  GUI classes to verify retention behaviour.
* Synchronous methods make unit tests straightforward – call → assert file
  exists – without dealing with futures.

Public interface
----------------
``save(state, path, level=3)``
    Persist *state* to *path*.  Compression level defaults to “3” (same as the
    old controller).

``load(path) -> ProjectState``
    Load project file synchronously.

``autosave(state) -> Optional[Path]``
    Write a timestamped snapshot into :pydata:`project_io.utils.TEMP_DIR` and
    return the *Path*.  Returns *None* when there is nothing worth saving
    (i.e. *state.clusters* is empty).

A *single* constructor parameter, ``max_keep``, governs how many autosave files
are retained per project.
"""

from pathlib import Path
from typing import Optional
import logging

from GUI.models.io.Persistence import ProjectState, save_state, load_state
from GUI.models.io.Utils import TEMP_DIR, rotate_backups

__all__ = ["StateRepository"]

# -----------------------------------------------------------------------------
#  Constants
# -----------------------------------------------------------------------------

AUTOSAVE_BASENAME = "project_autosave.slt"


# -----------------------------------------------------------------------------
#  Implementation
# -----------------------------------------------------------------------------

class StateRepository:
    """Synchronous persistence helper (no Qt, no threads)."""

    def __init__(self, *, max_keep: int = 10):
        self.max_keep = max_keep

    # .................................................................
    #  Public API
    # .................................................................

    @staticmethod
    def save(state: ProjectState, path: Path | str, *, level: int = 3) -> None:
        """Write *state* to *path* with Zstandard compression ``level``."""
        path = Path(path).expanduser()
        save_state(state, path, level=level)

        logging.debug("Saved state to %s (%.1f kB)", path, path.stat().st_size / 1024)

    @staticmethod
    def load(path: Path | str) -> ProjectState:  # noqa: D401
        """Load project file synchronously and return the :class:`ProjectState`."""
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        state = load_state(path)
        logging.debug("Loaded project %s (schema v%d)", path, state.schema_version)
        return state

    def autosave(self, state: ProjectState) -> Optional[Path]:  # noqa: D401
        """Write a *quick* snapshot and prune old autosaves.

        Returns
        -------
        Optional[Path]
            Path to the new autosave file, or *None* if nothing was saved.
        """
        if not state.clusters:
            logging.debug("Autosave skipped – no clusters.")
            return None

        target = rotate_backups(TEMP_DIR / AUTOSAVE_BASENAME, max_keep=self.max_keep)
        save_state(state, target, level=1)  # fast compression
        logging.info("Autosaved project → %s", target.name)
        return target
