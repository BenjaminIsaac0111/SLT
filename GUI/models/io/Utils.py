from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from hashlib import blake2b
from pathlib import Path
from tempfile import gettempdir
from typing import List, Optional

__all__ = [
    "TEMP_DIR",
    "hash_path",
    "fingerprint",
    "rotate_backups",
    "autosave_files",
]

# -----------------------------------------------------------------------------
#  Constants
# -----------------------------------------------------------------------------

TEMP_DIR: Path = Path(gettempdir()) / "SLT_Temp"
TEMP_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
#  Low‑level helpers
# -----------------------------------------------------------------------------

def hash_path(path: str, *, length: int = 8) -> str:  # noqa: D401 – imperative mood
    """Return a *case‑sensitive* Blake‑2 hash of *path*.

    Useful for generating opaque directory names that remain stable across
    sessions but reveal nothing about the original path.
    """
    return blake2b(path.encode("utf8"), digest_size=length).hexdigest()


def fingerprint(anchor: Optional[Path]) -> str:
    """Return an 8‑char tag unique to the *anchor* file or directory.

    Parameters
    ----------
    anchor
        A file path that identifies the dataset/project.  If *None*, a random
        tag is generated (used for brand‑new projects with no backing file
        yet).
    """
    if anchor is None:
        return uuid.uuid4().hex[:8]
    norm = os.path.normcase(str(anchor.resolve()))
    return hash_path(norm, length=8)


# -----------------------------------------------------------------------------
#  Backup / autosave helpers
# -----------------------------------------------------------------------------

def rotate_backups(base: Path, *, max_keep: int = 10) -> Path:
    """Create a timestamp‑suffixed backup file and purge old ones.

    The *base* argument should be the *desired* base name, e.g.::

        base = Path('project_autosave.slt')
        new = rotate_backups(base)
        print(new.name)
        project_autosave_20250527_103012.slt

    Files are sorted by last‑modified time; the oldest beyond *max_keep* are
    deleted to bound disk usage.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = base.with_name(f"{base.stem}_{ts}{base.suffix}")

    pattern = f"{base.stem}_*{base.suffix}"
    backups = sorted(base.parent.glob(pattern), key=lambda p: p.stat().st_mtime)

    for old in backups[:-max_keep + 1]:
        try:
            old.unlink(missing_ok=True)
            logging.debug("Deleted old autosave %s", old.name)
        except OSError:
            logging.warning("Could not delete backup %s", old)

    return new_path


def autosave_files(temp_dir: Path, basename: str) -> List[Path]:
    """Return autosave files sorted *newest‑first*.

    ``basename`` should *not* include the timestamp suffix – e.g.
    ``'project_autosave.slt'``.  The function appends ``_*`` automatically.
    """
    pattern = f"{basename.split('.')[0]}_*.{basename.split('.')[-1]}"
    return sorted(
        temp_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
