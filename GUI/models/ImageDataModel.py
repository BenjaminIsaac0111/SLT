# ImageDataModel.py
"""Back‑end–agnostic data‑access layer for HDF5 and SQLite files.

Key design points
-----------------
* **Thread‑local native resources** – every thread obtains and owns its own
  *h5py.File* or *sqlite3.Connection*.  This avoids undefined behaviour that
  occurs when native handles are shared across Python threads after the GIL has
  been released (NumPy C‑loops, Qt, etc.).
* **Immutable shared metadata** – expensive metadata such as the list of
  filenames is computed once under the GIL and then shared read‑only across
  threads.
* **Transparent caching** – heavy unmarshalling (BLOB→NumPy) is memorised via
  ``CacheManager`` (LRU).
* **Back‑end factory** – ``create_image_data_model()`` instantiates the correct
  concrete class based solely on the ``project_state`` dictionary found in each
  ``*.json.gz`` save‑file.

Both concrete subclasses implement the same public interface defined by
:class:`BaseImageDataModel`.
"""

import logging
import sqlite3
import threading
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Sequence, Optional

import h5py
import numpy as np
from PyQt5.QtCore import QObject

from GUI.models.CacheManager import CacheManager
from GUI.models.StatePersistance import ProjectState


# -----------------------------------------------------------------------------
# metaclass helper so that an abstract QObject can exist
# -----------------------------------------------------------------------------
class ABCQObjectMeta(ABCMeta, type(QObject)):
    """Combine ``ABCMeta`` and Qt's ``sip.wrappertype`` metaclasses."""


# -----------------------------------------------------------------------------
# abstract base class
# -----------------------------------------------------------------------------
class BaseImageDataModel(QObject, metaclass=ABCQObjectMeta):
    """Interface common to all image data back‑ends."""

    # ----------------------- abstract API -----------------------------------
    @property
    @abstractmethod
    def data_path(self) -> str:
        """Absolute or relative path of the underlying data file."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Short identifier such as ``"hdf5"`` or ``"sqlite"``."""

    @abstractmethod
    def get_image_data(self, index: int) -> Dict[str, Any]:
        """Return the *index*‑th sample as a dictionary with keys
        ``image``, ``logits``, ``uncertainty``, ``filename``."""

    @abstractmethod
    def get_filenames(self) -> List[str]:
        """Return all filenames stored in the dataset in order."""

    @abstractmethod
    def get_number_of_images(self) -> int:
        """Number of samples in the data set."""

    @abstractmethod
    def close(self) -> None:
        """Close any native handles owned by the *current* thread."""


# -----------------------------------------------------------------------------
# HDF5 implementation
# -----------------------------------------------------------------------------
class HDF5ImageDataModel(BaseImageDataModel):
    """Thread‑local SWMR HDF5 reader."""

    def __init__(self, project_state: ProjectState):
        super().__init__()
        self._db_path: str = project_state.data_path
        self._uncertainty_type: str = project_state.uncertainty

        # thread‑local storage for per‑thread h5py.File handle and datasets
        self._tls = threading.local()

        # immutable metadata (shared)
        self._filenames: List[str] = []
        self._meta_lock = threading.Lock()
        logging.info("HDF5ImageDataModel initialised for %s", self._db_path)

    # ------------ BaseImageDataModel implementation -------------------------
    @property
    def backend(self) -> str:
        return "hdf5"

    @property
    def data_path(self) -> str:
        return self._db_path

    # -------------------- public API ----------------------------------------
    def get_image_data(self, index: int) -> Dict[str, Any]:
        self._ensure_open()
        if not (0 <= index < len(self._filenames)):
            raise IndexError("index out of range")

        f: h5py.File = self._tls.h5f
        image = f["rgb_images"][index]
        logits = f["logits"][index]
        uncertainty = f[self._uncertainty_type][index]
        fname = self._filenames[index]

        record = {
            "image": image,
            "logits": logits,
            "uncertainty": uncertainty,
            "filename": fname,
        }
        return record

    def get_filenames(self) -> List[str]:
        self._ensure_open()
        return list(self._filenames)

    def get_number_of_images(self) -> int:
        self._ensure_open()
        return len(self._filenames)

    def close(self) -> None:
        """Close the per‑thread HDF5 file handle (no effect on other threads)."""
        h5f: Optional[h5py.File] = getattr(self._tls, "h5f", None)
        if h5f is not None:
            h5f.close()
            del self._tls.h5f
            logging.info("HDF5 file closed in thread %s", threading.get_ident())

    # ---------------- internal helpers --------------------------------------
    def _ensure_open(self) -> None:
        """Open HDF5 in *this* thread if not yet opened, and load filenames once."""
        if not hasattr(self._tls, "h5f"):
            rdcc_nbytes = 1024 * 1024 * 1024  # 1 GiB raw chunk cache
            self._tls.h5f = h5py.File(
                self._db_path,
                mode="r",
                libver="latest",
                rdcc_nbytes=rdcc_nbytes,
                swmr=True,
            )
            logging.debug("HDF5 opened in thread %s", threading.get_ident())

        # load filenames exactly once (under the GIL)
        if not self._filenames:
            with self._meta_lock:
                if not self._filenames:  # double‑checked locking
                    self._filenames = list(self._tls.h5f["filenames"].asstr())
                    logging.info("Loaded %d filenames from HDF5", len(self._filenames))


# -----------------------------------------------------------------------------
# SQLite implementation
# -----------------------------------------------------------------------------
class SQLiteImageDataModel(BaseImageDataModel):
    """Thread‑local SQLite reader with zero shared mutable state."""

    cache = CacheManager()

    def __init__(self, project_state: ProjectState):
        super().__init__()
        self._db_path: str = project_state["data_path"]
        self._uncertainty_type: str = project_state.get("uncertainty", "bald")

        # thread‑local storage for sqlite3.Connection
        self._tls = threading.local()

        # immutable shared metadata
        self._filenames: List[str] = []
        self._meta_lock = threading.Lock()
        logging.info("SQLiteImageDataModel initialised for %s", self._db_path)

    # ---------------------------------------------------------------------
    # connection helper – property returns a live connection for *this thread*
    # ---------------------------------------------------------------------
    @property
    def _conn(self) -> sqlite3.Connection:
        conn: Optional[sqlite3.Connection] = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            conn.row_factory = sqlite3.Row
            self._tls.conn = conn
            logging.debug("SQLite connection opened in thread %s", threading.get_ident())
        return conn

    # ------------------ BaseImageDataModel API -----------------------------
    @property
    def backend(self) -> str:
        return "sqlite"

    @property
    def data_path(self) -> str:
        return self._db_path

    def get_image_data(self, index: int) -> Dict[str, Any]:
        self._ensure_open()
        if not (0 <= index < len(self._filenames)):
            raise IndexError("index out of range")

        key = (index, self._uncertainty_type)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        sample_id = index + 1  # 1‑based in the DB schema
        cur = self._conn.cursor()

        def _load_array(name: str) -> np.ndarray:
            cur.execute(
                """
                SELECT a.data, s.ndims, s.dim0, s.dim1, s.dim2, s.dim3, s.dtype
                  FROM arrays a
                  JOIN shapes s ON a.array_id = s.array_id
                 WHERE a.sample_id = ? AND a.name = ?
                """,
                (sample_id, name),
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"missing array '{name}' for sample {sample_id}")
            dims: Sequence[int] = [row[f"dim{i}"] for i in range(row["ndims"])]
            blob_bytes: bytes = row["data"]  # this is the single allocation
            arr = np.frombuffer(blob_bytes, dtype=row["dtype"])
            return arr.reshape(dims)

        image = _load_array("rgb")
        logits = _load_array("logits")
        uncertainty = _load_array(self._uncertainty_type)
        fname = self._filenames[index]

        record = {
            "image": image,
            "logits": logits,
            "uncertainty": uncertainty,
            "filename": fname,
        }
        self.cache.set(key, record)
        return record

    def get_filenames(self) -> List[str]:
        self._ensure_open()
        return list(self._filenames)

    def get_number_of_images(self) -> int:
        self._ensure_open()
        return len(self._filenames)

    def close(self) -> None:
        conn: Optional[sqlite3.Connection] = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            del self._tls.conn
            logging.info("SQLite connection closed in thread %s", threading.get_ident())

    # ---------------- internal helpers ----------------------------------
    def _ensure_open(self) -> None:
        """Make sure this thread has a connection and filenames are loaded."""
        _ = self._conn  # triggers connection creation if needed
        if not self._filenames:
            with self._meta_lock:
                if not self._filenames:  # double‑checked locking
                    cur = self._conn.cursor()
                    cur.execute("SELECT filename FROM samples ORDER BY sample_id")
                    self._filenames = [row[0] for row in cur.fetchall()]
                    logging.info("Loaded %d filenames from SQLite", len(self._filenames))


# -----------------------------------------------------------------------------
# factory
# -----------------------------------------------------------------------------

def create_image_data_model(
        project_state: ProjectState
) -> BaseImageDataModel:
    backend = project_state.data_backend.lower()
    if backend == "hdf5":
        return HDF5ImageDataModel(project_state)
    if backend == "sqlite":
        return SQLiteImageDataModel(project_state)
    raise ValueError(f"Unsupported backend '{backend}'.")
