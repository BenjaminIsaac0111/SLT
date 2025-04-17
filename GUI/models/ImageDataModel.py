#!/usr/bin/env python3
import logging
import sqlite3
import threading
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
from PyQt5.QtCore import QObject

from GUI.models.CacheManager import CacheManager


class ABCQObjectMeta(ABCMeta, type(QObject)):
    """Metaclass combining ABCMeta and QObject's metaclass for abstract QObjects."""
    pass


class BaseImageDataModel(QObject, metaclass=ABCQObjectMeta):
    """
    Abstract base class for image data backends (HDF5, SQLite, etc.).
    """

    @property
    @abstractmethod
    def data_path(self) -> str:
        """Path to the underlying data file (HDF5 or SQLite)."""
        ...

    @property
    @abstractmethod
    def backend(self) -> str:
        """Return a short identifier such as ``'hdf5'`` or ``'sqlite'``."""
        ...

    @abstractmethod
    def get_image_data(self, index: int) -> Dict[str, Any]:
        """Retrieve a single sample by index."""
        ...

    @abstractmethod
    def get_filenames(self) -> List[str]:
        """List all sample filenames."""
        ...

    @abstractmethod
    def get_number_of_images(self) -> int:
        """Total number of samples."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up any open resources."""
        ...


class HDF5ImageDataModel(BaseImageDataModel):
    """
    HDF5-backed implementation of BaseImageDataModel.
    """
    cache = CacheManager()

    def __init__(self, project_state: dict):
        super().__init__()
        self._db_path = project_state['data_path']
        self._uncertainty_type = project_state['uncertainty']
        self._hdf5_file: Optional[h5py.File] = None
        self._images = None
        self._logits = None
        self._uncertainty = None
        self._filenames_list: List[str] = []
        self._lock = threading.Lock()
        logging.info(f"HDF5ImageDataModel initialized (uncertainty: '{project_state['uncertainty']}').")

    @property
    def backend(self) -> str:
        return "hdf5"

    @property
    def data_path(self) -> str:
        return self._db_path

    def get_image_data(self, index: int) -> Dict[str, Any]:
        self._ensure_open()
        with self._lock:
            if not (0 <= index < len(self._filenames_list)):
                raise IndexError("Index out of range.")
        key = (index, self._uncertainty_type)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        with self._lock:
            data = {
                'image': self._images[index],
                'logits': self._logits[index],
                'uncertainty': self._uncertainty[index],
                'filename': self._filenames_list[index],
            }
        self.cache.set(key, data)
        return data

    def get_filenames(self) -> List[str]:
        self._ensure_open()
        return list(self._filenames_list)

    def get_number_of_images(self) -> int:
        self._ensure_open()
        return len(self._filenames_list)

    def close(self) -> None:
        with self._lock:
            if self._hdf5_file is not None:
                self._hdf5_file.close()
                self._hdf5_file = None
                logging.info("HDF5 file closed.")

    def _ensure_open(self) -> None:
        """
        Opens the HDF5 file and loads datasets if necessary.
        """
        with self._lock:
            if self._images is not None:
                return
            if self._hdf5_file is None:
                rdcc_nbytes = 1024 * (1024 ** 2)
                rdcc_nslots = 1_000_003
                self._hdf5_file = h5py.File(
                    self._hdf5_file_path,
                    mode='r',
                    libver='latest',
                    rdcc_nbytes=rdcc_nbytes,
                    rdcc_nslots=rdcc_nslots,
                    swmr=True
                )
                logging.info("HDF5 file opened.")
            # load datasets
            self._images = self._hdf5_file['rgb_images']
            self._logits = self._hdf5_file['logits']
            if self._uncertainty_type not in self._hdf5_file:
                raise ValueError(f"Uncertainty dataset '{self._uncertainty_type}' not found.")
            self._uncertainty = self._hdf5_file[self._uncertainty_type]
            self._filenames_list = list(self._hdf5_file['filenames'].asstr())
            logging.info("HDF5 datasets ready.")


class SQLiteImageDataModel(BaseImageDataModel):
    """
    SQLite-backed implementation of BaseImageDataModel.
    """
    cache = CacheManager()

    def __init__(self, project_state: dict):
        super().__init__()
        self._db_path = project_state['data_path']
        self._uncertainty_type = project_state['uncertainty']
        self._conn: Optional[sqlite3.Connection] = None
        self._filenames_list: List[str] = []
        self._lock = threading.Lock()
        logging.info(f"SQLiteImageDataModel initialized (uncertainty: '{project_state['uncertainty']}').")

    @property
    def backend(self) -> str:
        return "sqlite"

    @property
    def data_path(self) -> str:
        return self._db_path

    def get_image_data(self, index: int) -> Dict[str, Any]:
        self._ensure_open()
        with self._lock:
            if not (0 <= index < len(self._filenames_list)):
                raise IndexError("Index out of range.")
        key = (index, self._uncertainty_type)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        sample_id = index + 1
        data: Dict[str, Any] = {}
        cur = self._conn.cursor()
        for name in ('rgb', 'logits', self._uncertainty_type):
            cur.execute(
                """
                SELECT a.data, s.ndims, s.dim0, s.dim1, s.dim2, s.dim3, s.dtype
                FROM arrays a JOIN shapes s ON a.array_id = s.array_id
                WHERE a.sample_id=? AND a.name=?
                """, (sample_id, name)
            )
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"Missing '{name}' for sample {sample_id}.")
            blob = row['data']
            dims = [row[f'dim{i}'] for i in range(row['ndims'])]
            arr = np.frombuffer(blob, dtype=row['dtype']).reshape(dims)
            if name == 'rgb':
                data['image'] = arr
            elif name == self._uncertainty_type:
                data['uncertainty'] = arr
            else:
                data[name] = arr
        data['filename'] = self._filenames_list[index]
        self.cache.set(key, data)
        return data

    def get_filenames(self) -> List[str]:
        self._ensure_open()
        return list(self._filenames_list)

    def get_number_of_images(self) -> int:
        return len(self.get_filenames())

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
                logging.info("SQLite DB closed.")

    def _ensure_open(self) -> None:
        """
        Opens the DB and loads filenames if necessary.
        """
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                logging.info("SQLite DB opened.")
            if not self._filenames_list:
                cur = self._conn.cursor()
                cur.execute("SELECT filename FROM samples ORDER BY sample_id")
                self._filenames_list = [r[0] for r in cur.fetchall()]
                logging.info(f"Loaded {len(self._filenames_list)} filenames.")


def create_image_data_model(project_state: Dict[str, Any]) -> BaseImageDataModel:
    """
    Build a concrete ImageDataModel from the *project_state* record that is
    serialized in every .json.gz save‑file.
    Expected keys:
        data_backend   – "hdf5" | "sqlite"
        data_path      – absolute/relative path to the data file
        uncertainty    – e.g. "bald"   (defaults to "bald" if absent)
    """
    if project_state['data_backend'] == "hdf5":
        return HDF5ImageDataModel(project_state)
    if project_state['data_backend'] == "sqlite":
        return SQLiteImageDataModel(project_state)

    raise ValueError(f"Unsupported backend {project_state['data_backend']!r}")
