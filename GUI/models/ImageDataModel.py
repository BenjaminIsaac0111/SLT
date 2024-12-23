import logging
import threading
from typing import Dict, Any, List, Optional

import h5py
from PyQt5.QtCore import QObject

from GUI.models.CacheManager import CacheManager


class ImageDataModel(QObject):
    """
    Loads/accesses image data, logits, uncertainties, and filenames from HDF5.
    Optimized for large datasets by performing partial reads and chunk caching.
    """

    # Shared singleton LRU cache (for repeated lookups of identical indexes)
    cache = CacheManager()

    def __init__(self, hdf5_file_path: str, uncertainty_type: str):
        """
        :param hdf5_file_path: Path to the HDF5 file.
        :param uncertainty_type: ID of the uncertainty dataset to read (e.g. 'variance').
        """
        super().__init__()
        self._hdf5_file_path = hdf5_file_path
        self._uncertainty_type = uncertainty_type
        self._hdf5_file: Optional[h5py.File] = None

        # Datasets remain as h5py objects for chunk-based partial reads
        self._images = None
        self._logits = None
        self._uncertainty = None
        self._filenames = None

        self._filenames_list: List[str] = []
        self._lock = threading.Lock()

        logging.info(f"ImageDataModel initialized (uncertainty: '{uncertainty_type}').")

    @property
    def hdf5_file_path(self) -> str:
        """
        Returns the path to the HDF5 file.
        """
        return self._hdf5_file_path

    def get_image_data(self, index: int) -> Dict[str, Any]:
        """
        Retrieves image data for a single index. Uses the class-level LRU cache
        to avoid repeated HDF5 reads for the same index.

        :param index: Integer index in [0..N-1].
        :return: Dict with 'image', 'logits', 'uncertainty', 'filename'.
        """
        self._ensure_hdf5_open()
        self._ensure_datasets_ready()

        with self._lock:
            if index < 0 or index >= len(self._filenames_list):
                logging.error(f"Index {index} out of range.")
                raise IndexError("Index out of range.")

        # Check the LRU cache
        cache_key = (index, self._uncertainty_type)
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logging.debug(f"Data for index {index} retrieved from cache.")
            return cached_data

        # Read chunked data on demand
        with self._lock:
            data = {
                'image': self._images[index],  # Partial read from HDF5
                'logits': self._logits[index],
                'uncertainty': self._uncertainty[index],
                'filename': self._filenames_list[index],
            }

        self.cache.set(cache_key, data)
        logging.debug(f"Data for index {index} cached.")
        return data

    def get_filenames(self) -> List[str]:
        """
        Returns all filenames (as a Python list).
        """
        self._ensure_hdf5_open()
        self._ensure_datasets_ready()
        with self._lock:
            return list(self._filenames_list)

    def get_number_of_images(self) -> int:
        """
        Returns the total number of images in the dataset.
        """
        self._ensure_hdf5_open()
        self._ensure_datasets_ready()
        with self._lock:
            return len(self._filenames_list)

    def close(self) -> None:
        """
        Closes the HDF5 file if it is open.
        """
        with self._lock:
            if self._hdf5_file is not None:
                self._hdf5_file.close()
                self._hdf5_file = None
                logging.info("HDF5 file closed.")

    # -------------------------------------------------------------------------
    #                              PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _ensure_hdf5_open(self) -> None:
        """
        Opens the HDF5 file if not already open.
        Potentially tune chunk cache parameters to improve partial read performance.
        """
        with self._lock:
            if self._hdf5_file is not None:
                return  # Already open

            try:
                # Example chunk cache tuning:
                rdcc_nbytes = 1024 * (1024 ** 2)  # 256MB chunk cache
                rdcc_nslots = 1_000_003  # # of chunk slots
                self._hdf5_file = h5py.File(
                    self._hdf5_file_path,
                    mode='r',
                    libver='latest',
                    rdcc_nbytes=rdcc_nbytes,
                    rdcc_nslots=rdcc_nslots,
                    swmr=True
                )
                logging.info("HDF5 file opened.")
            except Exception as exc:
                logging.error(f"Failed to open HDF5 file '{self._hdf5_file_path}': {exc}")
                raise

    def _ensure_datasets_ready(self) -> None:
        """
        Loads dataset references for chunk-based partial reads.
        Also reads the filenames into a list for quick in-memory access.
        """
        with self._lock:
            if self._images is not None and self._logits is not None and self._uncertainty is not None:
                return  # Already loaded references

            if self._hdf5_file is None:
                msg = "HDF5 file is not open; cannot load datasets."
                logging.error(msg)
                raise RuntimeError(msg)

            try:
                self._images = self._hdf5_file['rgb_images']
                self._logits = self._hdf5_file['logits']
                # Check if the user’s specified uncertainty dataset exists
                if self._uncertainty_type not in self._hdf5_file:
                    raise ValueError(f"Uncertainty dataset '{self._uncertainty_type}' not in HDF5.")
                self._uncertainty = self._hdf5_file[self._uncertainty_type]

                filenames_dataset = self._hdf5_file['filenames']
                # Convert filenames to strings in memory for fast access
                self._filenames_list = list(filenames_dataset.asstr())

                logging.info("Datasets references loaded for chunk-based reads.")
            except KeyError as err:
                logging.error(f"Missing dataset in HDF5 file: {err}")
                raise
            except Exception as exc:
                logging.error(f"Error loading dataset references: {exc}")
                raise

    # -------------------------------------------------------------------------
    #                         CONTEXT MANAGER SUPPORT
    # -------------------------------------------------------------------------

    def __enter__(self):
        self._ensure_hdf5_open()
        self._ensure_datasets_ready()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        logging.info("ImageDataModel context exited.")
