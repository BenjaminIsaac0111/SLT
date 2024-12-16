import logging
import threading
from typing import Dict, Any, List, Optional

import h5py
from PyQt5.QtCore import QObject
from PyQt5.QtGui import QPixmap

from GUI.models.CacheManager import CacheManager


class ImageDataModel(QObject):
    """
    Class for loading and accessing image data, logits, uncertainties, and filenames
    from an HDF5 file.
    """

    # Shared Singleton CacheManager
    cache = CacheManager()

    def __init__(self, hdf5_file_path: str, uncertainty_type: str):
        """
        Initialize the ImageDataModel with the HDF5 file path and the desired uncertainty dataset.

        Parameters:
            hdf5_file_path (str): Path to the HDF5 file.
            uncertainty_type (str): The ID of the uncertainty dataset to use (e.g., 'variance', 'bald').
        """
        super().__init__()
        self._hdf5_file_path = hdf5_file_path
        self._uncertainty_type = uncertainty_type  # Store the selected uncertainty dataset ID
        self._hdf5_file: Optional[h5py.File] = None
        self._images = None
        self._logits = None
        self._uncertainty = None
        self._filenames = []
        self._current_pixmap: Optional[QPixmap] = None
        self._lock = threading.Lock()  # To ensure thread-safe operations

        logging.info(f"ImageDataModel initialized with uncertainty dataset '{uncertainty_type}'.")

    @property
    def hdf5_file_path(self) -> str:
        """Public property to access the HDF5 file path."""
        return self._hdf5_file_path

    def _load_hdf5_file(self):
        """Opens the HDF5 file if not already open."""
        with self._lock:
            if self._hdf5_file is None:
                try:
                    self._hdf5_file = h5py.File(
                        self._hdf5_file_path,
                        'r',
                        libver='latest',
                        swmr=True,
                    )
                    logging.info("HDF5 file opened successfully.")
                except Exception as e:
                    logging.error(f"Failed to open HDF5 file: {e}")
                    raise

    def _load_datasets(self):
        """Loads datasets from the HDF5 file."""
        with self._lock:
            if self._images is None:
                try:
                    self._images = self._hdf5_file['rgb_images']
                    self._logits = self._hdf5_file['logits']
                    self._filenames = list(self._hdf5_file['filenames'].asstr())

                    # Load the specified uncertainty dataset
                    if self._uncertainty_type in self._hdf5_file:
                        self._uncertainty = self._hdf5_file[self._uncertainty_type]
                    else:
                        logging.error(f"Uncertainty dataset '{self._uncertainty_type}' not found in HDF5 file.")
                        raise ValueError(f"Uncertainty dataset '{self._uncertainty_type}' not found in HDF5 file.")

                    logging.info("Datasets loaded successfully.")
                except KeyError as e:
                    logging.error(f"Missing dataset in HDF5 file: {e}")
                    raise

    def get_image_data(self, index: int) -> Dict[str, Any]:
        """
        Fetches image data, using CacheManager to minimize I/O.

        Parameters:
            index (int): Index of the image data to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the image, logits, uncertainty, and filename.
        """
        self._load_hdf5_file()
        self._load_datasets()

        with self._lock:
            if index < 0 or index >= len(self._filenames):
                logging.error("Index out of range.")
                raise IndexError("Index out of range.")

        # Generate a unique cache key using the index and uncertainty type
        cache_key = (index, self._uncertainty_type)
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logging.debug(f"Data for index {index} retrieved from cache.")
            return cached_data

        # Retrieve data
        with self._lock:
            data = {
                'image': self._images[index],
                'logits': self._logits[index],
                'uncertainty': self._uncertainty[index],
                'filename': self._filenames[index]
            }

        self.cache.set(cache_key, data)
        logging.debug(f"Data for index {index} cached.")
        return data

    def get_filenames(self) -> List[str]:
        """Returns a copy of the filenames."""
        self._load_hdf5_file()
        self._load_datasets()
        with self._lock:
            return self._filenames[:]

    def get_number_of_images(self) -> int:
        """Returns the number of images in the dataset."""
        self._load_hdf5_file()
        self._load_datasets()
        with self._lock:
            return len(self._filenames)

    def close(self):
        """Closes the HDF5 file."""
        with self._lock:
            if self._hdf5_file is not None:
                self._hdf5_file.close()
                self._hdf5_file = None
                logging.info("HDF5 file closed.")

    def __enter__(self):
        self._load_hdf5_file()
        self._load_datasets()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        logging.info("ImageDataModel exited.")
