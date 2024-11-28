import logging
from typing import Dict, Any, List, Optional

import h5py
import numpy as np
from PyQt5.QtCore import QObject
from PyQt5.QtGui import QPixmap

from GUI.models.CacheManager import CacheManager  # Ensure the correct import path


class ImageDataModel(QObject):
    """
    Class for loading and accessing image data, logits, uncertainties, and filenames
    from an HDF5 file.
    """

    # Shared Singleton CacheManager
    cache = CacheManager(cache_size=1024)  # Adjust cache size as needed

    def __init__(self, hdf5_file_path: str):
        super().__init__()
        self._hdf5_file_path = hdf5_file_path
        self._hdf5_file: Optional[h5py.File] = None
        self._images = None
        self._logits = None
        self._variance_uncertainty = None
        self._bald_uncertainty = None
        self._filenames = []
        self._current_pixmap: Optional[QPixmap] = None
        logging.info("ImageDataModel initialized.")

    @property
    def hdf5_file_path(self) -> str:
        """Public property to access the HDF5 file path."""
        return self._hdf5_file_path

    def _load_hdf5_file(self):
        """Opens the HDF5 file if not already open."""
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
        """Loads datasets from HDF5 file if not already loaded."""
        if self._images is None:
            try:
                self._images = self._hdf5_file['rgb_images']
                self._logits = self._hdf5_file['logits']
                self._variance_uncertainty = self._hdf5_file['variance']
                self._bald_uncertainty = self._hdf5_file['bald']
                self._filenames = list(self._hdf5_file['filenames'].asstr())
                logging.info("Datasets loaded successfully.")
            except KeyError as e:
                logging.error(f"Missing dataset in HDF5 file: {e}")
                raise

    def get_image_data(self, index: int, uncertainty_type: str = 'bald') -> Dict[str, Any]:
        """
        Fetches image data, using CacheManager to minimize I/O.

        Parameters:
            index (int): Index of the image data to retrieve.
            uncertainty_type (str): Type of uncertainty to retrieve ('variance' or 'bald').

        Returns:
            Dict[str, Any]: A dictionary containing the image, logits, uncertainty, and filename.
        """
        self._load_hdf5_file()
        self._load_datasets()

        if index < 0 or index >= len(self._filenames):
            logging.error("Index out of range.")
            raise IndexError("Index out of range.")

        # Generate a unique cache key using the index and uncertainty type
        cache_key = (index, uncertainty_type)

        # Cache access using CacheManager
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logging.debug(f"Data for index {index} with uncertainty '{uncertainty_type}' retrieved from cache.")
            return cached_data

        # Select the appropriate uncertainty
        if uncertainty_type == 'variance':
            uncertainty = self._variance_uncertainty[index]
        elif uncertainty_type == 'bald':
            uncertainty = self._bald_uncertainty[index]
        else:
            logging.error(f"Unknown uncertainty type: {uncertainty_type}")
            raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")

        # Data retrieval and caching
        data = {
            'image': self._images[index],
            'logits': self._logits[index],
            'uncertainty': uncertainty,
            'filename': self._filenames[index]
        }
        # Store data in cache
        self.cache.set(cache_key, data)
        logging.debug(f"Data for index {index} with uncertainty '{uncertainty_type}' cached.")
        return data

    def get_filenames(self) -> List[str]:
        """Returns a copy of the filenames."""
        self._load_hdf5_file()
        self._load_datasets()
        return self._filenames[:]

    def get_number_of_images(self) -> int:
        """Returns the number of images in the dataset."""
        self._load_hdf5_file()
        self._load_datasets()
        return len(self._filenames)

    def close(self):
        """Closes the HDF5 file."""
        if self._hdf5_file is not None:
            self._hdf5_file.close()
            self._hdf5_file = None
            logging.info("HDF5 file closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        logging.info("ImageDataModel exited.")
