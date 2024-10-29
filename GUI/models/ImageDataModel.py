import logging
import threading
from typing import Dict, Any, List, Optional, Tuple

import h5py
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage


class ImageDataModel:
    """
    ImageDataModel is responsible for loading and providing access to image data, logits, uncertainties, and filenames
    from an HDF5 file.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, hdf5_file_path: str):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ImageDataModel, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, hdf5_file_path: str):
        """
        Initializes the ImageDataModel with the given HDF5 file path. Implements lazy loading of datasets.

        :param hdf5_file_path: Path to the HDF5 file containing the datasets.
        """
        if not self._initialized:
            self._hdf5_file_path: str = hdf5_file_path
            self._hdf5_file: Optional[h5py.File] = None
            self._images: Optional[h5py.Dataset] = None
            self._logits: Optional[h5py.Dataset] = None
            self._epistemic_uncertainties: Optional[h5py.Dataset] = None
            self._aleatoric_uncertainties: Optional[h5py.Dataset] = None
            self._filenames: List[str] = []
            self._data_lock = threading.Lock()
            self._current_pixmap: Optional[QPixmap] = None  # To store the current QPixmap
            self._initialized = True
            self._cache_lock = threading.Lock()
            self._cache_size = 256  # Adjust cache size as needed
            self._data_cache = {}
            logging.info("ImageDataModel instance created.")

    def _load_hdf5_file(self) -> None:
        """
        Opens the HDF5 file using a context manager to ensure it's properly managed.
        """
        if self._hdf5_file is None:
            try:
                # Open the file in SWMR mode for thread-safe read access
                self._hdf5_file = h5py.File(
                    self._hdf5_file_path,
                    'r',
                    libver='latest',
                    swmr=True,
                    rdcc_nbytes=1024 ** 3,  # 1GB chunk cache
                    rdcc_w0=0.75
                )
                logging.info("HDF5 file opened successfully in SWMR mode.")
            except Exception as e:
                logging.error(f"Failed to open HDF5 file: {e}")
                raise e

    def _load_datasets(self) -> None:
        """
        Lazily loads datasets from the HDF5 file when accessed.
        """
        with self._data_lock:
            if self._images is None:
                self._images = self._hdf5_file['rgb_images']
                logging.info("Images dataset loaded.")
            if self._logits is None:
                self._logits = self._hdf5_file['logits']
                logging.info("Logits dataset loaded.")
            if self._epistemic_uncertainties is None:
                self._epistemic_uncertainties = self._hdf5_file['epistemic_uncertainty']
            if self._aleatoric_uncertainties is None:
                self._aleatoric_uncertainties = self._hdf5_file['aleatoric_uncertainty']
                logging.info("Uncertainty datasets loaded.")
            if not self._filenames:
                filenames_dataset = self._hdf5_file['filenames']
                # Load filenames more efficiently
                self._filenames = [name for name in filenames_dataset.asstr()]
                logging.info("Filenames dataset loaded.")

    def get_image_data(self, index: int) -> Dict[str, Any]:
        """
        Retrieves the image data, logits, uncertainty, and filename for the given index.

        :param index: Index of the data to retrieve.
        :return: A dictionary containing 'image', 'logits', 'uncertainty', and 'filename'.
        """
        self._load_hdf5_file()
        self._load_datasets()

        if index < 0 or index >= len(self._filenames):
            logging.error("Index out of range.")
            raise IndexError("Index out of range.")

        # Use caching to reduce I/O
        with self._cache_lock:
            if index in self._data_cache:
                logging.debug(f"Cache hit for index {index}.")
                return self._data_cache[index]

        # Read data without holding the lock to reduce contention
        image = self._images[index]
        logits = self._logits[index]
        epistemic_uncertainties = self._epistemic_uncertainties[index]
        aleatoric_uncertainties = self._aleatoric_uncertainties[index][..., np.newaxis]
        uncertainties = epistemic_uncertainties - aleatoric_uncertainties
        filename = self._filenames[index]

        data = {
            'image': image,
            'logits': logits,
            'epistemic_uncertainty': epistemic_uncertainties,
            'aleatoric_uncertainty': aleatoric_uncertainties,
            'uncertainty': uncertainties,
            'filename': filename
        }

        # Update cache
        with self._cache_lock:
            if len(self._data_cache) >= self._cache_size:
                # Remove the oldest item
                removed_index = next(iter(self._data_cache))
                del self._data_cache[removed_index]
                logging.debug(f"Removed index {removed_index} from cache.")
            self._data_cache[index] = data
            logging.debug(f"Added index {index} to cache.")

        return data

    def create_zoomed_crop(self, image_array: np.ndarray, coord: Tuple[int, int], crop_size: int = 256,
                           zoom_factor: int = 2) -> Tuple[QPixmap, int, int]:
        """
        Creates a zoomed crop of the given image array around the specified coordinate.

        :param image_array: Numpy array of the original image (scaled 0.0-1.0 or uint8).
        :param coord: Tuple of (row, column) representing the center of the crop.
        :param crop_size: The size of the crop in pixels.
        :param zoom_factor: The factor by which to zoom the cropped image.
        :return: A tuple containing the zoomed QPixmap, x_start, and y_start.
        """
        logging.debug(
            f"Creating zoomed crop at coord: {coord} with crop_size: {crop_size} and zoom_factor: {zoom_factor}")

        # Convert the image to uint8 if necessary
        if image_array.dtype != np.uint8:
            logging.debug("Converting image array to uint8.")
            image_array = (image_array * 255).astype(np.uint8)

        height, width, channels = image_array.shape

        if channels != 3:
            raise ValueError(f"Expected 3 channels for RGB image, but got {channels}")

        # Convert numpy array to QImage without copying
        q_image = QImage(
            image_array.data,
            width,
            height,
            3 * width,
            QImage.Format_RGB888
        ).copy()  # Copy to ensure it doesn't reference temporary data
        original_pixmap = QPixmap.fromImage(q_image)

        x_center, y_center = coord[1], coord[0]  # (col, row) to (x, y)

        # Calculate the starting points of the crop
        x_start = max(0, x_center - crop_size // 2)
        y_start = max(0, y_center - crop_size // 2)

        # Adjust if crop exceeds image boundaries
        x_start = min(x_start, width - crop_size)
        y_start = min(y_start, height - crop_size)
        x_start = max(x_start, 0)
        y_start = max(y_start, 0)

        width_crop = min(crop_size, width - x_start)
        height_crop = min(crop_size, height - y_start)

        logging.debug(f"Crop bounds: x_start={x_start}, y_start={y_start}, width={width_crop}, height={height_crop}")

        # Crop the image
        cropped_pixmap = original_pixmap.copy(x_start, y_start, width_crop, height_crop)

        # Zoom the cropped pixmap
        zoomed_pixmap = cropped_pixmap.scaled(
            crop_size * zoom_factor,
            crop_size * zoom_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        logging.debug("Zoomed crop created successfully.")

        return zoomed_pixmap, x_start, y_start

    def get_current_pixmap(self) -> Optional[QPixmap]:
        """
        Retrieves the current QPixmap being displayed.

        :return: The current QPixmap or None if not set.
        """
        # No need for lock as we're only reading
        return self._current_pixmap

    def set_current_pixmap(self, pixmap: QPixmap):
        """
        Sets the current QPixmap.

        :param pixmap: The QPixmap to set as current.
        """
        # No need for lock as QPixmap is thread-safe
        self._current_pixmap = pixmap
        logging.debug("Current pixmap updated.")

    def get_pixmap_by_index(self, index: int) -> QPixmap:
        """
        Retrieves the QPixmap for the given index.

        :param index: Index of the image to retrieve.
        :return: The QPixmap of the image.
        """
        image_data = self.get_image_data(index)
        image_array = image_data['image']

        # Convert the numpy array to QPixmap efficiently
        if image_array.dtype != np.uint8:
            logging.debug("Converting image array to uint8.")
            image_array = (image_array * 255).astype(np.uint8)

        height, width, channels = image_array.shape
        bytes_per_line = channels * width

        q_image = QImage(
            image_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        ).copy()  # Copy to ensure it doesn't reference temporary data
        pixmap = QPixmap.fromImage(q_image)

        self.set_current_pixmap(pixmap)

        return pixmap

    def get_filenames(self) -> List[str]:
        """
        Retrieves the list of all filenames in the dataset.

        :return: A list of filenames as strings.
        """
        self._load_hdf5_file()
        self._load_datasets()
        logging.debug("Retrieving all filenames.")
        return self._filenames.copy()

    def get_number_of_images(self) -> int:
        """
        Returns the total number of entries in the dataset.

        :return: Total number of entries.
        """
        self._load_hdf5_file()
        self._load_datasets()
        return len(self._filenames)

    def get_index_by_filename(self, filename: str) -> int:
        """
        Retrieves the index of the given filename.

        :param filename: The filename to search for.
        :return: The index of the filename in the dataset.
        """
        self._load_hdf5_file()
        self._load_datasets()
        try:
            index = self._filenames.index(filename)
            return index
        except ValueError:
            logging.error(f"Filename '{filename}' not found in the dataset.")
            raise ValueError(f"Filename '{filename}' not found.")

    def close(self) -> None:
        """
        Closes the HDF5 file if it is open.
        """
        with self._data_lock:
            if self._hdf5_file is not None:
                self._hdf5_file.close()
                self._hdf5_file = None
                logging.info("HDF5 file closed.")

    def __del__(self):
        """
        Ensures that the HDF5 file is closed when the ImageDataModel is deleted.
        """
        self.close()
        logging.info("ImageDataModel instance deleted.")
