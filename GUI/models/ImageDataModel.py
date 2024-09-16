# models/ImageDataModel.py

import h5py
import numpy as np
import threading
from typing import Dict, Any, List, Optional, Tuple
import logging

from PIL import Image
from PIL import ImageDraw
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import io


class ImageDataModel:
    """
    ImageDataModel is responsible for loading and providing access to image data, logits, uncertainties, and filenames
    from an HDF5 file. It implements lazy loading, ensures thread safety, and follows the Singleton pattern.
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
            self._uncertainties: Optional[h5py.Dataset] = None
            self._filenames: List[str] = []
            self._data_lock = threading.Lock()
            self._current_pixmap: Optional[QPixmap] = None  # To store the current QPixmap
            self._initialized = True
            logging.info("ImageDataModel instance created.")

    def _load_hdf5_file(self) -> None:
        """
        Opens the HDF5 file using a context manager to ensure it's properly managed.
        """
        if self._hdf5_file is None:
            try:
                self._hdf5_file = h5py.File(self._hdf5_file_path, 'r')
                logging.info("HDF5 file opened successfully.")
            except Exception as e:
                logging.error(f"Failed to open HDF5 file: {e}")
                raise e

    def _load_datasets(self) -> None:
        """
        Lazily loads datasets from the HDF5 file when accessed.
        """
        with self._data_lock:
            if not self._images:
                self._images = self._hdf5_file['rgb_images']
                logging.info("Images dataset loaded.")
            if not self._logits:
                self._logits = self._hdf5_file['logits']
                logging.info("Logits dataset loaded.")
            if not self._uncertainties:
                self._uncertainties = self._hdf5_file['epistemic_uncertainty']
                logging.info("Uncertainties dataset loaded.")
            if not self._filenames:
                filenames_dataset = self._hdf5_file['filenames']
                self._filenames = [name.decode('utf-8') for name in filenames_dataset]
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

        with self._data_lock:
            image = self._images[index]
            logits = self._logits[index]
            uncertainty = self._uncertainties[index]
            filename = self._filenames[index]

        return {
            'image': image,
            'logits': logits,
            'uncertainty': uncertainty,
            'filename': filename
        }

    def create_zoomed_crop(self, image_array: np.ndarray, coord: Tuple[int, int], crop_size: int = 256,
                           zoom_factor: int = 2) -> Tuple[QPixmap, int, int]:
        """
        Creates a zoomed crop of the given image array around the specified coordinate.

        :param image_array: Numpy array of the original image (scaled 0.0-1.0).
        :param coord: Tuple of (row, column) representing the center of the crop.
        :param crop_size: The size of the crop in pixels.
        :param zoom_factor: The factor by which to zoom the cropped image.
        :return: A tuple containing the zoomed QPixmap, x_start, and y_start.
        """
        logging.debug(
            f"Creating zoomed crop at coord: {coord} with crop_size: {crop_size} and zoom_factor: {zoom_factor}")

        # Convert the image from 0.0-1.0 float range to 0-255 uint8
        if image_array.max() <= 1.0:
            logging.debug("Converting image array from float [0.0, 1.0] to uint8 [0, 255].")
            image_array = (image_array * 255).astype(np.uint8)

        height, width, channels = image_array.shape

        # Ensure the array is RGB (3 channels)
        if channels != 3:
            raise ValueError(f"Expected 3 channels for RGB image, but got {channels}")

        # Convert numpy array to QImage
        q_image = QImage(image_array.data, width, height, 3 * width, QImage.Format_RGB888)
        original_pixmap = QPixmap.fromImage(q_image)

        x_center, y_center = coord[1], coord[0]  # (col, row) to (x, y)

        # Calculate the starting points of the crop
        x_start = max(0, x_center - crop_size // 2)
        y_start = max(0, y_center - crop_size // 2)

        # Adjust if crop exceeds image boundaries
        if x_start + crop_size > width:
            x_start = width - crop_size
        if y_start + crop_size > height:
            y_start = height - crop_size

        x_start = max(x_start, 0)
        y_start = max(y_start, 0)

        width_crop = min(crop_size, width - x_start)
        height_crop = min(crop_size, height - y_start)

        logging.debug(f"Crop bounds: x_start={x_start}, y_start={y_start}, width={width_crop}, height={height_crop}")

        # Crop the image
        cropped_pixmap = original_pixmap.copy(x_start, y_start, width_crop, height_crop)

        # Zoom the cropped pixmap
        zoomed_pixmap = cropped_pixmap.scaled(crop_size * zoom_factor, crop_size * zoom_factor, Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)

        logging.debug("Zoomed crop created successfully.")

        return zoomed_pixmap, x_start, y_start

    def get_current_pixmap(self) -> Optional[QPixmap]:
        """
        Retrieves the current QPixmap being displayed.

        :return: The current QPixmap or None if not set.
        """
        with self._data_lock:
            return self._current_pixmap

    def set_current_pixmap(self, pixmap: QPixmap):
        """
        Sets the current QPixmap.

        :param pixmap: The QPixmap to set as current.
        """
        with self._data_lock:
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

        # Convert the numpy array to QPixmap
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
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

    def get_total_entries(self) -> int:
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
