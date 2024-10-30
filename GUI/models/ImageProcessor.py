# models/ImageProcessor.py
import hashlib
import logging
from typing import Tuple, Dict

import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap

from GUI.models.CacheManager import CacheManager


class ImageProcessor:
    """
    ImageProcessor handles all image processing tasks, including mask processing, overlay creation,
    uncertainty normalization, and heatmap generation.
    """

    def __init__(self, colormap_name: str = 'tab10', heatmap_cmap_name: str = 'Spectral_r', cache_size: int = 250):
        """
        Initializes the ImageProcessor with specified colormaps.

        :param colormap_name: Name of the matplotlib colormap to use for mask processing.
        :param heatmap_cmap_name: Name of the matplotlib colormap to use for heatmap generation.
        """
        self.cache = CacheManager(cache_size=cache_size)  # Use CacheManager instead of LRUCache
        self.colormap_name = colormap_name
        self.heatmap_cmap_name = heatmap_cmap_name
        self.colormap: Colormap = plt.get_cmap(self.colormap_name)
        self.heatmap_colormap: Colormap = plt.get_cmap(self.heatmap_cmap_name)
        self.class_color_map: Dict[int, Tuple[int, int, int]] = self._generate_class_color_map()
        logging.info("ImageProcessor initialized with colormap '%s' and heatmap colormap '%s'.",
                     self.colormap_name, self.heatmap_cmap_name)

    def _generate_class_color_map(self) -> Dict[int, Tuple[int, int, int]]:
        """
        Generates a color map for classes based on the specified colormap.

        :return: A dictionary mapping class indices to RGB color tuples.
        """
        class_color_map = {}
        num_colors = self.colormap.N
        for i in range(num_colors):
            color = self.colormap(i)[:3]  # Get RGB values
            class_color_map[i] = tuple(int(c * 255) for c in color)
        logging.debug("Class color map generated with %d colors.", num_colors)
        return class_color_map

    def process_mask(self, logits: np.ndarray) -> Image.Image:
        """
        Converts logits to a colored mask image.

        :param logits: A 3D numpy array of logits.
        :return: A PIL Image object representing the colored mask.
        """
        class_probs = self._softmax(logits)
        class_labels = np.argmax(class_probs, axis=-1)
        height, width = class_labels.shape
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_label in np.unique(class_labels):
            color = self.class_color_map.get(class_label, (255, 255, 255))  # Default to white
            rgb_mask[class_labels == class_label] = color

        mask_image = Image.fromarray(rgb_mask)
        logging.debug("Mask image processed with shape %s.", mask_image.size)
        return mask_image

    def create_overlay(self, image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
        """
        Creates an overlay by blending the original image with the mask.

        :param image: A PIL Image object of the original image.
        :param mask: A PIL Image object of the mask.
        :param alpha: The transparency level of the mask overlay.
        :return: A PIL Image object of the overlay.
        """
        image_rgba = image.convert('RGBA')
        mask_rgba = mask.convert('RGBA')

        blended_image = Image.blend(image_rgba, mask_rgba, alpha=alpha)
        logging.debug("Overlay image created with alpha %f.", alpha)
        return blended_image

    def normalize_uncertainty(self, uncertainty: np.ndarray) -> np.ndarray:
        """
        Normalizes the uncertainty map to the range [0, 255].

        :param uncertainty: A 2D or 3D numpy array of uncertainty values.
        :return: A 2D numpy array of normalized uncertainty values.
        """
        if uncertainty.ndim == 3:
            uncertainty = np.mean(uncertainty, axis=-1)
        min_val = np.min(uncertainty)
        max_val = np.max(uncertainty)
        normalized_uncertainty = (uncertainty - min_val) / (max_val - min_val + 1e-8)
        normalized_uncertainty_uint8 = (normalized_uncertainty * 255).astype(np.uint8)
        logging.debug("Uncertainty map normalized to range [0, 255].")
        return normalized_uncertainty_uint8

    def create_heatmap(self, normalized_uncertainty: np.ndarray) -> Image.Image:
        """
        Creates a heatmap image from the normalized uncertainty map.

        :param normalized_uncertainty: A 2D numpy array of normalized uncertainty values.
        :return: A PIL Image object of the heatmap.
        """
        heatmap_array = self.heatmap_colormap(normalized_uncertainty / 255.0)
        heatmap_rgb = (heatmap_array[:, :, :3] * 255).astype(np.uint8)
        heatmap_image = Image.fromarray(heatmap_rgb).convert('RGBA')
        logging.debug("Heatmap image created.")
        return heatmap_image

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """
        Applies softmax function to logits.

        :param logits: A numpy array of logits.
        :return: A numpy array of softmax probabilities.
        """
        e_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        softmax_probs = e_logits / np.sum(e_logits, axis=-1, keepdims=True)
        return softmax_probs

    def process_image_data(
            self, image_array: np.ndarray, logits: np.ndarray, uncertainty: np.ndarray
    ) -> Dict[str, QPixmap]:
        """
        Processes image data and returns a dictionary of processed images.

        :param image_array: A numpy array of the original image.
        :param logits: A numpy array of logits.
        :param uncertainty: A numpy array of uncertainty values.
        :return: A dictionary containing 'image', 'mask', 'overlay', 'heatmap' as QPixmaps.
        """
        image = Image.fromarray((image_array * 255).astype(np.uint8)).convert('RGB')
        mask = self.process_mask(logits)
        overlay = self.create_overlay(image, mask)
        normalized_uncertainty = self.normalize_uncertainty(uncertainty)
        heatmap = self.create_heatmap(normalized_uncertainty)
        logging.info("Image data processed.")
        return {
            'image': self.pil_image_to_qpixmap(image),
            'mask': self.pil_image_to_qpixmap(mask),
            'overlay': self.pil_image_to_qpixmap(overlay),
            'heatmap': self.pil_image_to_qpixmap(heatmap)
        }

    def _generate_cache_key(
            self, image: np.ndarray, coord: Tuple[int, int], crop_size: int, zoom_factor: int
    ) -> str:
        """
        Generates a cache key by hashing the image data and combining it with the other parameters.

        :param image: A numpy array representing the image.
        :param coord: A tuple (row, column) indicating the center of the crop.
        :param crop_size: Size of the crop.
        :param zoom_factor: Zoom factor for the crop.
        :return: A unique cache key as a string.
        """
        # Use image data's hash (you can alternatively use image.tobytes() for larger data)
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        # Create a key with image hash, coordinates, crop size, and zoom factor
        cache_key = f"{image_hash}_{coord[0]}_{coord[1]}_{crop_size}_{zoom_factor}"
        return cache_key

    def extract_crop_data(
            self,
            image: np.ndarray,
            coord: Tuple[int, int],
            crop_size: int = 256,
            zoom_factor: int = 2
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extracts a zoomed-in crop from the RGB image at the specified coordinate without padding.
        Handles image data in float (0-1) or uint8 (0-255) formats.

        :param image: A numpy array representing the RGB image (H x W x 3).
        :param coord: A tuple (row, column) indicating the center of the crop.
        :param crop_size: Desired size of the crop in pixels (crop_size x crop_size).
        :param zoom_factor: Factor by which to zoom the crop.
        :return: A tuple containing the processed image as a NumPy array
                 and the (x, y) position of the coordinate within the zoomed crop.
        """
        # Generate a cache key using a hash of the image and other parameters
        cache_key = self._generate_cache_key(image, coord, crop_size, zoom_factor)

        # Check if the result is in the cache
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logging.debug("Cache hit for crop")
            return cached_result

        # Ensure row and col are integers
        row, col = map(int, coord)
        original_height, original_width = image.shape[:-1]
        half_crop = crop_size // 2

        x_start = max(0, col - half_crop)
        y_start = max(0, row - half_crop)

        if x_start + crop_size > original_width:
            x_start = original_width - crop_size
        if y_start + crop_size > original_height:
            y_start = original_height - crop_size

        width_crop = min(crop_size, original_width - x_start)
        height_crop = min(crop_size, original_height - y_start)

        # Ensure indices are integers
        x_start = int(x_start)
        y_start = int(y_start)
        width_crop = int(width_crop)
        height_crop = int(height_crop)

        crop = image[y_start:y_start + height_crop, x_start:x_start + width_crop]

        # Ensure crop is uint8 for further processing
        if np.issubdtype(crop.dtype, np.floating):
            crop = (crop * 255).astype(np.uint8)

        pil_image = Image.fromarray(crop)
        new_size = (crop.shape[1] * zoom_factor, crop.shape[0] * zoom_factor)
        zoomed_pil = pil_image.resize(new_size, Image.BICUBIC)
        zoomed_crop = np.array(zoomed_pil)

        # Calculate the position of the original coordinate within the zoomed crop
        arrow_rel_x = col - x_start
        arrow_rel_y = row - y_start
        pos_x_zoomed = arrow_rel_x * zoom_factor
        pos_y_zoomed = arrow_rel_y * zoom_factor

        result = (zoomed_crop, (int(pos_x_zoomed), int(pos_y_zoomed)))

        # Store the result in the cache
        self.cache.set(cache_key, result)
        logging.debug("Cache miss, crop processed and stored.")

        return result

    def numpy_to_qimage(self, image: np.ndarray) -> QImage:
        """
        Converts a numpy array to QImage.

        :param image: A numpy array representing the image.
        :return: QImage object.
        """
        if image.dtype != np.uint8:
            image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)

        if len(image.shape) == 2:
            # Grayscale
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0],
                            QImage.Format_Grayscale8).copy()
        elif image.shape[2] == 3:
            # RGB
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0],
                            QImage.Format_RGB888).copy()
        elif image.shape[2] == 4:
            # RGBA
            qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0],
                            QImage.Format_RGBA8888).copy()
        else:
            raise ValueError("Unsupported image format for conversion to QImage.")
        return qimage

    def pil_image_to_qpixmap(self, pil_image: Image.Image) -> QPixmap:
        """
        Converts a PIL Image to QPixmap.

        :param pil_image: PIL Image to convert.
        :return: QPixmap representation of the image.
        """
        try:
            if pil_image.mode == "RGB":
                r, g, b = pil_image.split()
                image = Image.merge("RGB", (r, g, b))
                data = image.tobytes("raw", "RGB")
                qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)
            elif pil_image.mode == "RGBA":
                r, g, b, a = pil_image.split()
                image = Image.merge("RGBA", (r, g, b, a))
                data = image.tobytes("raw", "RGBA")
                qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
            else:
                # Convert to RGB if in a different mode
                image = pil_image.convert("RGB")
                data = image.tobytes("raw", "RGB")
                qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)

            qpixmap = QPixmap.fromImage(qimage)
            logging.debug("Converted PIL Image to QPixmap successfully.")
            return qpixmap
        except Exception as e:
            logging.error(f"Failed to convert PIL Image to QPixmap: {e}")
            raise e
