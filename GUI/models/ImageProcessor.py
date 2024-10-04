# models/ImageProcessor.py

import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from typing import Optional, Tuple, Dict
import logging

from GUI.utils.ImageConversion import pil_image_to_qpixmap


class ImageProcessor:
    """
    ImageProcessor handles all image processing tasks, including mask processing, overlay creation,
    uncertainty normalization, and heatmap generation.
    """

    def __init__(self, colormap_name: str = 'tab10', heatmap_cmap_name: str = 'Spectral_r'):
        """
        Initializes the ImageProcessor with specified colormaps.

        :param colormap_name: Name of the matplotlib colormap to use for mask processing.
        :param heatmap_cmap_name: Name of the matplotlib colormap to use for heatmap generation.
        """
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

    def process_image_data(self, image_array: np.ndarray, logits: np.ndarray, uncertainty: np.ndarray) -> Dict[str, Image.Image]:
        """
        Processes image data and returns a dictionary of processed images.

        :param image_array: A numpy array of the original image.
        :param logits: A numpy array of logits.
        :param uncertainty: A numpy array of uncertainty values.
        :return: A dictionary containing 'image', 'mask', 'overlay', 'heatmap' as PIL Images.
        """
        image = Image.fromarray((image_array * 255).astype(np.uint8)).convert('RGB')
        mask = self.process_mask(logits)
        overlay = self.create_overlay(image, mask)
        normalized_uncertainty = self.normalize_uncertainty(uncertainty)
        heatmap = self.create_heatmap(normalized_uncertainty)
        logging.info("Image data processed.")
        return {
            'image': pil_image_to_qpixmap(image),
            'mask': pil_image_to_qpixmap(mask),
            'overlay': pil_image_to_qpixmap(overlay),
            'heatmap': pil_image_to_qpixmap(heatmap)
        }

    def extract_crop(self, image: np.ndarray, coord: Tuple[int, int], crop_size: int = 256,
                     zoom_factor: int = 2) -> QImage:
        """
        Extracts a zoomed-in crop from the image at the specified coordinate.

        :param image: A numpy array representing the image.
        :param coord: A tuple (row, column) indicating the center of the crop.
        :param crop_size: Size of the crop in pixels.
        :param zoom_factor: Factor by which to zoom the crop.
        :return: A QImage object of the zoomed-in crop.
        """
        row, col = coord
        half_size = crop_size // 2

        # Define crop boundaries
        start_row = max(row - half_size, 0)
        end_row = min(row + half_size, image.shape[0])
        start_col = max(col - half_size, 0)
        end_col = min(col + half_size, image.shape[1])

        # Extract the crop
        crop = image[start_row:end_row, start_col:end_col]

        # Resize the crop based on zoom_factor
        resized_crop = self.resize_image(crop, zoom_factor)

        # Convert to QImage
        qimage = self.numpy_to_qimage(resized_crop)
        return qimage

    @staticmethod
    def resize_image(image: np.ndarray, zoom_factor: int) -> np.ndarray:
        """
        Resizes the image by the specified zoom factor.

        :param image: A numpy array representing the image.
        :param zoom_factor: Factor by which to resize the image.
        :return: Resized numpy array.
        """
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        new_size = (pil_image.width * zoom_factor, pil_image.height * zoom_factor)
        resized_pil = pil_image.resize(new_size)
        resized_np = np.array(resized_pil)
        return resized_np

    @staticmethod
    def numpy_to_qimage(image: np.ndarray) -> QImage:
        """
        Converts a numpy array to QImage.

        :param image: A numpy array representing the image.
        :return: QImage object.
        """
        if image.dtype != np.uint8:
            image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)

        if len(image.shape) == 2:
            # Grayscale
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8)
        elif image.shape[2] == 3:
            # RGB
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        elif image.shape[2] == 4:
            # RGBA
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGBA8888)
        else:
            raise ValueError("Unsupported image format for conversion to QImage.")
        return qimage
