# models/ImageProcessor.py

import logging
from typing import Tuple, Dict

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap

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

    def extract_crop(
            self,
            image: np.ndarray,
            coord: Tuple[int, int],
            crop_size: int = 256,
            zoom_factor: int = 2
    ) -> Tuple[QPixmap, Tuple[int, int]]:
        """
        Extracts a zoomed-in crop from the RGB image at the specified coordinate without padding.
        Handles image data in float (0-1) or uint8 (0-255) formats.
        Draws a red circle at the position of the coordinate within the zoomed crop.

        :param image: A numpy array representing the RGB image (H x W x 3).
                      Data type can be float (0-1) or uint8 (0-255).
        :param coord: A tuple (row, column) indicating the center of the crop.
        :param crop_size: Desired size of the crop in pixels (crop_size x crop_size).
        :param zoom_factor: Factor by which to zoom the crop.
        :return: A tuple containing the QPixmap object of the zoomed-in crop with annotation
                 and the (x, y) position of the coordinate within the zoomed crop.
        """
        row, col = coord
        original_height, original_width = image.shape[:-1]
        half_crop = crop_size // 2

        # Validate coordinates
        if not (0 <= row < original_height) or not (0 <= col < original_width):
            raise ValueError("Coordinate is outside the image boundaries.")

        # Calculate crop bounds
        x_start = max(0, col - half_crop)
        y_start = max(0, row - half_crop)

        if x_start + crop_size > original_width:
            x_start = original_width - crop_size
        if y_start + crop_size > original_height:
            y_start = original_height - crop_size

        x_start = max(x_start, 0)
        y_start = max(y_start, 0)

        width_crop = min(crop_size, original_width - x_start)
        height_crop = min(crop_size, original_height - y_start)

        arrow_rel_x = col - x_start
        arrow_rel_y = row - y_start

        logging.debug(
            f"Calculated crop bounds: x_start={x_start}, y_start={y_start}, width={width_crop}, height={height_crop}")
        logging.debug(f"Arrow relative position within crop: ({arrow_rel_x}, {arrow_rel_y})")

        # Extract the crop
        crop = image[y_start:y_start + height_crop, x_start:x_start + width_crop]

        # Handle data type and scaling
        if np.issubdtype(crop.dtype, np.floating):
            # Assuming the data is in [0, 1], scale to [0, 255]
            crop = np.clip(crop, 0.0, 1.0)  # Ensure values are within [0,1]
            crop = (crop * 255).astype(np.uint8)
        elif np.issubdtype(crop.dtype, np.integer):
            # If already in integer type, ensure it's uint8
            if crop.dtype != np.uint8:
                # Convert to uint8, scaling if necessary
                info = np.iinfo(crop.dtype)
                if info.max > 255:
                    # Scale down
                    crop = (crop / info.max * 255).astype(np.uint8)
                else:
                    crop = crop.astype(np.uint8)
        else:
            raise ValueError("Unsupported image data type. Expected float or integer type.")

        # Ensure the data is contiguous
        if not crop.flags['C_CONTIGUOUS']:
            crop = np.ascontiguousarray(crop)

        # Create QImage from the NumPy array
        height, width, channels = crop.shape
        bytes_per_line = 3 * width  # 3 bytes per pixel for RGB

        q_image = QImage(
            crop.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        ).copy()  # Make a deep copy to ensure data integrity

        # Calculate the position of the original coordinate within the crop
        pos_x = arrow_rel_x
        pos_y = arrow_rel_y

        # Scale the position by the zoom factor
        pos_x_zoomed = pos_x * zoom_factor
        pos_y_zoomed = pos_y * zoom_factor

        logging.debug(f"Arrow scaled position: ({pos_x_zoomed}, {pos_y_zoomed})")
        logging.debug(f"Crop Size: ({width}, {height})")
        logging.debug(f"Zoomed Image Size: ({width * zoom_factor}, {height * zoom_factor})")

        # Apply zoom by scaling the QImage
        zoomed_qimage = q_image.scaled(
            width * zoom_factor,
            height * zoom_factor,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        )

        # Draw the annotation directly on the zoomed image
        painter = QPainter(zoomed_qimage)
        pen = QPen(QColor(0, 255, 0))  # Red color for the circle
        pen.setWidth(5)  # Thickness of the circle outline
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing)

        # Ensure the position is within the zoomed image bounds
        pos_x_zoomed = min(max(pos_x_zoomed, 0), zoomed_qimage.width() - 1)
        pos_y_zoomed = min(max(pos_y_zoomed, 0), zoomed_qimage.height() - 1)

        # Define a smaller radius for precise annotation
        radius = max(8, min(zoomed_qimage.width(), zoomed_qimage.height()) // 40)

        # Draw the unfilled circle
        painter.drawEllipse(QPoint(int(pos_x_zoomed), int(pos_y_zoomed)), radius, radius)

        painter.end()

        # Convert QImage to QPixmap for display
        zoomed_pixmap = QPixmap.fromImage(zoomed_qimage)

        return zoomed_pixmap, (int(pos_x_zoomed), int(pos_y_zoomed))

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
        resized_pil = pil_image.resize(new_size, Image.BICUBIC)
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
