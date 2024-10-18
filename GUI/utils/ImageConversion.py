# utils/ImageConversion.py

from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import logging


def pil_image_to_qpixmap(pil_image: Image.Image) -> QPixmap:
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
            r, g, b = image.split()
            image = Image.merge("RGB", (r, g, b))
            data = image.tobytes("raw", "RGB")
            qimage = QImage(data, image.width, image.height, QImage.Format_RGB888)

        qpixmap = QPixmap.fromImage(qimage)
        logging.debug("Converted PIL Image to QPixmap successfully.")
        return qpixmap
    except Exception as e:
        logging.error(f"Failed to convert PIL Image to QPixmap: {e}")
        raise e


def numpy_to_qpixmap(np_img: np.ndarray) -> QPixmap:
    """
    Converts a NumPy array to QPixmap.

    :param np_img: NumPy array representing the image. Expected shape is (height, width, channels).
    :return: QPixmap object.
    """
    if np_img.ndim == 2:
        # Grayscale image
        height, width = np_img.shape
        bytes_per_line = width
        q_image = QImage(np_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif np_img.ndim == 3:
        height, width, channels = np_img.shape
        if channels == 3:
            # RGB image
            bytes_per_line = 3 * width
            q_image = QImage(np_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        elif channels == 4:
            # RGBA image
            bytes_per_line = 4 * width
            q_image = QImage(np_img.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
    else:
        raise ValueError(f"Unsupported image shape: {np_img.shape}")

    return QPixmap.fromImage(q_image)
