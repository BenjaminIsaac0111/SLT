from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PIL import Image
import numpy as np


class ImageLoaderWorker(QObject):
    finished = pyqtSignal()  # Signal to indicate processing is finished
    imageLoaded = pyqtSignal(object, object, object)  # Signal to update UI with images

    def __init__(self):
        super().__init__()
        self.color_map = {
            0: (0, 0, 0),  # Black (for background)
            1: (31, 119, 180),  # Blue
            2: (255, 127, 14),  # Orange
            3: (44, 160, 44),  # Green
            4: (214, 39, 40),  # Red
            5: (148, 103, 189),  # Purple
            6: (140, 86, 75),  # Brown
            7: (227, 119, 194),  # Pink
            8: (127, 127, 127),  # Gray
            9: (188, 189, 34),  # Olive
            10: (23, 190, 207)  # Cyan
        }

    @pyqtSlot(str)
    def run(self, image_path=None):
        patch = Image.open(image_path)
        width, height = patch.size

        # Split the image
        image = patch.crop((0, 0, width // 2, height))
        mask = patch.crop((width // 2, 0, width, height))
        mask = self.process_mask(mask)
        overlay = self.create_overlay(image, mask)
        self.imageLoaded.emit(image, mask, overlay)
        self.finished.emit()

    def process_mask(self, mask_image):
        # Convert the blue channel to a numpy array
        mask_array = np.array(mask_image)[:, :, 2]  # Extracting the blue channel

        # Create an empty array for the colored mask
        colored_mask_array = np.zeros((*mask_array.shape, 3), dtype=np.uint8)

        # Map each class to its corresponding color
        for class_label, color in self.color_map.items():
            colored_mask_array[mask_array == class_label] = color

        # Convert the numpy array back to a PIL image
        colored_mask = Image.fromarray(colored_mask_array)
        return colored_mask

    @staticmethod
    def create_overlay(image, mask, alpha=0.50):
        mask_rgba = np.array(mask.convert('RGBA'))
        image_rgba = np.array(image.convert('RGBA'))
        mask_alpha = mask_rgba[:, :, 3] / 255.0

        for i in range(3):
            image_rgba[:, :, i] = (
                    (alpha * mask_alpha * mask_rgba[:, :, i]) +
                    (1 - alpha * mask_alpha) * image_rgba[:, :, i]
            )
        return Image.fromarray(image_rgba)
