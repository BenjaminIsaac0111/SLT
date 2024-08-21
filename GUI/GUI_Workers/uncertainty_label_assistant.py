from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, Qt, QMetaObject, Q_ARG, QTimer, QLine, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsPixmapItem, \
    QGraphicsScene, QVBoxLayout, QListWidget, QListWidgetItem, QWidget, QSplitter, QSlider
from PIL import Image
import numpy as np
from scipy.special import softmax
import h5py
import sys
import logging
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

import math
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, Qt, QMetaObject, Q_ARG, QTimer, QLine
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsPixmapItem, QGraphicsScene, QVBoxLayout, QListWidget, \
    QListWidgetItem, QWidget, QSplitter
from PIL import Image
import numpy as np
from scipy.special import softmax
import h5py
import sys
import logging
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


class ImageLoaderWorker(QObject):
    finished = pyqtSignal()  # Signal to indicate processing is finished
    imageLoaded = pyqtSignal(object, object, object, object, object)  # Signal to update UI with images

    def __init__(self):
        super().__init__()
        cmap = plt.get_cmap('tab10')
        self.color_map = {i: tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(1, 11)}

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def run(self, image, mask, uncertainty):
        try:
            # Convert the numpy arrays to PIL Images
            image_pil = Image.fromarray((image * 255.).astype('uint8'))

            mask_processed = self.process_mask(mask)
            mask_processed = Image.fromarray(mask_processed)

            # Process the mask and create the overlay
            overlay = self.create_overlay(image_pil, mask_processed)

            # Process the uncertainty and blend with RGB, also drawing arrows at top N uncertain locations
            image_with_arrows, uncertainty_pixmap = self.process_uncertainty(image_pil, uncertainty)

            # Emit the signal to update UI with the processed images
            self.imageLoaded.emit(image_pil, mask_processed, overlay, image_with_arrows, uncertainty_pixmap)
            self.finished.emit()
        except Exception as e:
            logging.error(f"Error during image processing: {e}")

    @staticmethod
    def process_mask(mask):
        try:
            class_probs = softmax(mask, axis=-1)
            class_labels = np.argmax(class_probs, axis=-1)
            height, width = class_labels.shape
            rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
            cmap = plt.get_cmap('tab10')

            for class_label in np.unique(class_labels):
                color = np.array(cmap(class_label)[:3]) * 255  # Convert to [0, 255] range
                rgb_mask[class_labels == class_label] = color.astype(np.uint8)

            return rgb_mask
        except Exception as e:
            logging.error(f"Error during mask processing: {e}")
            return np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)  # Return a blank mask on error

    @staticmethod
    def create_overlay(image, mask, alpha=0.50):
        try:
            mask_rgba = np.array(mask.convert('RGBA'))
            image_rgba = np.array(image.convert('RGBA'))
            mask_alpha = mask_rgba[:, :, 3] / 255.0

            for i in range(3):
                image_rgba[:, :, i] = (
                        (alpha * mask_alpha * mask_rgba[:, :, i]) +
                        (1 - alpha * mask_alpha) * image_rgba[:, :, i]
                )
            return Image.fromarray(image_rgba)
        except Exception as e:
            logging.error(f"Error during overlay creation: {e}")
            return image  # Return the original image on error

    @staticmethod
    def process_uncertainty(image, uncertainty, alpha=1, top_n=32, min_distance=64):
        try:
            uncertainty = np.mean(uncertainty, axis=-1)
            normalized_uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty))
            normalized_uncertainty = (normalized_uncertainty * 255).astype(np.uint8)

            colormap = plt.get_cmap('Spectral_r')
            heatmap = colormap(normalized_uncertainty / 255.0)
            heatmap_rgba = (heatmap[:, :, :3] * 255).astype(np.uint8)
            heatmap_rgba = Image.fromarray(heatmap_rgba).convert('RGBA')

            image_rgba = image.convert('RGBA')

            flat_indices = np.argsort(-normalized_uncertainty, axis=None)[:top_n * 10]
            coords = np.unravel_index(flat_indices, normalized_uncertainty.shape)
            coords = list(zip(*coords))

            selected_coords = []
            for coord in coords:
                if len(selected_coords) >= top_n:
                    break
                too_close = any(math.hypot(coord[0] - sc[0], coord[1] - sc[1]) < min_distance for sc in selected_coords)
                if not too_close:
                    selected_coords.append(coord)

            pixmap = PatchImageViewer.convert_pil_to_pixmap(image_rgba)
            painter = QPainter(pixmap)
            neon_green = QColor(57, 255, 20)
            pen = QPen(neon_green, 3)
            painter.setPen(pen)

            arrow_size = 16

            for y, x in selected_coords:
                if x < 0 or y < 0 or x >= image_rgba.width or y >= image_rgba.height:
                    continue  # Skip invalid coordinates
                painter.drawLine(QLine(x, y, x, y - arrow_size))
                painter.drawLine(QLine(x, y - arrow_size, x - arrow_size // 2, y - arrow_size // 2))
                painter.drawLine(QLine(x, y - arrow_size, x + arrow_size // 2, y - arrow_size // 2))

            painter.end()

            heatmap_pixmap = PatchImageViewer.convert_pil_to_pixmap(heatmap_rgba)
            return pixmap, heatmap_pixmap
        except Exception as e:
            logging.error(f"Error during uncertainty processing: {e}")
            pixmap = PatchImageViewer.convert_pil_to_pixmap(image)  # Return the original image on error
            heatmap_pixmap = PatchImageViewer.convert_pil_to_pixmap(heatmap_rgba)
            return pixmap, heatmap_pixmap


class PatchImageViewer(QWidget):
    def __init__(self, *args, hdf5_file_path):
        super(PatchImageViewer, self).__init__(*args)
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')

        # Initialize attributes before setup_ui()
        self.image_dataset = None
        self.predicted_segmentation = None
        self.uncertainty_dataset = None

        self.initial_load = True
        self.arrows_visible = True  # Flag to track arrow visibility
        self.selected_arrow_index = 0  # Index to track the currently selected arrow
        self.selected_coords = []  # To store the coordinates of selected arrows

        self.current_class = 1  # Default class for labelling
        self.label_radius = 5  # Default label radius

        # Set up the UI components
        self.setup_ui()

        self.thread = QThread()
        self.worker = ImageLoaderWorker()
        self.worker.moveToThread(self.thread)
        self.worker.imageLoaded.connect(self.update_ui_with_images)
        self.thread.start()

        self.load_hdf5_data()

    def setup_ui(self):
        self.main_splitter = QSplitter(Qt.Horizontal, self)
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(self.on_file_selected)
        self.main_splitter.addWidget(self.file_list_widget)

        self.setup_graphics_view()

        layout = QVBoxLayout(self)
        layout.addWidget(self.main_splitter)
        self.setLayout(layout)

        # Slider for adjusting label radius
        self.label_slider = QSlider(Qt.Horizontal, self)
        self.label_slider.setRange(1, 32)
        self.label_slider.setValue(self.label_radius)
        self.label_slider.valueChanged.connect(self.update_label_radius)
        layout.addWidget(self.label_slider)

        self.main_splitter.splitterMoved.connect(self.scale_image_to_view)

    def setup_graphics_view(self):
        self.graphics_view = QGraphicsView(self)
        self.graphics_view.setFocusPolicy(Qt.StrongFocus)  # Ensure QGraphicsView can take focus

        # Initialize the scene before assigning it to the graphics view
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)

        self.image_item = QGraphicsPixmapItem()
        self.mask_item = QGraphicsPixmapItem()
        self.overlay_item = QGraphicsPixmapItem()
        self.uncertainty_item = QGraphicsPixmapItem()

        self.scene.addItem(self.image_item)
        self.scene.addItem(self.mask_item)
        self.scene.addItem(self.overlay_item)
        self.scene.addItem(self.uncertainty_item)

        self.main_splitter.addWidget(self.graphics_view)

        self.graphics_view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.graphics_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
    def update_label_radius(self, value):
        self.label_radius = value
        logging.info(f"Label radius updated to {value}")

    def calculate_global_uncertainty(self, i):
        return np.mean(self.uncertainty_dataset[i])

    def load_hdf5_data(self):
        self.image_dataset = self.hdf5_file['rgb_images']
        self.predicted_segmentation = self.hdf5_file['logits']
        self.uncertainty_dataset = self.hdf5_file['epistemic_uncertainty']
        filenames_dataset = self.hdf5_file['filenames']

        global_uncertainties = [self.calculate_global_uncertainty(i) for i in range(len(filenames_dataset))]

        sorted_indices = np.argsort(global_uncertainties)[::-1]  # Sort in descending order

        for idx in sorted_indices:
            filename = filenames_dataset[idx].decode('utf-8')
            uncertainty = global_uncertainties[idx]
            item_text = f"{filename} - Uncertainty: {uncertainty:.4f}"
            item = QListWidgetItem(item_text)
            self.file_list_widget.addItem(item)

        if len(filenames_dataset) > 0:
            self.file_list_widget.setCurrentRow(0)
            self.on_file_selected(self.file_list_widget.item(0))

    def on_file_selected(self, item):
        index = self.file_list_widget.row(item)
        logging.info(f"Selected file: {item.text()} (index {index})")

        image = self.image_dataset[index]
        mask = self.predicted_segmentation[index]
        uncertainty = self.uncertainty_dataset[index]

        QMetaObject.invokeMethod(self.worker, "run", Q_ARG(np.ndarray, image), Q_ARG(np.ndarray, mask),
                                 Q_ARG(np.ndarray, uncertainty))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.arrows_visible = not self.arrows_visible
            self.update_ui_with_images(self.current_image, self.current_mask, self.current_overlay,
                                       self.current_image_with_arrows, self.current_uncertainty_pixmap)
        elif event.key() == Qt.Key_Right:
            if self.selected_coords:
                self.selected_arrow_index = (self.selected_arrow_index + 1) % len(self.selected_coords)
                self.highlight_selected_arrow()
        elif event.key() == Qt.Key_Left:
            if self.selected_coords:
                self.selected_arrow_index = (self.selected_arrow_index - 1) % len(self.selected_coords)
                self.highlight_selected_arrow()
        elif event.key() == Qt.Key_Up:
            self.label_radius = min(self.label_radius + 1, self.label_slider.maximum())
            self.label_slider.setValue(self.label_radius)  # Sync the slider with the new radius
        elif event.key() == Qt.Key_Down:
            self.label_radius = max(self.label_radius - 1, self.label_slider.minimum())
            self.label_slider.setValue(self.label_radius)  # Sync the slider with the new radius
        elif event.key() in range(Qt.Key_1, Qt.Key_9 + 1):
            self.current_class = event.key() - Qt.Key_0  # Assign class based on the number key pressed
            self.label_current_arrow()
        else:
            super().keyPressEvent(event)

    def highlight_selected_arrow(self):
        if not self.selected_coords:
            return

        # Redraw arrows with the selected one highlighted in dark blue and larger
        self.update_ui_with_images(self.current_image, self.current_mask, self.current_overlay,
                                   self.current_image_with_arrows, self.current_uncertainty_pixmap)

        painter = QPainter(self.current_image_with_arrows)
        dark_blue = QColor(0, 0, 139)  # Dark blue color
        pen = QPen(dark_blue, 5)  # Larger pen size for selected arrow
        painter.setPen(pen)

        # Get the coordinates of the selected arrow
        coord = self.selected_coords[self.selected_arrow_index]
        x, y = coord[1], coord[0]

        # Draw the selected arrow larger and in dark blue
        arrow_size = 24
        painter.drawLine(QLine(x, y, x, y - arrow_size))
        painter.drawLine(QLine(x, y - arrow_size, x - arrow_size // 2, y - arrow_size // 2))
        painter.drawLine(QLine(x, y - arrow_size, x + arrow_size // 2, y - arrow_size // 2))

        painter.end()

        self.update_ui_with_images(self.current_image, self.current_mask, self.current_overlay,
                                   self.current_image_with_arrows, self.current_uncertainty_pixmap)

    def label_current_arrow(self):
        if not self.selected_coords:
            return
        coord = self.selected_coords[self.selected_arrow_index]
        x, y = coord[1], coord[0]

        # Draw the label spot
        painter = QPainter(self.current_image_with_arrows)
        pen = QPen(QColor(255, 0, 0))  # Red color for the label spot
        painter.setPen(pen)
        painter.setBrush(QColor(255, 0, 0))
        painter.drawEllipse(QPoint(x, y), self.label_radius, self.label_radius)
        painter.end()

        self.update_ui_with_images(self.current_image, self.current_mask, self.current_overlay,
                                   self.current_image_with_arrows, self.current_uncertainty_pixmap)

    def update_ui_with_images(self, image, mask, overlay, image_with_arrows, uncertainty_pixmap):
        self.current_image = image
        self.current_mask = mask
        self.current_overlay = overlay
        self.current_image_with_arrows = image_with_arrows
        self.current_uncertainty_pixmap = uncertainty_pixmap

        if self.arrows_visible:
            image_pixmap = image_with_arrows
        else:
            image_pixmap = self.convert_pil_to_pixmap(image)

        mask_pixmap = self.convert_pil_to_pixmap(mask)
        overlay_pixmap = self.convert_pil_to_pixmap(overlay)

        self.image_item.setPixmap(image_pixmap)
        self.mask_item.setPixmap(mask_pixmap)
        self.overlay_item.setPixmap(overlay_pixmap)
        self.uncertainty_item.setPixmap(uncertainty_pixmap)

        self.mask_item.setPos(image_pixmap.width(), 0)
        self.overlay_item.setPos(image_pixmap.width() + mask_pixmap.width(), 0)
        self.uncertainty_item.setPos(self.overlay_item.pos().x() + self.overlay_item.pixmap().width(), 0)

        self.scale_image_to_view()

        if self.initial_load:
            self.graphics_view.update()
            self.initial_load = False

    def scale_image_to_view(self):
        self.graphics_view.resetTransform()

        rect = self.scene.itemsBoundingRect()

        if rect.width() == 0 or rect.height() == 0:
            return

        scaleFactorX = self.graphics_view.width() / rect.width() if rect.width() > 0 else 1
        scaleFactorY = self.graphics_view.height() / rect.height() if rect.height() > 0 else 1
        scaleFactor = min(scaleFactorX, scaleFactorY)

        self.graphics_view.scale(scaleFactor, scaleFactor)
        self.graphics_view.centerOn(rect.center())

    def resizeEvent(self, event):
        self.scale_image_to_view()
        super().resizeEvent(event)

    @staticmethod
    def convert_pil_to_pixmap(pil_image):
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

        image_np = np.array(pil_image)
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)

        height, width, channels = image_np.shape
        bytes_per_line = channels * width
        q_image = QImage(image_np.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
        return QPixmap.fromImage(q_image)

# Main entry for testing
if __name__ == '__main__':
    app = QApplication(sys.argv)
    hdf5_file_path = r"Z:\PathologyData\unet_local_devel.h5_COLLECTED_UNCERTAINTIES.h5"  # Update with your HDF5 file path
    viewer = PatchImageViewer(hdf5_file_path=hdf5_file_path)
    viewer.show()
    logging.info("Application started")
    sys.exit(app.exec_())
