from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, Qt, QMetaObject, Q_ARG, QTimer, QLine, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsPixmapItem, \
    QGraphicsScene, QVBoxLayout, QListWidget, QListWidgetItem, QWidget, QSplitter, QSlider
from PIL import Image
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.special import softmax
import h5py
import sys
import logging
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

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
    imageLoaded = pyqtSignal(object, object, object, object, object, object)  # Signal to update UI with images

    def __init__(self):
        super().__init__()
        cmap = plt.get_cmap('tab10')
        self.color_map = {i: tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(1, 11)}

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def run(self, image, mask, uncertainty):
        try:
            # Convert numpy arrays to PIL Images
            image_pil = self.convert_to_pil(image)
            mask_pil = self.convert_to_pil(self.process_mask(mask))

            # Process uncertainty and get images with arrows
            (
                image_with_arrows,
                mask_with_arrows,
                overlay_with_arrows,
                uncertainty_with_arrows,
                arrow_coords
            ) = self.process_image(image_pil, mask_pil, uncertainty)

            # Emit signal to update UI with processed images
            self.imageLoaded.emit(image_pil, mask_pil, uncertainty, overlay_with_arrows, image_with_arrows,
                                  uncertainty_with_arrows)
            self.finished.emit()
        except Exception as e:
            logging.error(f"Error during image processing: {e}")

    @staticmethod
    def convert_to_pil(array):
        """Convert a numpy array to a PIL Image."""
        return Image.fromarray((array * 255).astype('uint8'))

    @staticmethod
    def process_mask(mask):
        """Process the mask to create an RGB representation."""
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
    def create_overlay(image, mask, alpha=0.5):
        """Create an overlay of the mask on the image."""
        try:
            # Ensure both images are in RGBA format
            image_rgba = image.convert('RGBA')
            mask_rgba = mask.convert('RGBA')

            # Convert to numpy arrays for easier manipulation
            image_np = np.array(image_rgba)
            mask_np = np.array(mask_rgba)

            # Blend the mask with the image using the alpha value
            blended_np = image_np.copy()
            for i in range(3):  # For RGB channels
                blended_np[:, :, i] = (
                        alpha * mask_np[:, :, i] + (1 - alpha) * image_np[:, :, i]
                )

            # Combine alpha channels
            blended_np[:, :, 3] = np.maximum(mask_np[:, :, 3] * alpha, image_np[:, :, 3])

            # Convert back to an Image
            blended_image = Image.fromarray(blended_np, 'RGBA')
            return blended_image
        except Exception as e:
            logging.error(f"Error during overlay creation: {e}")
            return image  # Return the original image on error

    def process_image(self, image, mask, uncertainty, top_n=16, min_distance=16):
        """Process the uncertainty and return images with and without arrows."""
        uncertainty = self.normalize_uncertainty(uncertainty)

        # Generate the heatmap from the uncertainty
        heatmap_rgba = self.create_heatmap(uncertainty)

        # Identify significant coordinates based on uncertainty
        selected_coords = self.identify_significant_coords(uncertainty, top_n, min_distance)
        logging.info(f"Generated {len(selected_coords)} significant coordinates: {selected_coords}")

        # Create images without arrows
        overlay = self.create_overlay(image, mask)

        # Draw arrows on the images
        image_with_arrows = self.draw_arrows(image.copy(), selected_coords)
        mask_with_arrows = self.draw_arrows(mask.copy(), selected_coords)
        overlay_with_arrows = self.draw_arrows(overlay.copy(), selected_coords)
        heatmap_with_arrows = self.draw_arrows(heatmap_rgba.copy(), selected_coords)

        # Return images with and without arrows
        return (
            image,
            mask,
            overlay,
            heatmap_rgba,
            image_with_arrows,
            mask_with_arrows,
            overlay_with_arrows,
            heatmap_with_arrows,
            selected_coords
        )

    @staticmethod
    def normalize_uncertainty(uncertainty):
        """Normalize the uncertainty array to the [0, 255] range."""
        uncertainty = np.mean(uncertainty, axis=-1)
        normalized_uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty))
        return (normalized_uncertainty * 255).astype(np.uint8)

    @staticmethod
    def create_heatmap(normalized_uncertainty):
        """Create a heatmap from normalized uncertainty."""
        colormap = plt.get_cmap('Spectral_r')
        heatmap = colormap(normalized_uncertainty / 255.0)
        return Image.fromarray((heatmap[:, :, :3] * 255).astype(np.uint8)).convert('RGBA')

    @staticmethod
    def identify_significant_coords(normalized_uncertainty, top_n, min_distance):
        """Identify significant coordinates based on the uncertainty map."""
        # Convert uncertainty to float32 to prevent overflow
        normalized_uncertainty = normalized_uncertainty.astype(np.float32)

        local_max = maximum_filter(normalized_uncertainty, size=min_distance) == normalized_uncertainty
        coords = np.column_stack(np.nonzero(local_max))

        if len(coords) > 0:
            clustering = DBSCAN(eps=min_distance, min_samples=1).fit(coords)
            cluster_centers = [coords[clustering.labels_ == cluster_id].mean(axis=0).astype(int)
                               for cluster_id in np.unique(clustering.labels_)]
            cluster_centers = sorted(cluster_centers, key=lambda x: -float(normalized_uncertainty[x[0], x[1]]))
            logging.info(f"Identified {len(cluster_centers)} cluster centers.")
            return cluster_centers[:top_n]
        logging.warning("No significant coordinates identified.")
        return []

    @staticmethod
    def draw_arrows(image, arrow_coords, arrow_color=(57, 255, 20), arrow_size=24, selected_arrow=None):
        if isinstance(image, QPixmap):
            # Convert QPixmap to QImage and then to PIL Image
            image = PatchImageViewer.pixmap_to_pil_image(image)

        image_rgba = image  # Ensure it's RGBA for PIL
        q_image = PatchImageViewer.convert_pil_to_pixmap(image_rgba)
        painter = QPainter(q_image)

        # Default pen for non-selected arrows
        pen_color = QColor(*arrow_color)
        pen = QPen(pen_color, 3)
        painter.setPen(pen)

        for y, x in arrow_coords:
            if selected_arrow is not None and np.array_equal((y, x), selected_arrow):
                painter.setPen(QPen(QColor(255, 0, 0), 4))  # Highlight the selected arrow in red
            else:
                painter.setPen(pen)

            if 0 <= x < image_rgba.width and 0 <= y < image_rgba.height:
                wing_left = QPoint(x - arrow_size // 2, y - arrow_size // 2)
                wing_right = QPoint(x + arrow_size // 2, y - arrow_size // 2)
                arrow_tip = QPoint(x, y - arrow_size)

                painter.drawLine(QLine(QPoint(x, y), arrow_tip))
                painter.drawLine(QLine(QPoint(x, y), wing_left))
                painter.drawLine(QLine(QPoint(x, y), wing_right))

        painter.end()
        return q_image


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
        self.label_radius = 6  # Default label radius

        # Set focus policy to ensure key events are captured
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        # Set up the UI components
        self.setup_ui()

        self.thread = QThread()
        self.worker = ImageLoaderWorker()
        self.worker.moveToThread(self.thread)
        self.worker.imageLoaded.connect(self.update_ui_with_images)
        self.thread.start()

        self.load_hdf5_data()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.arrows_visible = not self.arrows_visible
            logging.info(f"Arrows visibility toggled: {self.arrows_visible}")
            self.update_ui_with_images(
                self.current_image,
                self.current_mask,
                self.current_overlay,
                self.current_uncertainty_heatmap,
                self.current_image_with_arrows,
                self.current_mask_with_arrows,
                self.current_overlay_with_arrows,
                self.current_uncertainty_with_arrows
            )
        elif event.key() in [Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D]:
            logging.info(f"Arrow key pressed: {event.key()}")
            if len(self.selected_coords) > 0:
                if event.key() == Qt.Key_D or event.key() == Qt.Key_S:
                    self.selected_arrow_index = (self.selected_arrow_index + 1) % len(self.selected_coords)
                elif event.key() == Qt.Key_A or event.key() == Qt.Key_W:
                    self.selected_arrow_index = (self.selected_arrow_index - 1) % len(self.selected_coords)

                logging.info(f"Selected arrow index: {self.selected_arrow_index}")
                self.highlight_selected_arrow()
        else:
            super(PatchImageViewer, self).keyPressEvent(event)

    def highlight_selected_arrow(self):
        if not self.selected_coords:
            logging.warning("No arrow coordinates found to highlight.")
            return

        selected_coord = self.selected_coords[self.selected_arrow_index]
        logging.info(f"Highlighting arrow at index {self.selected_arrow_index}, coordinate: {selected_coord}")

        self.current_image_with_arrows = self.worker.draw_arrows(self.current_image, self.selected_coords,
                                                                 selected_arrow=selected_coord)
        self.current_mask_with_arrows = self.worker.draw_arrows(self.current_mask, self.selected_coords,
                                                                selected_arrow=selected_coord)
        self.current_overlay_with_arrows = self.worker.draw_arrows(self.current_overlay, self.selected_coords,
                                                                   selected_arrow=selected_coord)
        self.current_uncertainty_with_arrows = self.worker.draw_arrows(self.current_uncertainty_heatmap, self.selected_coords,
                                                                       selected_arrow=selected_coord)

        self.update_ui_with_images(
            self.current_image,
            self.current_mask,
            self.current_overlay,
            self.current_uncertainty_heatmap,
            self.current_image_with_arrows,
            self.current_mask_with_arrows,
            self.current_overlay_with_arrows,
            self.current_uncertainty_with_arrows
        )

    @staticmethod
    def pixmap_to_pil_image(pixmap):
        q_image = pixmap.toImage()
        buffer = q_image.bits().asstring(q_image.byteCount())
        image = Image.frombuffer("RGBA", (q_image.width(), q_image.height()), buffer, "raw", "RGBA", 0, 1)
        return image

    def setup_ui(self):
        self.main_splitter = QSplitter(Qt.Horizontal, self)
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(self.on_file_selected)
        self.main_splitter.addWidget(self.file_list_widget)

        self.setup_graphics_view()

        layout = QVBoxLayout(self)
        layout.addWidget(self.main_splitter)
        self.setLayout(layout)

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

        for filename in filenames_dataset:
            filename = filename.decode('utf-8')
            item_text = f"{filename}"
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

        # Convert to PIL images
        image_pil = Image.fromarray((image * 255).astype('uint8'))
        mask_pil = Image.fromarray(self.worker.process_mask(mask))

        # Process the image, mask, and uncertainty to generate images with arrows
        (
            pixmap_image_without_arrows,
            pixmap_mask_without_arrows,
            pixmap_overlay_without_arrows,
            pixmap_heatmap_without_arrows,
            pixmap_image_with_arrows,
            pixmap_mask_with_arrows,
            pixmap_overlay_with_arrows,
            pixmap_heatmap_with_arrows,
            selected_coords
        ) = self.worker.process_image(
            image_pil,
            mask_pil,
            uncertainty
        )
        self.selected_coords = selected_coords
        logging.info(f"Selected coordinates: {self.selected_coords}")

        # Call the method to update the UI with the processed images
        self.update_ui_with_images(
            pixmap_image_without_arrows,  # Pass the image without arrows
            pixmap_mask_without_arrows,  # Pass the mask without arrows
            pixmap_overlay_without_arrows,  # Pass the overlay without arrows
            pixmap_heatmap_without_arrows,  # Pass the heatmap without arrows
            pixmap_image_with_arrows,  # Pass the image with arrows
            pixmap_mask_with_arrows,  # Pass the mask with arrows
            pixmap_overlay_with_arrows,  # Pass the overlay with arrows
            pixmap_heatmap_with_arrows  # Pass the heatmap with arrows
        )

    def update_ui_with_images(
            self,
            image,
            mask,
            overlay,
            uncertainty_heatmap,
            image_with_arrows,
            mask_with_arrows,
            overlay_with_arrows,
            uncertainty_with_arrows
    ):
        """
        Update the UI with the images provided. The method will set the images for the corresponding
        QGraphicsPixmapItem objects and adjust their positions within the scene.
        """
        self.current_image = image
        self.current_mask = mask
        self.current_overlay = overlay
        self.current_uncertainty_heatmap = uncertainty_heatmap
        self.current_image_with_arrows = image_with_arrows
        self.current_mask_with_arrows = mask_with_arrows
        self.current_overlay_with_arrows = overlay_with_arrows
        self.current_uncertainty_with_arrows = uncertainty_with_arrows

        # Determine if arrows are to be shown
        if self.arrows_visible:
            image_pixmap = image_with_arrows
            mask_pixmap = mask_with_arrows
            overlay_pixmap = overlay_with_arrows
            uncertainty_pixmap = uncertainty_with_arrows
        else:
            image_pixmap = self.convert_pil_to_pixmap(image)
            mask_pixmap = self.convert_pil_to_pixmap(mask)
            overlay_pixmap = self.convert_pil_to_pixmap(overlay)
            uncertainty_pixmap = self.convert_pil_to_pixmap(uncertainty_heatmap)

        # Set pixmaps to the respective QGraphicsPixmapItem objects
        self.image_item.setPixmap(image_pixmap)
        self.mask_item.setPixmap(mask_pixmap)
        self.overlay_item.setPixmap(overlay_pixmap)
        self.uncertainty_item.setPixmap(uncertainty_pixmap)

        # Position the items in the scene
        self.mask_item.setPos(image_pixmap.width(), 0)
        self.overlay_item.setPos(self.mask_item.pos().x() + mask_pixmap.width(), 0)
        self.uncertainty_item.setPos(self.overlay_item.pos().x() + overlay_pixmap.width(), 0)

        # Ensure the image fits within the view
        self.scale_image_to_view()

        # If it's the initial load, update the view
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
        if isinstance(pil_image, QPixmap):
            # If the input is already a QPixmap, return it directly
            return pil_image

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
    hdf5_file_path = r"Z:\PathologyData\attention_unet_fl_f1.h5_COLLECTED_UNCERTAINTIES.h5"
    viewer = PatchImageViewer(hdf5_file_path=hdf5_file_path)
    viewer.show()
    logging.info("Application started")
    sys.exit(app.exec_())
