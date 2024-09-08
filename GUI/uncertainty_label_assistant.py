import sys
import logging
import numpy as np
import h5py
from PIL import Image
from matplotlib import pyplot as plt
from scipy.special import softmax
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, Qt, QTimer, QLineF, QRectF, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QTransform
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsPixmapItem, QGraphicsScene,
    QVBoxLayout, QListWidget, QListWidgetItem, QWidget, QSplitter, QGraphicsLineItem, QGraphicsItemGroup
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


class ZoomedArrowViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zoomed Arrow Viewer")
        self.zoomed_view = QGraphicsView(self)
        self.zoomed_scene = QGraphicsScene(self)
        self.zoomed_view.setScene(self.zoomed_scene)

        layout = QVBoxLayout(self)
        layout.addWidget(self.zoomed_view)
        self.setLayout(layout)

        self.zoomed_pixmap_item = QGraphicsPixmapItem()
        self.zoomed_scene.addItem(self.zoomed_pixmap_item)

    def update_image(self, zoomed_pixmap):
        """Update the viewer with the zoomed image."""
        self.zoomed_pixmap_item.setPixmap(zoomed_pixmap)
        self.zoomed_scene.setSceneRect(QRectF(zoomed_pixmap.rect()))
        self.zoomed_view.fitInView(self.zoomed_pixmap_item, Qt.KeepAspectRatio)


class UncertaintyRegionSelector:
    def __init__(self, top_n=16, min_distance=32):
        """
        Initialize the region selector.

        :param top_n: The number of top uncertain regions to return.
        :param min_distance: The minimum distance between uncertain regions.
        """
        self.top_n = top_n
        self.min_distance = min_distance

    def select_regions(self, uncertainty_map):
        """
        Identify the top N uncertain regions in the given uncertainty map.

        :param uncertainty_map: A 3D array representing uncertainty values for the image.
        :return: A list of coordinates for the top uncertain regions.
        """
        # Reduce the 3D uncertainty map to a 2D uncertainty map using the mean across the last dimension
        uncertainty_map_2d = np.mean(uncertainty_map, axis=-1)

        # Normalize the uncertainty map
        uncertainty_map_2d = self.normalize_uncertainty(uncertainty_map_2d)

        # Identify the coordinates of the most uncertain regions
        return self.identify_significant_coords(uncertainty_map_2d, self.top_n, self.min_distance)

    @staticmethod
    def normalize_uncertainty(uncertainty):
        """
        Normalize the uncertainty array to the range [0, 255].

        :param uncertainty: A 2D array of uncertainty values.
        :return: A normalized 2D array.
        """
        normalized_uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty))
        return (normalized_uncertainty * 255).astype(np.uint8)

    @staticmethod
    def identify_significant_coords(uncertainty_map, top_n, min_distance):
        """
        Identify the top uncertain regions based on the uncertainty map.

        :param uncertainty_map: A 2D array representing normalized uncertainty values.
        :param top_n: The number of top regions to return.
        :param min_distance: The minimum distance between selected regions.
        :return: A list of coordinates for the top uncertain regions.
        """
        # Apply a maximum filter to find local maxima
        local_max = maximum_filter(uncertainty_map, size=min_distance) == uncertainty_map
        coords = np.column_stack(np.nonzero(local_max))

        # If no significant coordinates are found, return an empty list
        if len(coords) == 0:
            return []

        # Cluster the coordinates to ensure minimum distance between them
        clustering = DBSCAN(eps=min_distance, min_samples=1).fit(coords)
        cluster_centers = [coords[clustering.labels_ == cluster_id].mean(axis=0).astype(int)
                           for cluster_id in np.unique(clustering.labels_)]

        # Ensure that each cluster center is a tuple of integer coordinates
        cluster_centers = [tuple(map(int, coord)) for coord in cluster_centers]

        # Sort the regions based on their uncertainty values (from highest to lowest)
        cluster_centers = sorted(cluster_centers, key=lambda coord: -float(uncertainty_map[coord[0], coord[1]]))

        return cluster_centers[:top_n]


class ClickableArrowGroup(QGraphicsItemGroup):
    def __init__(self, coord, arrow_color=QColor(57, 255, 20), arrow_size=24, parent=None):
        super().__init__(parent)
        x, y = coord

        # Create the main arrow body and wings
        arrow_tip = QPoint(x, y - arrow_size)
        wing_left = QPoint(x - arrow_size // 2, y - arrow_size // 2)
        wing_right = QPoint(x + arrow_size // 2, y - arrow_size // 2)

        # Create the QGraphicsLineItem for the body and wings
        pen = QPen(arrow_color, 2)
        self.body_item = QGraphicsLineItem(QLineF(QPoint(x, y), arrow_tip))
        self.body_item.setPen(pen)
        self.wing_left_item = QGraphicsLineItem(QLineF(QPoint(x, y), wing_left))
        self.wing_left_item.setPen(pen)
        self.wing_right_item = QGraphicsLineItem(QLineF(QPoint(x, y), wing_right))
        self.wing_right_item.setPen(pen)

        # Add the lines to the group
        self.addToGroup(self.body_item)
        self.addToGroup(self.wing_left_item)
        self.addToGroup(self.wing_right_item)

        # Store default color for resetting later
        self.default_color = arrow_color
        self.setAcceptHoverEvents(True)  # Enable hover events

        # Track if this arrow is selected
        self.is_selected = False

    def set_arrow_color(self, color):
        """Set the color of the entire arrow group."""
        pen = QPen(color, 2)
        self.body_item.setPen(pen)
        self.wing_left_item.setPen(pen)
        self.wing_right_item.setPen(pen)

    def reset_color(self):
        """Reset the arrow color to its default."""
        self.set_arrow_color(self.default_color)

    def mousePressEvent(self, event):
        """Handle mouse click on the arrow."""
        self.set_arrow_color(QColor(255, 0, 0))  # Change color to red when selected
        self.is_selected = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        super().mouseReleaseEvent(event)

    def hoverEnterEvent(self, event):
        """Highlight the arrow when hovered."""
        if not self.is_selected:
            self.set_arrow_color(QColor(255, 165, 0))  # Orange color on hover
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Reset color when hover ends."""
        if not self.is_selected:
            self.reset_color()
        super().hoverLeaveEvent(event)


class ImageLoaderWorker(QObject):
    finished = pyqtSignal()
    imageLoaded = pyqtSignal(object, object, object, object)

    def __init__(self):
        super().__init__()
        cmap = plt.get_cmap('tab10')
        self.color_map = {i: tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(1, 11)}

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def run(self, image, mask, uncertainty):
        pixmap_image, pixmap_mask, pixmap_overlay, pixmap_heatmap = self.process_image(image, mask, uncertainty)
        self.imageLoaded.emit(pixmap_image, pixmap_mask, pixmap_overlay, pixmap_heatmap)
        self.finished.emit()

    @staticmethod
    def process_mask(mask):
        class_probs = softmax(mask, axis=-1)
        class_labels = np.argmax(class_probs, axis=-1)
        height, width = class_labels.shape
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
        cmap = plt.get_cmap('tab10')

        for class_label in np.unique(class_labels):
            color = np.array(cmap(class_label)[:3]) * 255
            rgb_mask[class_labels == class_label] = color.astype(np.uint8)

        return rgb_mask

    @staticmethod
    def create_overlay(image, mask, alpha=0.5):
        image_rgba = image.convert('RGBA')
        mask_rgba = mask.convert('RGBA')

        image_np = np.array(image_rgba)
        mask_np = np.array(mask_rgba)

        blended_np = image_np.copy()
        for i in range(3):
            blended_np[:, :, i] = alpha * mask_np[:, :, i] + (1 - alpha) * image_np[:, :, i]

        blended_np[:, :, 3] = np.maximum(mask_np[:, :, 3] * alpha, image_np[:, :, 3])

        return Image.fromarray(blended_np, 'RGBA')

    def process_image(self, image, mask, uncertainty):
        uncertainty = self.normalize_uncertainty(uncertainty)
        overlay = self.convert_pil_to_pixmap(self.create_overlay(image, mask))
        image = self.convert_pil_to_pixmap(image)
        mask = self.convert_pil_to_pixmap(mask)
        heatmap_rgba = self.convert_pil_to_pixmap(self.create_heatmap(uncertainty))
        return image, mask, overlay, heatmap_rgba

    @staticmethod
    def convert_pil_to_pixmap(pil_image):
        if isinstance(pil_image, QPixmap):
            return pil_image

        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

        image_np = np.array(pil_image)
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)

        height, width, channels = image_np.shape
        q_image = QImage(image_np.data, width, height, width * channels, QImage.Format_RGBA8888)
        return QPixmap.fromImage(q_image)

    @staticmethod
    def normalize_uncertainty(uncertainty):
        uncertainty = np.mean(uncertainty, axis=-1)
        normalized_uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty))
        return (normalized_uncertainty * 255).astype(np.uint8)

    @staticmethod
    def create_heatmap(normalized_uncertainty):
        colormap = plt.get_cmap('Spectral_r')
        heatmap = colormap(normalized_uncertainty / 255.0)
        return Image.fromarray((heatmap[:, :, :3] * 255).astype(np.uint8)).convert('RGBA')


class CustomGraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_arrow = None  # Track the currently selected arrow

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), QTransform())
        if isinstance(item, ClickableArrowGroup):
            logging.info(f"Arrow clicked at position: {item.childItems()[0].line().p1()}")

            # If there's a previously selected arrow, reset its color
            if self.selected_arrow:
                self.selected_arrow.reset_color()

            # Set the clicked arrow as selected and change its color to red
            item.set_arrow_color(QColor(255, 0, 0))  # Set the color to red
            self.selected_arrow = item  # Track the currently selected arrow

        super().mousePressEvent(event)


class PatchImageViewer(QWidget):
    def __init__(self, *args, hdf5_file_path):
        super(PatchImageViewer, self).__init__(*args)
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')

        self.labels = []
        self.current_filename = None
        self.selected_arrow_index = 0
        self.selected_coords = []

        self.label_radius = 8
        self.current_class = 0

        self.class_components = {
            0: 'Non-Informative',
            1: 'Tumour',
            2: 'Stroma',
            3: 'Necrosis',
            4: 'Vessel',
            5: 'Inflammation',
            6: 'Tumour-Lumen',
            7: 'Mucin',
            8: 'Muscle'
        }

        self.region_selector = UncertaintyRegionSelector(top_n=16, min_distance=32)

        self.setup_ui()
        self.resize(1200, 800)

        self.thread = QThread()
        self.worker = ImageLoaderWorker()
        self.worker.moveToThread(self.thread)
        self.worker.imageLoaded.connect(self.update_ui_with_images)
        self.thread.start()

        self.annotation_manager = AnnotationManager(self.image_scene)

        self.load_hdf5_data()

    def setup_ui(self):
        self.main_splitter = QSplitter(Qt.Horizontal, self)
        self.content_splitter = QSplitter(Qt.Vertical, self)
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(self.on_file_selected)
        self.content_splitter.addWidget(self.file_list_widget)

        # Setup individual views for each image type (image, overlay, uncertainty)
        self.setup_individual_graphics_views()

        self.main_splitter.addWidget(self.content_splitter)
        layout = QVBoxLayout(self)
        layout.addWidget(self.main_splitter)
        self.setLayout(layout)

        # Connect splitter's resize signal to scaling
        self.main_splitter.splitterMoved.connect(self.scale_image_to_view)
        self.content_splitter.splitterMoved.connect(self.scale_image_to_view)

    def setup_individual_graphics_views(self):
        # Create separate scenes and views for the image, overlay, and uncertainty heatmap
        self.image_scene = QGraphicsScene(self)
        self.image_view = QGraphicsView(self.image_scene, self)
        self.image_item = QGraphicsPixmapItem()  # Initialize the QGraphicsPixmapItem for the image
        self.image_scene.addItem(self.image_item)

        self.overlay_scene = QGraphicsScene(self)
        self.overlay_view = QGraphicsView(self.overlay_scene, self)
        self.overlay_item = QGraphicsPixmapItem()  # Initialize the QGraphicsPixmapItem for the overlay
        self.overlay_scene.addItem(self.overlay_item)

        self.uncertainty_scene = QGraphicsScene(self)
        self.uncertainty_view = QGraphicsView(self.uncertainty_scene, self)
        self.uncertainty_item = QGraphicsPixmapItem()  # Initialize the QGraphicsPixmapItem for the uncertainty heatmap
        self.uncertainty_scene.addItem(self.uncertainty_item)

        # Set layouts for the views in left-to-right order (image, overlay, uncertainty)
        self.main_splitter.addWidget(self.image_view)
        self.main_splitter.addWidget(self.overlay_view)
        self.main_splitter.addWidget(self.uncertainty_view)

    def update_ui_with_images(self, image, mask, overlay, uncertainty_heatmap):
        self.current_image = image
        self.current_overlay = overlay
        self.current_uncertainty_heatmap = uncertainty_heatmap

        # Set the processed images to the respective QGraphicsPixmapItem objects
        self.image_item.setPixmap(image)
        self.overlay_item.setPixmap(overlay)
        self.uncertainty_item.setPixmap(uncertainty_heatmap)

        self.scale_image_to_view()  # Scale the view after updating the images

        # Draw the arrows after the images have been properly set
        self.draw_arrows()

    def draw_arrows(self):
        """Draw arrows on the first scene over the image."""
        if self.annotation_manager:
            self.annotation_manager.clear_annotations()  # Clear previous annotations
            # Ensure the arrows are drawn on the image scene
            self.annotation_manager.draw_arrows(self.selected_coords, self.image_item)

    def scale_image_to_view(self):
        """Scale all views to fit the available space."""
        for view in [self.image_view, self.overlay_view, self.uncertainty_view]:
            view.resetTransform()
            rect = view.scene().itemsBoundingRect()
            if rect.width() == 0 or rect.height() == 0:
                continue

            scaleFactorX = view.width() / rect.width() if rect.width() > 0 else 1
            scaleFactorY = view.height() / rect.height() if rect.height() > 0 else 1
            scaleFactor = min(scaleFactorX, scaleFactorY)

            view.scale(scaleFactor, scaleFactor)
            view.centerOn(rect.center())

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
        image = self.image_dataset[index]
        mask = self.predicted_segmentation[index]
        uncertainty = self.uncertainty_dataset[index]

        image_pil = Image.fromarray((image * 255).astype('uint8'))
        mask_pil = Image.fromarray(self.worker.process_mask(mask))

        # Select top uncertain regions for labeling
        self.selected_coords = self.region_selector.select_regions(uncertainty)
        logging.info(f"Selected coordinates for labeling: {self.selected_coords}")

        self.worker.run(image_pil, mask_pil, uncertainty)

    def resizeEvent(self, event):
        """Handle window resize event to rescale images."""
        self.scale_image_to_view()
        super(PatchImageViewer, self).resizeEvent(event)

    def closeEvent(self, event):
        self.hdf5_file.close()
        super(PatchImageViewer, self).closeEvent(event)


class AnnotationManager:
    def __init__(self, scene):
        self.scene = scene
        self.annotation_items = []

    def draw_arrows(self, arrow_coords, image_item, arrow_color=(57, 255, 20), arrow_size=24):
        """Draw arrows on the scene at the given coordinates."""
        for coord in arrow_coords:
            y, x = coord  # Switch x and y coordinates here
            scene_x = image_item.pos().x() + x  # x coordinate
            scene_y = image_item.pos().y() + y  # y coordinate
            arrow_group = ClickableArrowGroup((scene_x, scene_y), arrow_color=QColor(*arrow_color), arrow_size=arrow_size)
            self.scene.addItem(arrow_group)
            self.annotation_items.append(arrow_group)

    def clear_annotations(self):
        """Clear all annotations from the scene."""
        # Safely remove all annotation items from the scene and empty the list
        while self.annotation_items:
            item = self.annotation_items.pop()  # Remove from the list first
            if item.scene():  # Only remove the item if it's still in the scene
                self.scene.removeItem(item)  # Then remove from the scene
            item = None  # Explicitly dereference the item to avoid further access





# Main entry for testing
if __name__ == '__main__':
    app = QApplication(sys.argv)
    hdf5_file_path = r"C:\Users\benja\OneDrive - University of Leeds\DATABACKUP\attention_unet_fl_f1.h5_COLLECTED_UNCERTAINTIES.h5"
    viewer = PatchImageViewer(hdf5_file_path=hdf5_file_path)
    viewer.show()
    logging.info("Application started")
    sys.exit(app.exec_())
