import sys
import logging
import numpy as np
import h5py
from PIL import Image
from matplotlib import pyplot as plt
from scipy.special import softmax
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN, HDBSCAN

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, Qt, QTimer, QLineF, QRectF, QPoint, QEvent, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QTransform, QPainter, QPainterPath, QBrush
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsPixmapItem, QGraphicsScene,
    QVBoxLayout, QListWidget, QListWidgetItem, QWidget, QSplitter, QGraphicsLineItem, QGraphicsItemGroup, QHBoxLayout,
    QPushButton
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


class ZoomedArrowViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zoomed Arrow Viewer")

        # Set minimum size to avoid being resized too small
        self.setMinimumSize(256, 256)

        # Make it a floating window that stays on top
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        # Setup the graphics view and scene
        self.zoomed_view = QGraphicsView(self)
        self.zoomed_scene = QGraphicsScene(self)
        self.zoomed_view.setScene(self.zoomed_scene)

        # Ensure the image resizes with the window
        self.zoomed_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        layout = QVBoxLayout(self)
        layout.addWidget(self.zoomed_view)
        self.setLayout(layout)

        self.zoomed_pixmap_item = QGraphicsPixmapItem()
        self.zoomed_scene.addItem(self.zoomed_pixmap_item)

        # Store the geometry (size and position) of the window before hiding
        self.saved_geometry = self.geometry()

    def update_image(self, zoomed_pixmap):
        """Update the viewer with the zoomed image and resize to fit."""
        self.zoomed_pixmap_item.setPixmap(zoomed_pixmap)
        self.zoomed_scene.setSceneRect(QRectF(zoomed_pixmap.rect()))
        self.zoomed_view.fitInView(self.zoomed_pixmap_item, Qt.KeepAspectRatio)

    def toggle_visibility(self):
        """Toggle the visibility of the widget without changing its size."""
        if self.isVisible():
            self.saved_geometry = self.geometry()  # Save the current size and position before hiding
            self.hide()
        else:
            self.setGeometry(self.saved_geometry)  # Restore the size and position
            self.show()

    def resizeEvent(self, event):
        """Ensure the image scales when the window is resized."""
        self.zoomed_view.fitInView(self.zoomed_pixmap_item, Qt.KeepAspectRatio)
        super().resizeEvent(event)

    def keyPressEvent(self, event):
        """Handle key press events for toggling the zoom viewer with 'Z' and arrow visibility with 'T'."""
        if event.key() == Qt.Key_Z:
            self.toggle_visibility()
        elif event.key() == Qt.Key_T and self.arrow_manager:
            self.arrow_manager.toggle_arrow_visibility_on_zoom()  # Toggle arrow visibility
        super().keyPressEvent(event)



class UncertaintyRegionSelector:
    def __init__(self, top_n=32, min_distance=24, aggregation_method='mean', clustering_eps=None):
        """
        Initialize the region selector.

        :param top_n: The number of top uncertain regions to return.
        :param min_distance: The minimum distance between uncertain regions.
        :param aggregation_method: The method to reduce the 3D uncertainty map to 2D (mean, max, std).
        :param clustering_eps: Optional custom epsilon for DBSCAN clustering.
        """
        self.top_n = top_n
        self.min_distance = min_distance
        self.aggregation_method = aggregation_method
        self.clustering_eps = clustering_eps if clustering_eps else min_distance  # Default to min_distance

        # Validate parameters
        self._validate_params()

    def _validate_params(self):
        """Validate the input parameters to ensure they are within acceptable ranges."""
        if self.top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        if self.min_distance <= 0:
            raise ValueError("min_distance must be a positive integer")

    def select_regions(self, uncertainty_map):
        """
        Identify the top N uncertain regions in the given uncertainty map.

        :param uncertainty_map: A 3D array representing uncertainty values for the image.
        :return: A list of coordinates for the top uncertain regions.
        """
        # Reduce the 3D uncertainty map to a 2D uncertainty map using the chosen aggregation method
        uncertainty_map_2d = self.aggregate_uncertainty(uncertainty_map)

        # Normalize the uncertainty map
        uncertainty_map_2d = self.normalize_uncertainty(uncertainty_map_2d)

        # Identify the coordinates of the most uncertain regions
        return self.identify_significant_coords(uncertainty_map_2d, self.top_n, self.min_distance)

    def aggregate_uncertainty(self, uncertainty_map):
        """
        Aggregate the 3D uncertainty map into a 2D map using the specified method.

        :param uncertainty_map: A 3D array of uncertainty values.
        :return: A 2D array representing the aggregated uncertainty map.
        """
        if self.aggregation_method == 'mean':
            return np.mean(uncertainty_map, axis=-1)
        elif self.aggregation_method == 'max':
            return np.max(uncertainty_map, axis=-1)
        elif self.aggregation_method == 'std':
            return np.std(uncertainty_map, axis=-1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    @staticmethod
    def normalize_uncertainty(uncertainty):
        """
        Normalize the uncertainty array to the range [0, 255].

        :param uncertainty: A 2D array of uncertainty values.
        :return: A normalized 2D array.
        """
        normalized_uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty))
        return (normalized_uncertainty * 255).astype(np.uint8)

    def identify_significant_coords(self, uncertainty_map, top_n, min_distance):
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

        # If no significant coordinates are found, return an empty list or log a warning
        if len(coords) == 0:
            print("Warning: No significant uncertainty regions found.")
            return []

        # Cluster the coordinates to ensure minimum distance between them
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=1).fit(coords)
        cluster_centers = np.array([coords[clustering.labels_ == cluster_id].mean(axis=0).astype(int)
                                    for cluster_id in np.unique(clustering.labels_)])

        # Sort the regions based on their uncertainty values (from highest to lowest)
        sorted_centers = cluster_centers[np.argsort(-uncertainty_map[cluster_centers[:, 0], cluster_centers[:, 1]])]

        # Return the top N regions
        return sorted_centers[:top_n]


class ClickableArrowGroup(QGraphicsItemGroup):
    def __init__(self, coord, arrow_color=QColor(57, 255, 20), arrow_size=24, parent=None, manager=None):
        super().__init__(parent)
        self.manager = manager  # Reference to the ArrowManager
        x, y = int(coord[0]), int(coord[1])

        # Create the main arrow body and wings
        arrow_tip = QPoint(int(x), int(y - arrow_size))
        wing_left = QPoint(int(x - arrow_size // 2), int(y - arrow_size // 2))
        wing_right = QPoint(int(x + arrow_size // 2), int(y - arrow_size // 2))

        # Create the QGraphicsLineItem for the body and wings
        pen = QPen(arrow_color, 3)
        self.body_item = QGraphicsLineItem(QLineF(QPoint(int(x), int(y)), arrow_tip))
        self.body_item.setPen(pen)
        self.wing_left_item = QGraphicsLineItem(QLineF(QPoint(int(x), int(y)), wing_left))
        self.wing_left_item.setPen(pen)
        self.wing_right_item = QGraphicsLineItem(QLineF(QPoint(int(x), int(y)), wing_right))
        self.wing_right_item.setPen(pen)

        # Add the lines to the group
        self.addToGroup(self.body_item)
        self.addToGroup(self.wing_left_item)
        self.addToGroup(self.wing_right_item)

        # Store default color for resetting later
        self.default_color = arrow_color
        self.setAcceptHoverEvents(True)

        # Track if this arrow is selected and hovered
        self.is_selected = False
        self.hovered = False

        # To sync arrows across scenes
        self.sibling_arrows = []

        # Make the arrow selectable and expand the clickable area
        self.setFlags(QGraphicsItemGroup.ItemIsSelectable)
        self.setAcceptedMouseButtons(Qt.LeftButton)

    def set_sibling_arrows(self, siblings):
        """Set sibling arrows in other scenes."""
        self.sibling_arrows = siblings

    def set_arrow_color(self, color):
        """Set the color of the entire arrow group and sync with sibling arrows."""
        pen = QPen(color, 3)
        self.body_item.setPen(pen)
        self.wing_left_item.setPen(pen)
        self.wing_right_item.setPen(pen)
        self.sync_sibling_colors(color)

    def reset_color(self):
        """Reset the arrow color to its default and sync with sibling arrows."""
        self.set_arrow_color(self.default_color)

    def sync_sibling_colors(self, color):
        """Sync the color of all sibling arrows across scenes."""
        for sibling in self.sibling_arrows:
            if sibling != self:
                sibling._set_color_directly(color)

    def _set_color_directly(self, color):
        """Set color directly without triggering sync to avoid recursion."""
        pen = QPen(color, 3)
        self.body_item.setPen(pen)
        self.wing_left_item.setPen(pen)
        self.wing_right_item.setPen(pen)

    def mousePressEvent(self, event):
        """Handle mouse click on the arrow and sync selection."""
        super().mousePressEvent(event)
        if not self.is_selected:
            # Ask the manager to handle selection, which will deselect others
            self.manager.select_arrow(self)
            # Select the arrow and set it to red
            self.is_selected = True
            self.set_arrow_color(QColor(255, 0, 0))

    def deselect_arrow(self):
        """Deselect this arrow."""
        self.is_selected = False
        self.reset_color()

    def hoverEnterEvent(self, event):
        """Highlight the arrow when hovered and sync hover with siblings."""
        if not self.is_selected and not self.hovered:
            self.set_arrow_color(QColor(255, 165, 0))  # Orange color on hover
            self.hovered = True
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Reset color when hover ends and sync with siblings."""
        if not self.is_selected and self.hovered:
            self.reset_color()
            self.hovered = False
        super().hoverLeaveEvent(event)

    def boundingRect(self):
        """Expand the bounding rectangle slightly to increase the clickable area."""
        return self.childrenBoundingRect().adjusted(-5, -5, 5, 5)

    def shape(self):
        """Return a larger QPainterPath to improve hit detection."""
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path


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

        # Initialize the zoom viewer as a standalone floating window
        self.zoom_viewer = ZoomedArrowViewer(self)
        self.zoom_viewer.setWindowFlags(Qt.Window)  # Make it a floating window
        self.zoom_viewer.hide()  # Start with it hidden

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

        self.region_selector = UncertaintyRegionSelector()

        self.setup_ui()
        self.resize(1200, 800)

        self.thread = QThread()
        self.worker = ImageLoaderWorker()
        self.worker.moveToThread(self.thread)
        self.worker.imageLoaded.connect(self.update_ui_with_images)
        self.thread.start()

        # Pass the zoom viewer and image pixmap item to the ArrowManager
        self.annotation_manager = ArrowManager(self.image_scene, self.overlay_scene, self.uncertainty_scene,
                                               self.image_item, self.zoom_viewer)

        self.zoom_viewer.arrow_manager = self.annotation_manager
        self.load_hdf5_data()

        # Defer scaling until layout is initialized
        QTimer.singleShot(0, self.scale_image_to_view)

    def setup_ui(self):
        """Setup UI layout with a splitter between file selector and image views."""
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(self.on_file_selected)

        # Setup individual views for each image type (image, overlay, uncertainty)
        self.setup_individual_graphics_views()

        # Organize the views in a horizontal layout
        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.image_view)
        self.image_layout.addWidget(self.overlay_view)
        self.image_layout.addWidget(self.uncertainty_view)

        # Create a widget to hold the image views and set the layout for the image views
        image_container = QWidget()
        image_container.setLayout(self.image_layout)

        # Use a splitter between the image views and the file list
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.addWidget(image_container)  # Add the image views to the splitter
        self.splitter.addWidget(self.file_list_widget)  # Add the file selector to the splitter

        # Set up the main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.splitter)
        self.setLayout(self.main_layout)

        # Connect the splitterMoved signal to resize the images when the splitter is moved
        self.splitter.splitterMoved.connect(self.scale_image_to_view)

    def setup_individual_graphics_views(self):
        """Create the scenes and views for the image, overlay, and uncertainty heatmap."""
        self.image_scene = QGraphicsScene(self)
        self.image_view = QGraphicsView(self.image_scene, self)
        self.image_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_item = QGraphicsPixmapItem()
        self.image_scene.addItem(self.image_item)

        self.overlay_scene = QGraphicsScene(self)
        self.overlay_view = QGraphicsView(self.overlay_scene, self)
        self.overlay_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.overlay_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.overlay_item = QGraphicsPixmapItem()
        self.overlay_scene.addItem(self.overlay_item)

        self.uncertainty_scene = QGraphicsScene(self)
        self.uncertainty_view = QGraphicsView(self.uncertainty_scene, self)
        self.uncertainty_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.uncertainty_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.uncertainty_item = QGraphicsPixmapItem()
        self.uncertainty_scene.addItem(self.uncertainty_item)

    def keyPressEvent(self, event):
        """Handle key press events for toggling the zoom viewer and arrow selection."""
        if event.key() == Qt.Key_Z:
            self.toggle_zoom_viewer()
        elif event.key() == Qt.Key_A:
            self.cycle_arrows(-1)  # Move to the previous arrow
        elif event.key() == Qt.Key_D:
            self.cycle_arrows(1)  # Move to the next arrow
        super().keyPressEvent(event)

    def cycle_arrows(self, direction):
        """Cycle through arrows based on the direction (-1 for previous, 1 for next)."""
        num_arrows = len(self.annotation_manager.synced_arrows)
        if num_arrows == 0:
            return  # No arrows to select

        # Deselect the current arrow by resetting its color
        current_arrow_group = self.annotation_manager.synced_arrows[self.selected_arrow_index][0]
        current_arrow_group.deselect_arrow()

        # Update the selected arrow index
        self.selected_arrow_index = (self.selected_arrow_index + direction) % num_arrows
        logging.info(f"Selected arrow index: {self.selected_arrow_index}")

        # Get the new arrow to be selected
        new_arrow_group = self.annotation_manager.synced_arrows[self.selected_arrow_index][0]

        # Ask the annotation manager to select this arrow
        self.annotation_manager.select_arrow(new_arrow_group)

    def toggle_zoom_viewer(self):
        """Toggle the visibility of the zoom viewer."""
        if self.zoom_viewer.isVisible():
            self.zoom_viewer.hide()  # Hide if currently visible
        else:
            self.zoom_viewer.show()  # Show if currently hidden

    def update_ui_with_images(self, image, mask, overlay, uncertainty_heatmap):
        """Update the UI with the images and scale them properly on load."""
        self.current_image = image
        self.current_overlay = overlay
        self.current_uncertainty_heatmap = uncertainty_heatmap

        # Set the processed images to the respective QGraphicsPixmapItem objects
        self.image_item.setPixmap(image)
        self.overlay_item.setPixmap(overlay)
        self.uncertainty_item.setPixmap(uncertainty_heatmap)

        # Call scale_image_to_view to ensure they are scaled appropriately
        self.scale_image_to_view()

        # Draw the arrows after the images have been properly set
        self.draw_arrows()

    def draw_arrows(self):
        """Draw arrows on all scenes and sync them."""
        if self.annotation_manager:
            self.annotation_manager.clear_annotations()  # Clear previous annotations
            # Ensure the arrows are drawn and synced across all scenes
            self.annotation_manager.draw_arrows(self.selected_coords, self.image_item)

    def scale_image_to_view(self):
        """Scale all views to fit the available space, ensuring synchronized scaling across views."""
        # Reset the transformation for all views
        for view in [self.image_view, self.overlay_view, self.uncertainty_view]:
            view.resetTransform()

        # Fit the scenes to the view without exceeding the view's bounds
        for view in [self.image_view, self.overlay_view, self.uncertainty_view]:
            view.fitInView(view.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

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
        # Rescale the images to fit the views
        self.scale_image_to_view()

        # Call the parent class resize event
        super(PatchImageViewer, self).resizeEvent(event)

    def closeEvent(self, event):
        self.hdf5_file.close()
        super(PatchImageViewer, self).closeEvent(event)

    def changeEvent(self, event):
        """Handle window state changes like maximization or minimization."""
        if event.type() == QEvent.WindowStateChange:
            if self.isMaximized() or self.isFullScreen():
                self.scale_image_to_view()
        super().changeEvent(event)


class ArrowManager:
    def __init__(self, image_scene, overlay_scene, uncertainty_scene, image_pixmap_item, zoom_viewer):
        self.scenes = [image_scene, overlay_scene, uncertainty_scene]
        self.image_pixmap_item = image_pixmap_item  # Reference to the image pixmap item
        self.zoom_viewer = zoom_viewer  # Reference to the zoom viewer
        self.synced_arrows = []  # List of synced arrows across scenes
        self.selected_arrow = None  # Track the currently selected arrow
        self.show_arrow_on_zoom = True  # Flag to toggle arrow visibility on the zoomed crop
        self.initial_size_set = False  # Flag to track if the initial size was set

    def draw_arrows(self, arrow_coords, image_item, arrow_color=(57, 255, 20), arrow_size=24):
        """Draw arrows on all scenes at the given coordinates and sync their selection."""
        self.synced_arrows.clear()  # Clear previous arrows
        for coord in arrow_coords:
            y, x = coord
            scene_x = image_item.pos().x() + x
            scene_y = image_item.pos().y() + y

            # Create arrow groups for each scene
            arrow_group_instances = []
            for scene in self.scenes:
                arrow_group = ClickableArrowGroup((scene_x, scene_y), arrow_color=QColor(*arrow_color),
                                                  arrow_size=arrow_size, manager=self)
                scene.addItem(arrow_group)
                arrow_group_instances.append(arrow_group)

            # Sync the color and selection state of all arrows across scenes
            for arrow_group in arrow_group_instances:
                arrow_group.set_sibling_arrows(arrow_group_instances)  # Pass the list of sibling arrows

            self.synced_arrows.append(arrow_group_instances)

    def clear_annotations(self):
        """Clear all arrows (annotations) from all scenes."""
        for scene in self.scenes:
            for arrow_group_instances in self.synced_arrows:
                for arrow_group in arrow_group_instances:
                    if arrow_group.scene():
                        scene.removeItem(arrow_group)
        self.synced_arrows.clear()

    def select_arrow(self, arrow_group):
        """Handle arrow selection, ensuring only one arrow is selected at a time."""
        # Deselect the currently selected arrow if one exists
        if self.selected_arrow and self.selected_arrow != arrow_group:
            self.selected_arrow.deselect_arrow()

        # Set the new arrow as the selected one
        self.selected_arrow = arrow_group

        # Set the selected arrow color to red
        arrow_group.set_arrow_color(QColor(255, 0, 0))  # Red color for selected

        # Show the zoom viewer if hidden and zoom in on the selected arrow
        if not self.zoom_viewer.isVisible():
            self.zoom_viewer.show()

        self.zoom_in_on_arrow(arrow_group)

    def create_zoomed_crop(self, x_center, y_center, crop_size):
        """Create a zoomed crop from the current image using a fixed rectangle of 128x128."""
        current_pixmap = self.image_pixmap_item.pixmap()

        # Get the dimensions of the current image
        img_width = current_pixmap.width()
        img_height = current_pixmap.height()

        # Calculate starting points, ensuring they don't go below 0
        x_start = max(0, x_center - crop_size // 2)
        y_start = max(0, y_center - crop_size // 2)

        # Adjust the starting points if the crop exceeds image boundaries
        if x_start + crop_size > img_width:
            x_start = img_width - crop_size  # Shift left to fit the crop within the image
        if y_start + crop_size > img_height:
            y_start = img_height - crop_size  # Shift up to fit the crop within the image

        # Make sure the width and height are valid and fit within the image
        width = min(crop_size, img_width - x_start)
        height = min(crop_size, img_height - y_start)

        # Crop the specified area from the pixmap
        cropped_pixmap = current_pixmap.copy(x_start, y_start, width, height)

        # Scale the cropped area to fit the zoom window
        zoom_factor = 2  # Adjust zoom factor for clarity
        zoomed_pixmap = cropped_pixmap.scaled(crop_size * zoom_factor, crop_size * zoom_factor, Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)

        return zoomed_pixmap, x_start, y_start  # Return the start points to adjust the arrow position

    def toggle_arrow_visibility_on_zoom(self):
        """Toggle the visibility of the arrow on the zoomed image."""
        self.show_arrow_on_zoom = not self.show_arrow_on_zoom
        if self.selected_arrow:
            self.zoom_in_on_arrow(self.selected_arrow)  # Redraw the zoomed image with or without the arrow

    def zoom_in_on_arrow(self, arrow_group):
        """Zoom in on the area around the selected arrow and adjust zoomed view size."""
        # Get the position of the arrow (scene coordinates)
        arrow_position = arrow_group.body_item.line().p1()

        # Define the initial size of the zoomed region
        crop_size = 256
        x_center = int(arrow_position.x())
        y_center = int(arrow_position.y())

        # Create the zoomed crop and get the adjusted crop boundaries
        zoomed_pixmap, x_start, y_start = self.create_zoomed_crop(x_center, y_center, crop_size)

        # Calculate the arrow's position relative to the zoomed crop
        arrow_rel_x = x_center - x_start  # Relative to the crop's top-left corner
        arrow_rel_y = y_center - y_start  # Relative to the crop's top-left corner

        # Adjust the arrow position based on the zoom factor
        zoom_factor = 2
        arrow_rel_x_scaled = arrow_rel_x * zoom_factor
        arrow_rel_y_scaled = arrow_rel_y * zoom_factor

        # Draw the point on the zoomed pixmap if the flag is True
        if self.show_arrow_on_zoom:
            zoomed_pixmap = self.draw_arrow_on_zoomed_crop(zoomed_pixmap, arrow_rel_x_scaled, arrow_rel_y_scaled)

        # Set the initial size of the zoom view only if it hasn't been set yet
        if not self.initial_size_set:
            self.set_sensible_zoom_view_size()

        # Update the zoom viewer with the zoomed image
        if zoomed_pixmap:
            self.zoom_viewer.update_image(zoomed_pixmap)

    def set_sensible_zoom_view_size(self):
        """Set a sensible initial size for the zoom view window when opened, but allow resizing."""
        # Define a sensible initial size for the zoomed view window (e.g., 400x400 pixels)
        initial_width = 512
        initial_height = 512

        self.zoom_viewer.resize(initial_width, initial_height)
        self.initial_size_set = True  # Mark that the initial size has been set
        logging.info(f"Set zoom view to initial size: {initial_width}x{initial_height}")

    def draw_arrow_on_zoomed_crop(self, zoomed_pixmap, arrow_x, arrow_y):
        """Draw a point at the arrow position on the zoomed pixmap."""
        painter = QPainter(zoomed_pixmap)
        painter.setPen(QPen(Qt.red, 2))  # Set red color for the point
        painter.setBrush(QBrush(Qt.red))  # Fill the circle with red

        # Define point size
        point_radius = 5

        # Draw a filled circle (point) at the given coordinates
        painter.drawEllipse(QPointF(arrow_x, arrow_y), point_radius, point_radius)

        painter.end()  # Finish painting

        return zoomed_pixmap


# Main entry for testing
if __name__ == '__main__':
    app = QApplication(sys.argv)
    hdf5_file_path = r"C:\Users\benja\OneDrive - University of Leeds\DATABACKUP\attention_unet_fl_f1.h5_COLLECTED_UNCERTAINTIES.h5"
    viewer = PatchImageViewer(hdf5_file_path=hdf5_file_path)
    viewer.show()
    logging.info("Application started")
    sys.exit(app.exec_())
