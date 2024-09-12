import sys
import logging
import numpy as np
import h5py
from PIL import Image
from matplotlib import pyplot as plt
from scipy.special import softmax
from scipy.ndimage import maximum_filter, gaussian_filter
from skimage.measure import regionprops
from skimage.segmentation import slic
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, Qt, QTimer, QLineF, QRectF, QPoint, QEvent, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QTransform, QPainter, QPainterPath, QBrush, QFont
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsPixmapItem, QGraphicsScene,
    QVBoxLayout, QListWidget, QListWidgetItem, QWidget, QSplitter, QGraphicsLineItem, QGraphicsItemGroup, QHBoxLayout,
    QPushButton, QGraphicsTextItem
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)


class ZoomedArrowViewer(QWidget):
    """
    This class provides a zoomed view for images. The view shows a zoomed-in portion of an image,
    which can be updated and reset as needed. The zoom is managed using QGraphicsView and QGraphicsScene.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zoomed Arrow Viewer")

        # Set minimum size to avoid being resized too small
        self.setMinimumSize(256, 256)

        # Setup the graphics view and scene for displaying the zoomed image
        self.zoomed_view = QGraphicsView(self)
        self.zoomed_scene = QGraphicsScene(self)
        self.zoomed_view.setScene(self.zoomed_scene)

        # Layout to hold the zoomed view
        layout = QVBoxLayout(self)
        layout.addWidget(self.zoomed_view)
        self.setLayout(layout)

        # Item to display the zoomed image
        self.zoomed_pixmap_item = QGraphicsPixmapItem()
        self.zoomed_scene.addItem(self.zoomed_pixmap_item)

    def update_image(self, zoomed_pixmap):
        """
        Update the viewer with the new zoomed image and fit it to the view size.

        :param zoomed_pixmap: QPixmap image to display
        """
        self.zoomed_pixmap_item.setPixmap(zoomed_pixmap)

        # Set the scene size based on the image size
        self.zoomed_scene.setSceneRect(QRectF(zoomed_pixmap.rect()))

        # Fit the image within the view space while maintaining the aspect ratio
        self.zoomed_view.fitInView(self.zoomed_pixmap_item, Qt.KeepAspectRatio)

    def reset_zoom(self):
        """
        Reset the zoom level of the zoomed view without altering the scene size.
        """
        self.zoomed_view.resetTransform()


class UncertaintyRegionSelector:
    """
    This class identifies uncertain regions from an uncertainty map and applies clustering methods
    to ensure that the selected regions are diverse.
    """

    def __init__(self, filter_size=64, aggregation_method='max', clustering_eps=.8, gaussian_sigma=2):
        """
        Initialize the UncertaintyRegionSelector class.

        :param filter_size: Minimum distance between selected uncertain regions.
        :param aggregation_method: How to aggregate the 3D uncertainty map into 2D ('mean', 'max', 'std').
        :param clustering_eps: Epsilon parameter for DBSCAN clustering to cluster regions.
        :param gaussian_sigma: Sigma parameter for the Gaussian filter (controls the amount of smoothing).
        """
        self.filter_size = filter_size
        self.aggregation_method = aggregation_method
        self.clustering_eps = clustering_eps if clustering_eps else filter_size
        self.gaussian_sigma = gaussian_sigma

        # Validate input parameters
        self._validate_params()

    def _validate_params(self):
        """
        Ensure the parameters passed during initialization are within acceptable ranges.
        """
        if self.filter_size <= 0:
            raise ValueError("filter_size must be a positive integer")

    def select_regions(self, uncertainty_map, logits):
        """
        Select the top uncertain regions from the uncertainty map.

        :param uncertainty_map: A 3D array representing uncertainty values.
        :param logits: A 3D array of logit values (per pixel).
        :return: A list of coordinates representing the top uncertain regions.
        """
        # Reduce the 3D uncertainty map to a 2D uncertainty map
        uncertainty_map_2d = self.aggregate_uncertainty(uncertainty_map)

        # Apply Gaussian smoothing
        uncertainty_map_2d = self.apply_gaussian_filter(uncertainty_map_2d)

        # Normalize the uncertainty map to [0, 255]
        uncertainty_map_2d = self.normalize_uncertainty(uncertainty_map_2d)

        # Identify the coordinates of the most uncertain regions
        initial_coords = self.identify_significant_coords(uncertainty_map_2d, logits, self.filter_size)

        # Apply post-processing to remove redundant centers
        final_coords = self.filter_close_centers(initial_coords, 32)

        return final_coords

    def aggregate_uncertainty(self, uncertainty_map):
        """
        Aggregate the 3D uncertainty map into a 2D map using the selected method.
        """
        if self.aggregation_method == 'mean':
            return np.mean(uncertainty_map, axis=-1)
        elif self.aggregation_method == 'max':
            return np.max(uncertainty_map, axis=-1)
        elif self.aggregation_method == 'std':
            return np.std(uncertainty_map, axis=-1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def apply_gaussian_filter(self, uncertainty_map):
        """
        Apply Gaussian smoothing to the 2D uncertainty map.

        :param uncertainty_map: 2D uncertainty map.
        :return: Smoothed uncertainty map.
        """
        return gaussian_filter(uncertainty_map, sigma=self.gaussian_sigma)

    @staticmethod
    def normalize_uncertainty(uncertainty):
        """
        Normalize the uncertainty array to the range [0, 255].
        """
        normalized_uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty))
        return (normalized_uncertainty * 255).astype(np.uint8)

    def identify_significant_coords(self, uncertainty_map, logits, min_distance, edge_buffer=8):
        """
        Identify significant coordinates based on uncertainty and logits and apply clustering.

        :param uncertainty_map: 2D uncertainty values.
        :param logits: Logits per pixel.
        :param min_distance: Minimum distance between selected regions.
        :param edge_buffer: Buffer from image edges to exclude certain coordinates.
        :return: Top uncertain regions (coordinates).
        """
        # Find local maxima in the smoothed uncertainty map
        local_max = maximum_filter(uncertainty_map, size=min_distance) == uncertainty_map
        coords = np.column_stack(np.nonzero(local_max))

        # Filter out points near the edges
        valid_coords = [coord for coord in coords
                        if edge_buffer <= coord[0] <= uncertainty_map.shape[0] - edge_buffer and
                        edge_buffer <= coord[1] <= uncertainty_map.shape[1] - edge_buffer]

        if len(valid_coords) == 0:
            print("Warning: No significant uncertainty regions found after edge filtering.")
            return []

        valid_coords = np.array(valid_coords)

        # Extract corresponding logit features for each valid coordinate
        logit_features = np.array([logits[coord[0], coord[1], :] for coord in valid_coords])

        # Combine coordinates and logit features for clustering
        features = np.hstack((valid_coords, logit_features))

        # Scale features for clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=1, n_jobs=-1).fit(features_scaled)

        # Get cluster centers
        cluster_centers = np.array([valid_coords[clustering.labels_ == cluster_id].mean(axis=0).astype(int)
                                    for cluster_id in np.unique(clustering.labels_)])

        return cluster_centers

    def filter_close_centers(self, centers, min_distance):
        """
        Filter out cluster centers that are too close to each other.
        """
        filtered_centers = []
        for center in centers:
            if all(np.linalg.norm(center - c) >= min_distance for c in filtered_centers):
                filtered_centers.append(center)
        return np.array(filtered_centers)


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
            # Get the current filename from the parent manager
            current_filename = self.manager.current_filename
            self.manager.select_arrow(current_filename, self)

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
    """
    A worker class responsible for processing images, masks, and uncertainty maps in a separate thread.
    This class converts the processed data into QPixmap objects and emits signals when processing is complete.

    Signals:
        finished: Emitted when the image processing is complete.
        imageLoaded: Emitted with the processed image, mask, overlay, and heatmap.

    Attributes:
        color_map (dict): A color mapping for class labels (used for mask processing).
    """

    finished = pyqtSignal()
    imageLoaded = pyqtSignal(object, object, object, object)  # Emitted with the processed image, mask, overlay, heatmap

    def __init__(self):
        """
        Initializes the ImageLoaderWorker class, setting up the color map for mask processing.
        The color map is based on the 'tab10' colormap from Matplotlib, and each class is assigned an RGB color.
        """
        super().__init__()
        cmap = plt.get_cmap('tab10')
        self.color_map = {i: tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(1, 11)}  # Class colors

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def run(self, image, mask, uncertainty):
        """
        Slot that runs the image processing in the background. It converts the image, mask, and uncertainty data into
        QPixmap objects and emits the imageLoaded signal with the processed results.

        :param image: A numpy array representing the image data.
        :param mask: A numpy array representing the class probabilities for each pixel.
        :param uncertainty: A numpy array representing the uncertainty values for the image.
        """
        pixmap_image, pixmap_mask, pixmap_overlay, pixmap_heatmap = self.process_image(image, mask, uncertainty)
        self.imageLoaded.emit(pixmap_image, pixmap_mask, pixmap_overlay, pixmap_heatmap)  # Emit processed results
        self.finished.emit()  # Signal that processing is complete

    @staticmethod
    def process_mask(mask):
        """
        Converts a mask (containing class probabilities) into an RGB mask image, where each class is represented by a
        different color.

        :param mask: A 3D numpy array of class probabilities.
        :return: An RGB mask image (numpy array) with unique colors for each class.
        """
        class_probs = softmax(mask, axis=-1)  # Apply softmax to get class probabilities
        class_labels = np.argmax(class_probs, axis=-1)  # Get the class with the highest probability per pixel
        height, width = class_labels.shape
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)  # Initialize an RGB mask

        cmap = plt.get_cmap('tab10')  # Use the 'tab10' colormap for consistent class color mapping
        for class_label in np.unique(class_labels):
            color = np.array(cmap(class_label)[:3]) * 255  # Get the color for the class
            rgb_mask[class_labels == class_label] = color.astype(np.uint8)  # Assign color to corresponding class pixels

        return rgb_mask  # Return the RGB mask

    @staticmethod
    def create_overlay(image, mask, alpha=0.5):
        """
        Creates a blended overlay image by combining the input image and the mask with a specified transparency (alpha).
        The blending is done on the RGBA channels.

        :param image: A PIL Image object representing the input image.
        :param mask: A PIL Image object representing the mask.
        :param alpha: The transparency factor for the mask overlay (default is 0.5).
        :return: A PIL Image with the overlay applied.
        """
        image_rgba = image.convert('RGBA')  # Convert image to RGBA format
        mask_rgba = mask.convert('RGBA')  # Convert mask to RGBA format

        image_np = np.array(image_rgba)  # Convert the image to a numpy array
        mask_np = np.array(mask_rgba)  # Convert the mask to a numpy array

        blended_np = image_np.copy()  # Make a copy of the image for blending
        for i in range(3):  # Blend the RGB channels of the mask and the image
            blended_np[:, :, i] = alpha * mask_np[:, :, i] + (1 - alpha) * image_np[:, :, i]

        blended_np[:, :, 3] = np.maximum(mask_np[:, :, 3] * alpha, image_np[:, :, 3])  # Blend the alpha channel

        return Image.fromarray(blended_np, 'RGBA')  # Return the blended image as a PIL Image

    def process_image(self, image, mask, uncertainty):
        """
        Processes the input image, mask, and uncertainty data. It creates an overlay image and converts all
        images (including a heatmap from the uncertainty) to QPixmap format.

        :param image: The input image in numpy array format.
        :param mask: The input mask containing class information.
        :param uncertainty: The uncertainty values for the image.
        :return: A tuple of QPixmap objects (image, mask, overlay, heatmap).
        """
        uncertainty = self.normalize_uncertainty(uncertainty)  # Normalize the uncertainty data
        overlay = self.convert_pil_to_pixmap(self.create_overlay(image, mask))  # Create an overlay image
        image = self.convert_pil_to_pixmap(image)  # Convert image to QPixmap
        mask = self.convert_pil_to_pixmap(mask)  # Convert mask to QPixmap
        heatmap_rgba = self.convert_pil_to_pixmap(self.create_heatmap(uncertainty))  # Generate a heatmap

        return image, mask, overlay, heatmap_rgba  # Return the processed QPixmap objects

    @staticmethod
    def convert_pil_to_pixmap(pil_image):
        """
        Converts a PIL image into a QPixmap object. If the image is already a QPixmap, it is returned as-is.
        If the PIL image is not in RGBA format, it is converted to RGBA first.

        :param pil_image: A PIL Image object to be converted to QPixmap.
        :return: A QPixmap representation of the input PIL image.
        """
        if isinstance(pil_image, QPixmap):  # If it's already a QPixmap, return it
            return pil_image

        if pil_image.mode != 'RGBA':  # Convert to RGBA if necessary
            pil_image = pil_image.convert('RGBA')

        image_np = np.array(pil_image)  # Convert the PIL image to a numpy array
        if image_np.dtype != np.uint8:  # Ensure the data type is uint8
            image_np = (image_np * 255).astype(np.uint8)

        # Convert the numpy array to QImage and then to QPixmap
        height, width, channels = image_np.shape
        q_image = QImage(image_np.data, width, height, width * channels, QImage.Format_RGBA8888)
        return QPixmap.fromImage(q_image)

    @staticmethod
    def normalize_uncertainty(uncertainty):
        """
        Normalizes the uncertainty map to the range [0, 255].

        :param uncertainty: A 3D uncertainty array (from multiple channels).
        :return: A normalized 2D array of uncertainty values.
        """
        uncertainty = np.mean(uncertainty, axis=-1)  # Take the mean of the uncertainty across the last axis
        normalized_uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty))
        return (normalized_uncertainty * 255).astype(np.uint8)  # Normalize to [0, 255] and return

    @staticmethod
    def create_heatmap(normalized_uncertainty):
        """
        Creates a heatmap from the normalized uncertainty map using a colormap.

        :param normalized_uncertainty: A normalized 2D uncertainty array.
        :return: A PIL Image representing the heatmap.
        """
        colormap = plt.get_cmap('Spectral_r')  # Use the 'Spectral_r' colormap for visualization
        heatmap = colormap(normalized_uncertainty / 255.0)  # Normalize the uncertainty to [0, 1] for colormap
        return Image.fromarray((heatmap[:, :, :3] * 255).astype(np.uint8)).convert('RGBA')  # Return the heatmap


class CustomGraphicsScene(QGraphicsScene):
    """
    A custom QGraphicsScene subclass to handle mouse events, specifically for selecting arrows in the scene.
    It tracks the currently selected arrow and ensures that only one arrow is selected at a time.
    When an arrow is clicked, its color is changed to red, and any previously selected arrow is reset to its default color.
    """

    def __init__(self, parent=None):
        """
        Initialize the custom graphics scene.

        :param parent: Optional parent widget.
        """
        super().__init__(parent)
        self.selected_arrow = None  # Stores the currently selected arrow

    def mousePressEvent(self, event):
        """
        Handles mouse press events to detect when an arrow (ClickableArrowGroup) is clicked.
        If an arrow is clicked, it logs the position, deselects the previous arrow, and selects the new arrow.

        :param event: QGraphicsSceneMouseEvent that contains information about the mouse press.
        """
        # Get the item at the position of the mouse press
        item = self.itemAt(event.scenePos(), QTransform())

        # Check if the item is an instance of ClickableArrowGroup
        if isinstance(item, ClickableArrowGroup):
            # Log the position of the arrow clicked
            logging.info(f"Arrow clicked at position: {item.childItems()[0].line().p1()}")

            # If an arrow is already selected, reset its color to the default
            if self.selected_arrow:
                self.selected_arrow.reset_color()

            # Set the clicked arrow as the selected arrow and change its color to red
            item.set_arrow_color(QColor(255, 0, 0))  # Change color to red
            self.selected_arrow = item  # Track the newly selected arrow

        # Call the base class's mousePressEvent to ensure normal behavior continues
        super().mousePressEvent(event)


class PatchImageViewer(QWidget):
    """
    A QWidget class responsible for displaying and interacting with patch images, masks, uncertainty maps, and arrows.
    The images are loaded from an HDF5 file, and users can interact with the images and annotate uncertain regions.
    The viewer provides a zoomed-in view of selected areas, allowing for better inspection and annotation.
    """

    def __init__(self, *args, hdf5_file_path):
        """
        Initializes the PatchImageViewer with a given HDF5 file path. The class manages the UI layout,
        image loading, and interaction with annotation features.

        :param args: Additional arguments passed to the QWidget.
        :param hdf5_file_path: Path to the HDF5 file containing image, mask, and uncertainty data.
        """
        super(PatchImageViewer, self).__init__(*args)
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')

        self.labels = []  # List to store labels for annotation
        self.current_filename = None  # Currently selected filename from the file list
        self.selected_arrow_index = 0  # Track the index of the selected arrow
        self.selected_coords = []  # Coordinates of uncertain regions selected for labeling

        self.label_radius = 8  # Radius of the annotation region
        self.current_class = 0  # Currently selected class for annotation

        # Mapping of class components to their labels
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

        self.region_selector = UncertaintyRegionSelector()  # To select uncertain regions

        # Initialize the user interface
        self.setup_ui()

        # Initialize threading and the image loading worker
        self.thread = QThread()
        self.worker = ImageLoaderWorker()
        self.worker.moveToThread(self.thread)
        self.worker.imageLoaded.connect(self.update_ui_with_images)
        self.thread.start()

        # Initialize the arrow annotation manager
        self.annotation_manager = ArrowManager(self.image_scene, self.overlay_scene, self.uncertainty_scene,
                                               self.image_item, self.zoom_viewer)
        self.zoom_viewer.arrow_manager = self.annotation_manager  # Associate the zoom viewer with the arrow manager

        # Load the HDF5 data for display
        self.load_hdf5_data()

        # Resize the viewer to the appropriate size and scale the images on load
        self.resize(1200, 800)
        QTimer.singleShot(0, self.scale_image_to_view)  # Ensure proper scaling after layout initialization

    def setup_ui(self):
        """
        Sets up the user interface layout, including a file selector, image views, and a zoomed-in view.
        A QSplitter is used to organize the image views and file selector, ensuring a responsive layout.
        """
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(self.on_file_selected)  # Connect the file selection event

        # Set up individual graphics views for image, overlay, and uncertainty heatmap
        self.setup_individual_graphics_views()

        # Create the zoom viewer and add it to the main layout
        self.zoom_viewer = ZoomedArrowViewer(self)
        self.zoom_viewer.setMinimumSize(256, 256)  # Set a minimum size for the zoom view

        # Organize the image views horizontally
        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.image_view)
        self.image_layout.addWidget(self.overlay_view)
        self.image_layout.addWidget(self.uncertainty_view)

        # Create a widget to hold the image views and set the layout
        image_container = QWidget()
        image_container.setLayout(self.image_layout)

        # Create a vertical layout for the zoomed view and the file selector
        right_side_layout = QVBoxLayout()
        right_side_layout.addWidget(self.zoom_viewer)  # Add the zoom view on top
        right_side_layout.addWidget(self.file_list_widget)  # Add the file selector below the zoom view

        # Create a widget for the right-side layout (zoomed view + file selector)
        right_side_widget = QWidget()
        right_side_widget.setLayout(right_side_layout)

        # Set up the splitter between the image views and the right side (zoomed view + file selector)
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.addWidget(image_container)  # Add the image views to the splitter
        self.splitter.addWidget(right_side_widget)  # Add the right-side widget (zoomed view + file selector)

        # Set up the main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.splitter)
        self.setLayout(self.main_layout)

        # Ensure proper scaling of images on load
        QTimer.singleShot(0, self.scale_image_to_view)

        # Connect the splitterMoved signal to resize the images when the splitter is moved
        self.splitter.splitterMoved.connect(self.scale_image_to_view)

    def setup_individual_graphics_views(self):
        """
        Initializes individual graphics views for displaying the main image, overlay, and uncertainty heatmap.
        Each view is placed in its own QGraphicsView with horizontal and vertical scrollbars disabled.
        """
        # Image view setup
        self.image_scene = QGraphicsScene(self)
        self.image_view = QGraphicsView(self.image_scene, self)
        self.image_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.image_item = QGraphicsPixmapItem()
        self.image_scene.addItem(self.image_item)

        # Overlay view setup
        self.overlay_scene = QGraphicsScene(self)
        self.overlay_view = QGraphicsView(self.overlay_scene, self)
        self.overlay_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.overlay_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.overlay_item = QGraphicsPixmapItem()
        self.overlay_scene.addItem(self.overlay_item)

        # Uncertainty heatmap view setup
        self.uncertainty_scene = QGraphicsScene(self)
        self.uncertainty_view = QGraphicsView(self.uncertainty_scene, self)
        self.uncertainty_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.uncertainty_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.uncertainty_item = QGraphicsPixmapItem()
        self.uncertainty_scene.addItem(self.uncertainty_item)

    def keyPressEvent(self, event):
        """
        Handle key press events for interacting with the viewer. Supports toggling the zoom view (Z key),
        cycling through arrows (A and D keys), assigning labels to arrows using number keys (0-8),
        and removing labels using the '-' key.
        """
        if event.key() == Qt.Key_Z:
            self.toggle_zoom_viewer()  # Toggle the visibility of the zoom viewer
        elif event.key() == Qt.Key_A:
            self.cycle_arrows(-1)  # Move to the previous arrow
        elif event.key() == Qt.Key_D:
            self.cycle_arrows(1)  # Move to the next arrow
        elif event.key() in [Qt.Key_0, Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4,
                             Qt.Key_5, Qt.Key_6, Qt.Key_7, Qt.Key_8]:
            label = int(event.text())  # Get the label from the pressed number key
            if self.annotation_manager.selected_arrow is not None:
                # Assign the label to the selected arrow
                self.annotation_manager.label_arrow(self.annotation_manager.selected_arrow, label)
        elif event.key() == Qt.Key_Minus:  # Handle the '-' key to remove the label
            if self.annotation_manager.selected_arrow is not None:
                self.annotation_manager.remove_label_from_arrow(self.annotation_manager.selected_arrow)
        super().keyPressEvent(event)  # Call the parent class's keyPressEvent

    def toggle_zoom_viewer(self):
        """
        Toggles the visibility of the zoom viewer. If visible, hides the zoom viewer, otherwise shows it.
        """
        if self.zoom_viewer.isVisible():
            self.zoom_viewer.hide()
        else:
            self.zoom_viewer.show()

    def cycle_arrows(self, direction):
        """
        Cycles through the arrows on the images based on the given direction (-1 for previous, 1 for next).
        When an arrow is selected, the current one is deselected and the next one is highlighted.
        """
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

        # Select the new arrow using the annotation manager
        self.annotation_manager.select_arrow(new_arrow_group)

    def update_ui_with_images(self, image, mask, overlay, uncertainty_heatmap):
        """
        Updates the UI with the loaded image, mask, overlay, and uncertainty heatmap. Ensures that
        the images are properly scaled and arrows are drawn on the views.
        """
        self.current_image = image
        self.current_overlay = overlay
        self.current_uncertainty_heatmap = uncertainty_heatmap

        # Update the respective QGraphicsPixmapItem objects with the new images
        self.image_item.setPixmap(image)
        self.overlay_item.setPixmap(overlay)
        self.uncertainty_item.setPixmap(uncertainty_heatmap)

        # Scale the images to fit the view
        self.scale_image_to_view()

        # Draw arrows on the updated images
        self.draw_arrows()

    def draw_arrows(self):
        """
        Draws arrows on all image scenes (image, overlay, uncertainty). Ensures that arrows are cleared before
        drawing new ones and that they are synchronized across scenes.
        """
        if self.annotation_manager:
            self.annotation_manager.clear_annotations()  # Clear existing arrows
            # Draw and sync arrows across all scenes
            self.annotation_manager.draw_arrows(self.selected_coords, self.image_item)

    def scale_image_to_view(self):
        """
        Scales all image views (image, overlay, uncertainty, and zoom) to fit the available space,
        maintaining the aspect ratio.
        """
        # Reset the transformations for all views
        for view in [self.image_view, self.overlay_view, self.uncertainty_view, self.zoom_viewer.zoomed_view]:
            view.resetTransform()

        # Fit each view's scene to the view size, maintaining aspect ratio
        for view in [self.image_view, self.overlay_view, self.uncertainty_view]:
            view.fitInView(view.scene().itemsBoundingRect(), Qt.KeepAspectRatio)

        # Ensure the zoomed image fits within the zoom viewer
        self.zoom_viewer.zoomed_view.fitInView(self.zoom_viewer.zoomed_pixmap_item, Qt.KeepAspectRatio)

    def load_hdf5_data(self):
        """
        Loads image, logits, uncertainty, and filename data from the HDF5 file. Populates the file list widget
        with the filenames for user selection.
        """
        self.image_dataset = self.hdf5_file['rgb_images']
        self.logits = self.hdf5_file['logits']
        self.uncertainty_dataset = self.hdf5_file['epistemic_uncertainty']
        filenames_dataset = self.hdf5_file['filenames']

        # Populate the file list widget with filenames from the HDF5 dataset
        for filename in filenames_dataset:
            filename = filename.decode('utf-8')
            item_text = f"{filename}"
            item = QListWidgetItem(item_text)
            self.file_list_widget.addItem(item)

        # Automatically select and load the first file if available
        if len(filenames_dataset) > 0:
            self.file_list_widget.setCurrentRow(0)
            self.on_file_selected(self.file_list_widget.item(0))

    def on_file_selected(self, item):
        """
        Handles the event when a file is selected from the file list. Loads the corresponding image, logits,
        and uncertainty data, processes the mask, and identifies uncertain regions for labeling.

        :param item: The QListWidgetItem representing the selected file.
        """
        index = self.file_list_widget.row(item)  # Get the index of the selected file
        image = self.image_dataset[index]
        logits = self.logits[index]
        uncertainty = self.uncertainty_dataset[index]

        # Get the filename from the selected item
        filename = item.text()

        # Convert the image and mask to PIL format
        image_pil = Image.fromarray((image * 255).astype('uint8'))
        mask_pil = Image.fromarray(self.worker.process_mask(logits))

        # Select the top uncertain regions for annotation
        self.selected_coords = self.region_selector.select_regions(uncertainty_map=uncertainty, logits=logits)
        logging.info(f"Selected coordinates for labeling: {self.selected_coords}")

        # Store the selected coordinates and initialize labels and clusters in arrow_data
        self.annotation_manager.store_arrow_data(filename, self.selected_coords)

        # Retrieve the feature matrix (logits) and uncertainty values for the selected coordinates
        feature_matrix = np.array([logits[coord[0], coord[1], :] for coord in self.selected_coords])
        uncertainty_values = np.array([uncertainty[coord[0], coord[1]] for coord in self.selected_coords])

        # Multiply the logits by the uncertainty values to weight the logits by uncertainty
        weighted_feature_matrix = feature_matrix * uncertainty_values

        # Cluster the weighted logits
        self.annotation_manager.cluster_arrows(filename, weighted_feature_matrix)

        # Render the arrows for the selected image
        self.annotation_manager.render_arrows_for_image(filename)

        # Run the worker to process and load the image, mask, and uncertainty
        self.worker.run(image_pil, mask_pil, uncertainty)

        # Set the filename for the current session to track
        self.annotation_manager.current_filename = filename

    def resizeEvent(self, event):
        """
        Handles the resize event for the viewer, ensuring that the images are rescaled to fit the new window size.

        :param event: QResizeEvent triggered by window resizing.
        """
        # Rescale the images to fit the views
        self.scale_image_to_view()

        # Call the parent class's resize event
        super(PatchImageViewer, self).resizeEvent(event)

    def closeEvent(self, event):
        """
        Handles the window close event, ensuring that the HDF5 file is properly closed before exiting.

        :param event: QCloseEvent triggered when the window is closed.
        """
        self.hdf5_file.close()  # Close the HDF5 file
        super(PatchImageViewer, self).closeEvent(event)

    def changeEvent(self, event):
        """
        Handles window state changes (such as maximizing or minimizing), ensuring that images are rescaled when
        the window size changes.

        :param event: QEvent triggered by window state changes.
        """
        if event.type() == QEvent.WindowStateChange:
            if self.isMaximized() or self.isFullScreen():
                self.scale_image_to_view()
        super().changeEvent(event)


class ArrowManager:
    def __init__(self, image_scene, overlay_scene, uncertainty_scene, image_pixmap_item, zoom_viewer):
        """
        Initializes the ArrowManager with the given scenes and viewer, and sets up the colormap and class labels.
        """
        self.scenes = [image_scene, overlay_scene, uncertainty_scene]  # Store the scenes for arrow management
        self.image_pixmap_item = image_pixmap_item  # Reference to the image pixmap item for positioning
        self.zoom_viewer = zoom_viewer  # Reference to the zoomed image viewer
        self.synced_arrows = []  # List of arrows synced across all scenes
        self.selected_arrow = None  # Currently selected arrow
        self.arrow_data = {}
        # Default color for arrows before they are labeled
        self.default_arrow_color = QColor(0, 0, 0)  # Black

        # Use 'tab10' colormap for up to 10 classes
        self.colormap = plt.get_cmap('tab10')
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
        self.num_classes = len(self.class_components)

    def set_arrow_data(self, filename, coords):
        """
        Initialize arrow data with coordinates and prepare empty labels and clusters for the specified filename.
        """
        self.arrow_data[filename] = {
            'coords': coords,
            'labels': [None] * len(coords),
            'clusters': [None] * len(coords)
        }

    def cluster_arrows(self, filename, feature_matrix, distance_threshold=2):
        """
        Clusters arrows based on the provided feature matrix (e.g., logits) and stores the cluster information in arrow_data.

        :param filename: The current file being processed.
        :param feature_matrix: Array of features for each arrow (e.g., logit values or coordinates).
        :param distance_threshold: The linkage distance threshold above which clusters will not be merged.
        """
        # Standardize the feature matrix
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        # Perform Agglomerative Clustering with dynamic number of clusters based on distance threshold
        clustering = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=None).fit(
            feature_matrix_scaled)

        # Get the cluster labels from the clustering result
        cluster_labels = clustering.labels_

        # Store clusters in arrow_data
        for i, coords in enumerate(self.arrow_data[filename]['coords']):
            self.arrow_data[filename]['clusters'][i] = cluster_labels[i]  # Save cluster for each arrow

        logging.info(f"Clustered arrows for {filename} into {len(np.unique(cluster_labels))} clusters.")

        # Optional: plot the clustering result in feature space
        self.plot_clustering_result(feature_matrix_scaled, cluster_labels)

    def plot_clustering_result(self, feature_matrix_scaled, cluster_labels):
        """
        Plots the clustering results in a 2D space for visualization.

        :param feature_matrix_scaled: The standardized feature matrix.
        :param cluster_labels: The cluster labels for each point.
        """
        # Reduce feature space to 2D for visualization using PCA
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        feature_matrix_2d = pca.fit_transform(feature_matrix_scaled)

        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.scatter(feature_matrix_2d[:, 0], feature_matrix_2d[:, 1], c=cluster_labels, cmap='Spectral', s=50)
        plt.title("Agglomerative Clustering Result")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(label="Cluster")
        plt.show()

    def select_arrow(self, filename, arrow_group):
        """
        Handles arrow selection, ensuring only one arrow is selected at a time. If the selected arrow is part of a
        cluster, select all arrows in the same cluster.
        """
        logging.info(f"Attempting to select arrow: {arrow_group}")

        # Find the index of the selected arrow using coordinates comparison
        selected_index = None
        for i, group in enumerate(self.synced_arrows):
            if group[0] == arrow_group or any(arrow == arrow_group for arrow in group):
                selected_index = i
                break

        if selected_index is None:
            logging.error(f"Arrow group not found in synced_arrows: {arrow_group}")
            return

        # Get the cluster ID of the selected arrow
        cluster_id = self.arrow_data[filename]['clusters'][selected_index]

        # Deselect the currently selected arrow, if any
        if self.selected_arrow and self.selected_arrow != arrow_group:
            self.selected_arrow.deselect_arrow()

        # Select all arrows in the same cluster
        for i, cluster in enumerate(self.arrow_data[filename]['clusters']):
            if cluster == cluster_id:
                self.synced_arrows[i][0].set_arrow_color(QColor(255, 0, 0))  # Select by changing color
                self.selected_arrow = self.synced_arrows[i][0]  # Update selected arrow

        # Update zoom view
        self.zoom_in_on_arrow(self.selected_arrow)

    def select_cluster(self, filename, cluster_id):
        """
        Selects all arrows that belong to the same cluster as the currently selected arrow.
        """
        for i, arrow_group_list in enumerate(self.synced_arrows):
            if self.arrow_data[filename]['clusters'][i] == cluster_id:
                arrow_group_list[0].set_arrow_color(QColor(255, 0, 0))  # Highlight arrows in the same cluster

    def render_arrows_for_image(self, filename):
        """
        Render the arrows for the given filename based on the stored arrow data.
        """
        # Clear existing arrows
        self.clear_annotations()

        # Retrieve the arrow data for the current image
        if filename not in self.arrow_data:
            logging.warning(f"No arrow data found for {filename}.")
            return

        arrows_data = self.arrow_data[filename]

        for i, coords in enumerate(arrows_data['coords']):
            arrow_group = self.create_arrow_group_at_coords(coords)
            label = arrows_data['labels'][i]
            if label is not None:
                self.assign_label_to_arrow(arrow_group, label)

    def create_arrow_group_at_coords(self, coords):
        """
        Helper function to create and sync an arrow group at given coordinates across all scenes.
        """
        arrow_group_instances = []
        for scene in self.scenes:
            arrow_group = ClickableArrowGroup(coords, arrow_color=self.default_arrow_color, manager=self)
            scene.addItem(arrow_group)
            arrow_group_instances.append(arrow_group)

        # Set sibling arrows across scenes for syncing behavior
        for arrow_group in arrow_group_instances:
            arrow_group.set_sibling_arrows(arrow_group_instances)

        # Add the synced arrows to the list for reference
        self.synced_arrows.append(arrow_group_instances)
        logging.info(f"Added new arrow group at coords: {coords} to synced_arrows")

        return arrow_group_instances[0]

    def get_arrow_data(self, filename):
        """Retrieve the arrow data for the given filename."""
        return self.arrow_data.get(filename, {})

    def remove_label_from_arrow(self, arrow_group):
        """Remove the label text from the selected arrow and reset its color."""
        if hasattr(arrow_group, 'label_text_items') and arrow_group.label_text_items:
            for text_item in arrow_group.label_text_items:
                for scene in self.scenes:
                    scene.removeItem(text_item)

        arrow_group.label_text_items.clear()

        for arrow in arrow_group.sibling_arrows:
            arrow.set_arrow_color(self.default_arrow_color)

        arrow_group.label = None

    def update_label_text_for_arrow(self, arrow_group, class_text):
        """Adds or updates the QGraphicsTextItem next to the arrow."""
        if hasattr(arrow_group, 'label_text_items') and arrow_group.label_text_items:
            for text_item in arrow_group.label_text_items:
                for scene in self.scenes:
                    scene.removeItem(text_item)

        arrow_position = arrow_group.body_item.line().p1()
        arrow_group.label_text_items = []

        for scene in self.scenes:
            text_item = QGraphicsTextItem(class_text)
            font = QFont("Arial", 12)
            text_item.setFont(font)
            text_item.setDefaultTextColor(QColor(255, 255, 255))
            text_item.setPos(arrow_position.x() + 10, arrow_position.y() - 10)
            scene.addItem(text_item)
            arrow_group.label_text_items.append(text_item)

    def label_arrow(self, arrow_group, label):
        """Assign a class label to the selected arrow and update its color."""
        self.assign_label_to_arrow(arrow_group, label)
        self.selected_arrow = arrow_group

    def draw_arrows(self, arrow_coords, image_item, arrow_color=(0, 0, 0), arrow_size=24):
        """Draw arrows at the specified coordinates on all scenes."""
        self.synced_arrows.clear()

        for coord in arrow_coords:
            y, x = coord
            scene_x = image_item.pos().x() + x
            scene_y = image_item.pos().y() + y

            arrow_group_instances = []
            for scene in self.scenes:
                arrow_group = ClickableArrowGroup((scene_x, scene_y), arrow_color=QColor(*arrow_color),
                                                  arrow_size=arrow_size, manager=self)
                scene.addItem(arrow_group)
                arrow_group_instances.append(arrow_group)

            for arrow_group in arrow_group_instances:
                arrow_group.set_sibling_arrows(arrow_group_instances)

            self.synced_arrows.append(arrow_group_instances)

    def clear_annotations(self):
        """Clears all arrows from all scenes."""
        for scene in self.scenes:
            for arrow_group_instances in self.synced_arrows:
                for arrow_group in arrow_group_instances:
                    scene.removeItem(arrow_group)
        self.synced_arrows.clear()

    def highlight_arrow(self, arrow_group):
        """
        Highlights the given arrow and all its sibling arrows.
        """
        arrow_group.set_arrow_color(QColor(255, 0, 0))  # Highlight in red
        self.selected_arrow = arrow_group

    def zoom_in_on_arrow(self, arrow_group):
        """Zoom in on the selected arrow in the zoom viewer."""
        arrow_position = arrow_group.body_item.line().p1()
        crop_size = 256
        zoomed_pixmap, x_start, y_start = self.create_zoomed_crop(int(arrow_position.x()), int(arrow_position.y()),
                                                                  crop_size)

        if zoomed_pixmap:
            zoomed_pixmap = self.draw_arrow_on_zoomed_crop(zoomed_pixmap, arrow_position, x_start, y_start)
            self.zoom_viewer.update_image(zoomed_pixmap)

    def create_zoomed_crop(self, x_center, y_center, crop_size):
        """Creates a zoomed crop of the current image around the given center point."""
        current_pixmap = self.image_pixmap_item.pixmap()
        img_width = current_pixmap.width()
        img_height = current_pixmap.height()

        x_start = max(0, x_center - crop_size // 2)
        y_start = max(0, y_center - crop_size // 2)

        if x_start + crop_size > img_width:
            x_start = img_width - crop_size
        if y_start + crop_size > img_height:
            y_start = img_height - crop_size

        width = min(crop_size, img_width - x_start)
        height = min(crop_size, img_height - y_start)

        cropped_pixmap = current_pixmap.copy(x_start, y_start, width, height)
        zoom_factor = 2
        zoomed_pixmap = cropped_pixmap.scaled(crop_size * zoom_factor, crop_size * zoom_factor, Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)

        return zoomed_pixmap, x_start, y_start

    def draw_arrow_on_zoomed_crop(self, zoomed_pixmap, arrow_position, x_start, y_start):
        """Draw the selected arrow on the zoomed-in pixmap."""
        painter = QPainter(zoomed_pixmap)
        painter.setPen(QPen(Qt.red, 2))
        arrow_rel_x = arrow_position.x() - x_start
        arrow_rel_y = arrow_position.y() - y_start
        zoom_factor = 2
        arrow_rel_x_scaled = arrow_rel_x * zoom_factor
        arrow_rel_y_scaled = arrow_rel_y * zoom_factor
        arrow_size = 10
        painter.drawEllipse(QPointF(arrow_rel_x_scaled, arrow_rel_y_scaled), arrow_size, arrow_size)
        painter.end()

        return zoomed_pixmap

    def label_cluster(self, filename, arrow_group, label):
        """Assign the same label to all arrows in the same cluster as the selected arrow."""
        selected_index = next(i for i, arrows in enumerate(self.synced_arrows) if arrow_group in arrows)
        cluster_id = self.arrow_data[filename]['clusters'][selected_index]

        for i, cluster in enumerate(self.arrow_data[filename]['clusters']):
            if cluster == cluster_id:
                self.arrow_data[filename]['labels'][i] = label
                arrow_group = self.synced_arrows[i][0]
                self.assign_label_to_arrow(arrow_group, label)

    def assign_label_to_arrow(self, arrow_group, label):
        """Assign a class label to the arrow and update its color."""
        color = self.get_color_for_label(label)
        for arrow in arrow_group.sibling_arrows:
            arrow.set_arrow_color(color)

        arrow_group.label = label
        self.update_label_text_for_arrow(arrow_group, self.class_components[label])

    def store_arrow_data(self, filename, coords):
        """Store the arrow data, including coordinates and empty labels and clusters."""
        self.arrow_data[filename] = {
            'coords': coords,
            'labels': [None] * len(coords),
            'clusters': [None] * len(coords)
        }


# Main entry for testing
if __name__ == '__main__':
    app = QApplication(sys.argv)
    hdf5_file_path = r"C:\Users\wispy\OneDrive - University of Leeds\DATABACKUP\attention_unet_fl_f1.h5_COLLECTED_UNCERTAINTIES_2.h5"
    viewer = PatchImageViewer(hdf5_file_path=hdf5_file_path)
    viewer.show()
    logging.info("Application started")
    sys.exit(app.exec_())
