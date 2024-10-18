import logging
from typing import List

from PyQt5.QtWidgets import (
    QWidget, QComboBox, QPushButton, QVBoxLayout, QLabel,
    QProgressBar, QSpinBox, QHBoxLayout, QSlider, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QScrollArea
)
from PyQt5.QtCore import pyqtSignal, Qt, QRectF, QPointF, QEvent
from PyQt5.QtGui import QPixmap, QTransform, QWheelEvent, QMouseEvent, QPainter


class ClusteredCropsView(QWidget):
    # Define signals to communicate with the controller
    request_clustering = pyqtSignal()
    sample_cluster = pyqtSignal(int)
    sampling_parameters_changed = pyqtSignal(int, int)  # Emits (cluster_id, crops_per_cluster)

    def __init__(self):
        super().__init__()
        self.zoom_level = 0  # Initial zoom level
        self.sampled_crops = []  # Store sampled crops for dynamic rearrangement
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Clustering Controls
        clustering_layout = QHBoxLayout()
        self.clustering_button = QPushButton("Start Clustering")
        self.clustering_button.clicked.connect(self.request_clustering.emit)
        clustering_layout.addWidget(self.clustering_button)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Initially hidden
        clustering_layout.addWidget(self.progress_bar)

        main_layout.addLayout(clustering_layout)

        # Cluster Selection
        self.cluster_combo = QComboBox()
        self.cluster_combo.currentIndexChanged.connect(self.on_cluster_selected)
        main_layout.addWidget(QLabel("Select Cluster:"))
        main_layout.addWidget(self.cluster_combo)

        # Sampling Parameters
        sampling_layout = QHBoxLayout()
        sampling_layout.addWidget(QLabel("Number of Crops per Cluster:"))
        self.crops_spinbox = QSpinBox()
        self.crops_spinbox.setRange(1, 100)
        self.crops_spinbox.setValue(10)
        self.crops_spinbox.valueChanged.connect(self.on_crops_changed)
        sampling_layout.addWidget(self.crops_spinbox)
        main_layout.addLayout(sampling_layout)

        # Zoom Slider
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(-10)
        self.zoom_slider.setMaximum(10)
        self.zoom_slider.setValue(self.zoom_level)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)
        main_layout.addLayout(zoom_layout)

        # Graphics View for Cropped Images
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.graphics_view.viewport().installEventFilter(self)
        self.graphics_view.setMouseTracking(True)

        # Graphics Scene
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)

        main_layout.addWidget(QLabel("Sampled Crops:"))
        main_layout.addWidget(self.graphics_view)

        self.setLayout(main_layout)
        self.setWindowTitle("Clustered Crops Viewer")
        self.resize(800, 600)

    def populate_cluster_selection(self, cluster_ids):
        """
        Populates the cluster selection ComboBox with available cluster IDs.
        """
        self.cluster_combo.blockSignals(True)  # Prevent triggering selection signal during population
        self.cluster_combo.clear()
        for cid in cluster_ids:
            self.cluster_combo.addItem(f"Cluster {cid}", cid)
        self.cluster_combo.blockSignals(False)
        logging.debug(f"Populated cluster selection with IDs: {cluster_ids}")

    def on_cluster_selected(self, index):
        """
        Emits a signal to sample crops from the selected cluster.
        """
        cluster_id = self.cluster_combo.itemData(index)
        if cluster_id is not None:
            logging.debug(f"Cluster selected: {cluster_id}")
            self.sample_cluster.emit(cluster_id)

    def on_crops_changed(self, value):
        """
        Emits a signal when the number of crops per cluster changes.
        """
        current_cluster_id = self.get_selected_cluster_id()
        if current_cluster_id is not None:
            logging.debug(f"Number of crops per cluster changed to: {value}")
            self.sampling_parameters_changed.emit(current_cluster_id, value)

    def get_selected_cluster_id(self):
        """
        Retrieves the currently selected cluster ID.
        """
        return self.cluster_combo.currentData()

    def display_sampled_crops(self, sampled_crops):
        """
        Displays the sampled crops in a grid layout within the graphics scene.
        """
        self.sampled_crops = sampled_crops
        self.arrange_crops()

    def arrange_crops(self):
        """
        Arranges the sampled crops in a grid layout that adapts to the window size
        and supports zooming functionality. Images expand and contract to fill the space.
        """
        self.scene.clear()

        if not self.sampled_crops:
            logging.info("No sampled crops to display.")
            return

        # Get the viewport size
        viewport_rect = self.graphics_view.viewport().rect()
        viewport_width = viewport_rect.width()
        viewport_height = viewport_rect.height()
        logging.debug(f"Viewport size: {viewport_width}x{viewport_height}")

        # Calculate the number of columns based on the zoom level
        base_size = 200  # Base size for images at zoom level 0
        scale_factor = pow(1.2, self.zoom_level)  # Exponential scale factor
        image_size = base_size * scale_factor

        # Ensure image_size is not zero
        if image_size <= 0:
            image_size = 1  # Set minimum image size to prevent division by zero

        num_columns = max(1, int(viewport_width // (image_size + 10)))  # Include spacing in calculation

        # Adjust image size to fill the width
        total_spacing = 10 * (num_columns - 1)
        if num_columns > 0:
            image_size = (viewport_width - total_spacing) / num_columns
        else:
            num_columns = 1
            image_size = viewport_width - total_spacing

        logging.debug(f"Calculated image size: {image_size}, Number of columns: {num_columns}")

        # Arrange images in the scene
        row = 0
        col = 0
        x_offset = 0
        y_offset = 0
        max_row_height = 0

        for crop in self.sampled_crops:
            if crop['crop'].isNull():
                logging.warning(f"Invalid QPixmap for image index {crop['image_index']}. Skipping.")
                continue

            pixmap_item = QGraphicsPixmapItem(crop['crop'])
            pixmap_item.setTransformationMode(Qt.SmoothTransformation)
            pixmap_item.setFlag(QGraphicsItem.ItemIsSelectable, True)

            # Scale the pixmap item
            original_width = crop['crop'].width()
            if original_width == 0:
                logging.warning("Crop width is zero, skipping this image to prevent division by zero.")
                continue
            scale = image_size / original_width
            pixmap_item.setScale(scale)

            # Position the item
            pixmap_item.setPos(x_offset, y_offset)
            self.scene.addItem(pixmap_item)

            # Update offsets
            x_offset += image_size + 10  # Add spacing
            max_row_height = max(max_row_height, image_size)

            col += 1
            if col >= num_columns:
                col = 0
                row += 1
                x_offset = 0
                y_offset += max_row_height + 10  # Add spacing
                max_row_height = 0

        # Adjust the scene rect
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def on_zoom_changed(self, value):
        """
        Handles the zoom level change from the slider.
        """
        self.zoom_level = max(-10, min(value, 10))  # Limit zoom_level between -10 and 10
        logging.debug(f"Zoom level changed to: {self.zoom_level}")
        self.arrange_crops()

    def eventFilter(self, source, event):
        """
        Event filter to handle mouse wheel events for zooming.
        """
        if event.type() == QEvent.Wheel and isinstance(event, QWheelEvent):
            modifiers = event.modifiers()
            if modifiers == Qt.ControlModifier:
                delta = event.angleDelta().y() / 120  # 120 is the standard step
                self.zoom_level += delta
                self.zoom_level = max(-10, min(self.zoom_level, 10))  # Limit zoom_level
                self.zoom_slider.setValue(self.zoom_level)
                return True
        return super().eventFilter(source, event)

    def resizeEvent(self, event):
        """
        Overrides the resize event to rearrange crops when the window size changes.
        """
        super().resizeEvent(event)
        logging.debug("Window resized. Rearranging crops.")
        self.arrange_crops()

    def update_progress(self, progress):
        """
        Updates the progress bar with the current progress value.
        """
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        self.progress_bar.setValue(progress)
        logging.debug(f"Progress updated to: {progress}%")

    def reset_progress(self):
        """
        Resets and shows the progress bar.
        """
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        logging.debug("Progress bar reset.")

    def hide_progress_bar(self):
        """
        Hides the progress bar.
        """
        self.progress_bar.setVisible(False)
        logging.debug("Progress bar hidden.")
