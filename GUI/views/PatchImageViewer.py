from typing import Tuple

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QSplitter, QGraphicsView, \
    QGraphicsScene
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
import logging

from GUI.views.ZoomedView import ZoomedView
from GUI.views.AnnotationView import AnnotationView


class PatchImageViewer(QWidget):
    """
    The main window of the application. It displays images, overlays, heatmaps, and handles user interactions.
    """

    # Signals emitted by the view
    fileSelected = pyqtSignal(int)
    arrowClicked = pyqtSignal(int)
    keyPressed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """
        Sets up the user interface layout, including file selector, image views, and zoomed-in view.
        """
        # File list widget
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMinimumWidth(200)

        # Image views
        self.image_view = AnnotationView()
        self.overlay_view = AnnotationView()
        self.heatmap_view = AnnotationView()

        # Zoomed image viewer
        self.zoomed_viewer = ZoomedView()

        # Layout setup
        self.setup_layout()

    def setup_layout(self):
        """
        Configures the layout of the UI components.
        """
        # Left side layout (image views)
        left_layout = QHBoxLayout()
        left_layout.addWidget(self.image_view)
        left_layout.addWidget(self.overlay_view)
        left_layout.addWidget(self.heatmap_view)

        left_container = QWidget()
        left_container.setLayout(left_layout)

        # Right side layout (file list and zoomed view)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.zoomed_viewer)
        right_layout.addWidget(self.file_list_widget)

        right_container = QWidget()
        right_container.setLayout(right_layout)

        # Splitter to divide the main window
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Window title and size
        self.setWindowTitle("Patch Image Viewer")
        self.resize(1200, 800)

    def connect_signals(self):
        """
        Connects signals emitted by UI components to internal slots.
        """
        self.file_list_widget.itemClicked.connect(self.on_file_item_clicked)

        # Forward signals from AnnotationViews
        self.image_view.arrowClicked.connect(self.on_arrow_clicked)
        self.overlay_view.arrowClicked.connect(self.on_arrow_clicked)
        self.heatmap_view.arrowClicked.connect(self.on_arrow_clicked)

    def on_file_item_clicked(self, item: QListWidgetItem):
        """
        Slot that handles file selection from the list widget.
        """
        index = self.file_list_widget.row(item)
        self.fileSelected.emit(index)

    def on_arrow_clicked(self, arrow_id: int):
        """
        Slot that handles arrow clicks in the image views.
        """
        self.arrowClicked.emit(arrow_id)

    def update_file_list(self, filenames: list):
        """
        Updates the file list widget with the provided filenames.

        :param filenames: List of filenames to display.
        """
        self.file_list_widget.clear()
        for filename in filenames:
            item = QListWidgetItem(filename)
            self.file_list_widget.addItem(item)
        logging.info("File list updated with %d entries.", len(filenames))

    def update_images(self, images: dict, filename: str):
        """
        Updates the image views with the provided images.

        :param images: Dictionary containing 'image', 'overlay', and 'heatmap' as QPixmap objects.
        :param filename: The filename corresponding to the images.
        """
        self.image_view.set_image(images['image'])
        self.overlay_view.set_image(images['overlay'])
        self.heatmap_view.set_image(images['heatmap'])
        self.setWindowTitle(f"Patch Image Viewer - {filename}")
        logging.info("Images updated for file '%s'.", filename)

    def update_annotations(self, annotations: dict):
        """
        Updates the annotations (arrows) in all image views.

        :param annotations: List of annotations to display. Each annotation can be a dict containing 'id' and 'coord'.
        """
        self.image_view.set_annotations(annotations)
        self.overlay_view.set_annotations(annotations)
        self.heatmap_view.set_annotations(annotations)
        logging.info("Annotations updated in all views.")

    def keyPressEvent(self, event):
        """
        Handles key press events.

        :param event: QKeyEvent.
        """
        self.keyPressed.emit(event.key())
        super().keyPressEvent(event)
