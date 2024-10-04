# GUI/views/AnnotationView.py
import math
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QColor
import logging

from GUI.views.ArrowAnnotationItem import ArrowAnnotationItem


class AnnotationView(QGraphicsView):
    """
    A custom QGraphicsView that displays an image and annotations (arrows).
    Captures user interactions on the annotations and emits signals.
    """

    # Signals
    arrowClicked = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item = None
        self.annotation_items = {}  # Dictionary to store arrow_id: ArrowAnnotationItem
        self.selected_annotation_id = -1  # No selection initially
        self.clustered_coords = {}  # To store the cluster data
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def set_image(self, pixmap: QPixmap):
        """
        Sets the image to display and ensures it fits the view.
        """
        self.scene.clear()  # Clear previous items
        self.image_item = self.scene.addPixmap(pixmap)
        self.image_item.setZValue(0)  # Ensure image is below annotations
        self.fitInView(self.image_item, Qt.KeepAspectRatio)  # Ensure the image fits the view
        logging.debug("Image set in AnnotationView.")

    def set_annotations(self, annotations: dict):
        # Clear previous annotations safely
        self.annotation_items.clear()

        # Add new annotations and assign cluster labels
        for annotation in annotations:
            # Create the arrow item using the dictionary directly
            arrow_item = ArrowAnnotationItem(annotation)
            arrow_item.clicked.connect(self.on_arrow_clicked)  # Connect the signal
            self.annotation_items[annotation['id']] = arrow_item
            self.scene.addItem(arrow_item)

            # Ensure the cluster ID is set correctly
            arrow_item.cluster_id = annotation.get('cid', -1)  # Default to -1 if no cluster ID
            logging.debug(
                f"Added ArrowAnnotationItem with ID {annotation['id']} at {annotation['position']}, "
                f"cluster {annotation['cid']}.")

    def set_clusters(self, clustered_coords: dict):
        """
        Sets the clustered coordinates. This should be a dictionary where each cluster ID maps to a list of coordinates.
        :param clustered_coords: Dictionary with cluster ID as keys and lists of coordinates as values.
        """
        self.clustered_coords = clustered_coords
        logging.debug(f"Clusters set with {len(clustered_coords)} clusters.")

    def on_arrow_clicked(self, annotation_id: int):
        """
        Slot that handles when an annotation is clicked.
        Emits the arrowClicked signal with the annotation_id.
        """
        logging.debug(f"Arrow {annotation_id} clicked in AnnotationView.")
        self.arrowClicked.emit(annotation_id)

    def highlight_arrow(self, annotation_id: int):
        """
        Highlights the specified arrow and unhighlights others.
        :param annotation_id: The ID of the arrow to highlight.
        """
        # Unhighlight previously selected arrow
        if self.selected_annotation_id in self.annotation_items:
            self.annotation_items[self.selected_annotation_id].set_highlighted(False)
            logging.debug(f"Arrow {self.selected_annotation_id} unhighlighted.")

        # Highlight the new arrow
        if annotation_id in self.annotation_items:
            self.annotation_items[annotation_id].set_highlighted(True)
            self.selected_annotation_id = annotation_id
            logging.debug(f"Arrow {annotation_id} highlighted.")
        else:
            self.selected_annotation_id = -1
            logging.debug("No arrow highlighted.")

    def highlight_cluster(self, cluster_id: int):
        # Unhighlight all previously highlighted arrows
        for arrow_item in self.annotation_items.values():
            arrow_item.set_highlighted(False)

        # Highlight all arrows that belong to the selected cluster
        for arrow_id, arrow_item in self.annotation_items.items():
            logging.debug(f"Checking arrow {arrow_id} with cluster {arrow_item.cluster_id} for cluster {cluster_id}.")
            if arrow_item.cluster_id == cluster_id:
                arrow_item.set_highlighted(True)
                logging.debug(f"Arrow {arrow_id} in cluster {cluster_id} highlighted.")

    def get_selected_arrow_id(self) -> int:
        """
        Retrieves the currently selected arrow ID.
        :return: ID of the selected arrow.
        """
        return self.selected_annotation_id

    def resizeEvent(self, event):
        """
        Ensures the image fits the view when the window is resized.
        """
        super().resizeEvent(event)  # Call the parent resize event
        if self.image_item is not None:
            self.fitInView(self.image_item, Qt.KeepAspectRatio)
            logging.debug("AnnotationView resized and image re-fitted.")

