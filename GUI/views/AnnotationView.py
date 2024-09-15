# GUI/views/AnnotationView.py

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

    def set_annotations(self, annotations: list):
        """
        Adds annotations to the view.
        :param annotations: List of dictionaries with 'id' and 'position' keys.
        """
        # Add new annotations
        for annotation in annotations:
            arrow_item = self.create_arrow_item(annotation)
            arrow_item.clicked.connect(self.on_arrow_clicked)  # Connect the signal
            self.annotation_items[annotation['id']] = arrow_item
            self.scene.addItem(arrow_item)
            logging.debug(f"Added ArrowAnnotationItem with ID {annotation['id']} at {annotation['position']}.")
        logging.debug("All annotations set in AnnotationView.")

    def on_arrow_clicked(self, annotation_id: int):
        """
        Slot that handles when an annotation is clicked.
        Emits the arrowClicked signal with the annotation_id.
        """
        logging.debug(f"Arrow {annotation_id} clicked in AnnotationView.")
        self.arrowClicked.emit(annotation_id)

    def create_arrow_item(self, annotation: dict) -> ArrowAnnotationItem:
        """
        Creates an arrow item based on the annotation data.
        :param annotation: Dictionary containing annotation data ('id' and 'position').
        :return: ArrowAnnotationItem representing the arrow.
        """
        coord = annotation['position']  # (row, column)
        arrow_item = ArrowAnnotationItem(
            annotation_id=annotation['id'],
            color=QColor(255, 0, 0)  # Default color: Red
        )
        # Set position in the scene (x, y)
        arrow_item.setPos(coord[1], coord[0])  # x = column, y = row
        return arrow_item

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
