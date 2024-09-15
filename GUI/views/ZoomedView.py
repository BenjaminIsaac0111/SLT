from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage, QPen
import logging

from GUI.views.ArrowAnnotationItem import ArrowAnnotationItem  # Ensure correct import path

class ZoomedView(QWidget):
    """
    Displays a zoomed-in view of a selected region.
    Ensures that the image always fits the view upon resizing.
    Also places an arrow at the specified location within the zoomed image.
    """

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """
        Sets up the user interface components.
        """
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.image_item = None
        self.arrow_item = None  # To keep track of the arrow in the zoomed view

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

    def calculate_crop_bounds(self, coord: tuple, original_image_size: tuple, crop_size: int):
        """
        Calculate the cropping bounds and the arrow's position within the crop.

        :param coord: The (row, col) coordinate of the arrow in the original image.
        :param original_image_size: The size of the original image (height, width).
        :param crop_size: The fixed size of the crop (e.g., 256x256).
        :return: The cropping bounds (top-left corner of the crop) and the arrow's position within the crop.
        """
        row, col = coord
        original_height, original_width = original_image_size
        x_start = max(0, col - crop_size // 2)
        y_start = max(0, row - crop_size // 2)

        if x_start + crop_size > original_width:
            x_start = original_width - crop_size
        if y_start + crop_size > original_height:
            y_start = original_height - crop_size

        x_start = max(x_start, 0)
        y_start = max(y_start, 0)

        width_crop = min(crop_size, original_width - x_start)
        height_crop = min(crop_size, original_height - y_start)

        arrow_rel_x = col - x_start
        arrow_rel_y = row - y_start

        logging.debug(f"Calculated crop bounds: x_start={x_start}, y_start={y_start}, width={width_crop}, height={height_crop}")
        logging.debug(f"Arrow relative position within crop: ({arrow_rel_x}, {arrow_rel_y})")

        return (x_start, y_start), (arrow_rel_x, arrow_rel_y)

    def update_zoomed_image(self, pixmap: QPixmap, coord: tuple, original_image_size: tuple, crop_size: int = 256, zoom_factor: int = 2):
        """
        Updates the zoomed view with the given pixmap and places an arrow at the correct position.

        :param pixmap: The QPixmap to display.
        :param coord: The (row, column) coordinate of the selected arrow in the original image.
        :param original_image_size: The size of the original image (height, width).
        :param crop_size: The fixed size of the crop to display in the zoomed view (e.g., 256x256).
        :param zoom_factor: The factor by which the cropped image was zoomed.
        """
        logging.debug("Updating zoomed image.")

        if self.image_item:
            self.scene.removeItem(self.image_item)
            self.image_item = None

        if self.arrow_item:
            self.scene.removeItem(self.arrow_item)
            self.arrow_item = None

        # Add the zoomed pixmap to the scene
        self.image_item = self.scene.addPixmap(pixmap)
        self.image_item.setZValue(-1)  # Ensure the image is below the arrow

        # Calculate the relative position of the arrow within the zoomed image
        (x_start, y_start), (arrow_rel_x, arrow_rel_y) = self.calculate_crop_bounds(coord, original_image_size, crop_size)

        # Scale the arrow position according to the zoom factor
        arrow_rel_x_scaled = arrow_rel_x * zoom_factor
        arrow_rel_y_scaled = arrow_rel_y * zoom_factor

        logging.debug(f"Arrow scaled position: ({arrow_rel_x_scaled}, {arrow_rel_y_scaled})")

        # Draw the arrow as an ellipse (you can customize this to use ArrowAnnotationItem if needed)
        self.arrow_item = ArrowAnnotationItem(
            annotation_id=0,  # Since it's a single arrow, ID can be arbitrary
            color=QColor(255, 0, 0)  # Red color for visibility
        )
        self.arrow_item.setPos(arrow_rel_x_scaled, arrow_rel_y_scaled)
        self.scene.addItem(self.arrow_item)

        # Fit the view to the zoomed image
        self.view.fitInView(self.image_item, Qt.KeepAspectRatio)
        logging.info("Zoomed image updated successfully.")

    def resizeEvent(self, event):
        """
        Overrides the resize event to ensure the image always fits the view upon resizing.

        :param event: The resize event.
        """
        super().resizeEvent(event)  # Call the base class implementation

        if self.image_item is not None:
            # Re-fit the entire scene in the view after resizing
            self.view.fitInView(self.image_item, Qt.KeepAspectRatio)
            logging.debug("ZoomedView resized. Image re-fitted to view.")
