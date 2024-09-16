from PyQt5.QtCore import pyqtSignal, QRectF, Qt, QPointF
from PyQt5.QtWidgets import QGraphicsObject
from PyQt5.QtGui import QPainter, QPen, QColor


class ArrowAnnotationItem(QGraphicsObject):
    """
    A custom QGraphicsObject that represents an arrow annotation.
    Emits a signal when clicked and holds relevant information like name and optional class label.
    """

    # Signal emitted when the arrow is clicked, sending its ID
    clicked = pyqtSignal(int)

    def __init__(self, arrow_info: dict):
        """
        Initializes the ArrowAnnotationItem using a dictionary for the arrow's properties.

        :param annotation_id: The unique identifier for the arrow.
        :param arrow_info: Dictionary containing information such as coords, cluster_id, color, name, class_label,
        tags, etc.
        """
        super().__init__()
        self.annotation_id = arrow_info.get('id')
        self.cluster_id = arrow_info.get('cid', -1)
        self.name = arrow_info.get('name', f'Annotation {self.annotation_id}')  # Default name if none provided
        self.class_label = arrow_info.get('class_label', None)  # Optional class label
        self.default_color = arrow_info.get('color', QColor(255, 0, 0))  # Default to red if not provided
        self.tags = arrow_info.get('tags', [])  # Default to empty list if no tags are provided
        self.is_hovered = False
        self.is_highlighted = False
        self.setFlags(QGraphicsObject.ItemIsSelectable | QGraphicsObject.ItemIsFocusable)
        self.setAcceptHoverEvents(True)  # Enable hover events

        # Set the position of the arrow based on the provided coordinates
        coords = arrow_info.get('position', (0, 0))  # Default to (0, 0) if no coordinates are provided
        self.setPos(QPointF(coords[1], coords[0]))  # coords are (row, col), which map to (y, x)

    def boundingRect(self) -> QRectF:
        """
        Defines the bounding rectangle of the item.
        """
        return QRectF(-10, -10, 20, 20)

    def paint(self, painter: QPainter, option, widget=None):
        """
        Paints the arrow based on the current state.
        """
        # Determine the color based on hover and highlighted states
        if self.is_hovered:
            current_color = QColor(255, 165, 0)  # Orange for hover
        elif self.is_highlighted:
            current_color = QColor(0, 255, 0)  # Green for highlighted
        else:
            current_color = self.default_color  # Default color (red)

        pen = QPen(current_color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(0, -10, 0, 10)  # Vertical line
        painter.drawLine(-10, 0, 10, 0)  # Horizontal line

    def mousePressEvent(self, event):
        """
        Handles mouse press events to emit the clicked signal and toggle highlight.
        """
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.annotation_id)
            self.set_highlighted(not self.is_highlighted)  # Toggle highlight on click
        super().mousePressEvent(event)

    def set_highlighted(self, highlighted: bool):
        """
        Highlights or unhighlights the arrow.
        """
        self.is_highlighted = highlighted
        self.update()  # Trigger a repaint

    def hoverEnterEvent(self, event):
        """
        Changes the arrow's state to hovered.
        """
        self.is_hovered = True
        self.update()  # Trigger a repaint to apply the hover color
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """
        Resets the arrow's hover state.
        """
        self.is_hovered = False
        self.update()  # Trigger a repaint to apply the original or highlighted color
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        """
        Handle changes to the item's state.
        """
        if change == QGraphicsObject.ItemSelectedChange:
            self.set_highlighted(value)
        return super().itemChange(change, value)

    def set_class_label(self, class_label: str):
        """
        Set the class label for the arrow.
        """
        self.class_label = class_label

    def set_cluster(self, cluster_id: int):
        """
        Set the cluster ID for the arrow.
        """
        self.cluster_id = cluster_id
