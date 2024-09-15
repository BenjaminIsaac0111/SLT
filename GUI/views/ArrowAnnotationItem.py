# GUI/views/ArrowAnnotationItem.py

from PyQt5.QtCore import pyqtSignal, QRectF, Qt, QPointF
from PyQt5.QtWidgets import QGraphicsObject
from PyQt5.QtGui import QPainter, QPen, QColor


class ArrowAnnotationItem(QGraphicsObject):
    """
    A custom QGraphicsObject that represents an arrow annotation.
    Emits a signal when clicked.
    """

    # Signal emitted when the arrow is clicked, sending its ID
    clicked = pyqtSignal(int)

    def __init__(self, annotation_id: int, color: QColor = QColor(255, 0, 0)):
        super().__init__()
        self.annotation_id = annotation_id
        self.color = color
        self.setFlags(QGraphicsObject.ItemIsSelectable | QGraphicsObject.ItemIsFocusable)
        self.setAcceptHoverEvents(True)

    def boundingRect(self) -> QRectF:
        """
        Defines the bounding rectangle of the item.
        """
        return QRectF(-10, -10, 20, 20)

    def paint(self, painter: QPainter, option, widget=None):
        """
        Paints the arrow.
        """
        pen = QPen(self.color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(0, -10, 0, 10)  # Vertical line
        painter.drawLine(-10, 0, 10, 0)  # Horizontal line

    def mousePressEvent(self, event):
        """
        Handles mouse press events to emit the clicked signal.
        """
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.annotation_id)
        super().mousePressEvent(event)

    def set_highlighted(self, highlighted: bool):
        """
        Highlights or unhighlights the arrow.
        """
        if highlighted:
            self.color = QColor(0, 255, 0)  # Green for highlighted
        else:
            self.color = QColor(255, 0, 0)  # Red for default
        self.update()  # Trigger a repaint
