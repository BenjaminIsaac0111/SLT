from functools import partial

from PyQt5.QtCore import pyqtSignal, QRectF, Qt
from PyQt5.QtGui import QPixmap, QFont, QFontMetrics, QPainter, QPen
from PyQt5.QtWidgets import QGraphicsObject, QApplication, QMenu, QAction

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.Annotation import Annotation


class ClickablePixmapItem(QGraphicsObject):
    """
    A QGraphicsObject that displays a QPixmap and emits signals when interacted with.
    It holds a reference to an Annotation instance.
    """
    class_label_changed = pyqtSignal(dict, int)  # Emits (annotation_dict, class_id)

    def __init__(self, annotation: Annotation, pixmap: QPixmap, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation = annotation
        self.pixmap = pixmap
        self.class_id = annotation.class_id
        self.setAcceptHoverEvents(True)
        self.hovered = False
        self.selected = False
        self.scale_factor = 1.0  # Initialize scale factor

        # Calculate font size based on screen DPI
        screen = QApplication.primaryScreen()
        logical_dpi = screen.logicalDotsPerInch()
        standard_dpi = 96.0  # Standard Windows DPI
        self.dpi_scaling = logical_dpi / standard_dpi
        self.base_font_size = 12  # Base font size

        # Set up font and metrics
        font_size = int(self.base_font_size * self.dpi_scaling)
        self.font = QFont("Arial", font_size)
        self.font_metrics = QFontMetrics(self.font)
        self.label_height = self.font_metrics.height()

    def setScaleFactor(self, scale):
        """
        Sets the scale factor for the image.
        """
        self.scale_factor = scale
        self.update()  # Trigger a repaint

    def set_crop_class(self, class_id: int):
        self.annotation.class_id = class_id
        self.class_id = class_id
        self.class_label_changed.emit(self.annotation.to_dict(), self.class_id)
        self.update()  # Redraw the item to reflect the new label

    def boundingRect(self):
        """
        Adjusts the bounding rectangle to include the scaled image and the label above it.
        """
        pixmap_width = self.pixmap.width() * self.scale_factor
        pixmap_height = self.pixmap.height() * self.scale_factor
        return QRectF(0, -self.label_height, pixmap_width, pixmap_height + self.label_height)

    def paint(self, painter: QPainter, option, widget):
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Draw the scaled pixmap
        painter.save()
        painter.scale(self.scale_factor, self.scale_factor)
        painter.drawPixmap(0, 0, self.pixmap)
        painter.restore()

        # Draw border around the scaled image
        pixmap_width = self.pixmap.width() * self.scale_factor
        pixmap_height = self.pixmap.height() * self.scale_factor
        pen = QPen(Qt.black if not self.hovered else Qt.blue)
        pen.setWidth(2 if self.selected or self.hovered else 1)
        painter.setPen(pen)
        painter.drawRect(QRectF(0, 0, pixmap_width, pixmap_height))

        # Draw label text directly above the image, aligned with its top edge
        painter.save()
        # Position the label above the image in the item's local coordinates
        label_x = 0  # Align text with the left edge of the image
        label_y = -self.label_height + self.font_metrics.ascent()  # Offset to position text above image
        painter.translate(label_x, label_y)
        painter.setFont(self.font)
        painter.setPen(Qt.black)

        # Display "Unsure" for class_id -2, "Unlabelled" for -1, or the regular class name
        if self.class_id == -2:
            label_text = "Unsure"
        elif self.class_id == -1:
            label_text = "Unlabelled"
        else:
            label_text = CLASS_COMPONENTS.get(self.class_id, "Unlabelled")

        painter.drawText(0, 0, label_text)
        painter.restore()

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selected = not self.selected  # Toggle selection
            self.update()

    def contextMenuEvent(self, event):
        self.scene().context_menu_open = True  # Set the flag in the scene
        self.menu = QMenu()

        # Add class actions
        for class_id, class_name in CLASS_COMPONENTS.items():
            action = QAction(class_name, self.menu)
            action.triggered.connect(partial(self.set_crop_class, class_id))
            self.menu.addAction(action)

        # Add a separator to distinguish the special actions
        self.menu.addSeparator()

        # Add "Unsure" option with class_id -2
        unsure_action = QAction("Unsure (?)", self.menu)
        unsure_action.triggered.connect(partial(self.set_crop_class, -2))
        self.menu.addAction(unsure_action)

        # Add "Unlabel" option with class_id -1
        unlabel_action = QAction("Unlabel (-)", self.menu)
        unlabel_action.triggered.connect(partial(self.set_crop_class, -1))
        self.menu.addAction(unlabel_action)

        self.menu.exec_(event.screenPos())
        self.scene().context_menu_open = False  # Unset the flag
