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
        self.model_prediction = annotation.model_prediction  # Use model_prediction from annotation
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

        # Prefixes for labels
        self.human_prefix = "Human: "
        self.model_prefix = "Model: "

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
        # Adjust the bounding rect to include space for labels above the image
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

        # Determine the color for the human label based on class_id
        if self.class_id == -1:
            human_label_color = Qt.black  # Unlabeled
        elif self.class_id == -2:
            human_label_color = Qt.darkYellow  # Unsure/Amber
        elif self.class_id == -3:
            human_label_color = Qt.magenta  # Artifact
        elif self.class_id is not None and self.class_id >= 0 and self.class_id in CLASS_COMPONENTS:
            human_label_color = Qt.green  # Human-labeled (recognized class)
        else:
            human_label_color = Qt.red  # Unknown/error

        # Draw human label in the top-left corner above the image
        painter.save()
        painter.setFont(self.font)
        painter.setPen(human_label_color)
        human_label_text = self.human_prefix + self.get_human_label_text()
        label_x = 0  # Left-aligned
        label_y = -self.font_metrics.descent()  # Slight adjustment for baseline
        painter.drawText(label_x, label_y, human_label_text)
        painter.restore()

        # Draw model prediction in the top-right corner above the image
        if self.model_prediction:
            # Determine model_prediction class_id by reverse lookup in CLASS_COMPONENTS
            model_class_id = None
            for cid, cname in CLASS_COMPONENTS.items():
                if cname == self.model_prediction:
                    model_class_id = cid
                    break

            # If model_class_id matches human class_id, color is green, else red
            painter.save()
            painter.setFont(self.font)
            if model_class_id is not None and model_class_id == self.class_id:
                painter.setPen(Qt.green)  # Agree: green text
            else:
                painter.setPen(Qt.red)  # Disagree: red text

            prediction_text = self.model_prefix + self.model_prediction
            text_width = self.font_metrics.width(prediction_text)

            prediction_x = pixmap_width - text_width  # Right-aligned
            prediction_y = -self.font_metrics.descent()  # Slight adjustment for baseline
            painter.drawText(prediction_x, prediction_y, prediction_text)
            painter.restore()

    def get_human_label_text(self) -> str:
        """
        Returns the human label text to display based on class_id.
        """
        if self.class_id == -2:
            return "Unsure"
        elif self.class_id == -1 or None:
            return "Unlabelled"
        elif self.class_id == -3:
            return "Artefact"
        elif self.class_id in CLASS_COMPONENTS:
            # If it's a recognized class_id in CLASS_COMPONENTS
            return CLASS_COMPONENTS[self.class_id]
        else:
            return "ERROR"

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

        # Add numbered class actions
        for index, (class_id, class_name) in enumerate(CLASS_COMPONENTS.items(), start=0):
            action_text = f"{index}. {class_name}"  # Include the list number
            action = QAction(action_text, self.menu)
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

        # Add "Artifact" option with class_id -3
        artifact_action = QAction("Artifact (!)", self.menu)
        artifact_action.triggered.connect(partial(self.set_crop_class, -3))
        self.menu.addAction(artifact_action)

        self.menu.exec_(event.screenPos())
        self.scene().context_menu_open = False  # Unset the flag

