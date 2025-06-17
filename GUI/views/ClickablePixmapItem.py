from functools import partial
from typing import Tuple

import numpy as np
from PyQt5.QtCore import QRectF, Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QFontMetrics, QPainter, QPen, QImage
from PyQt5.QtWidgets import QGraphicsObject, QApplication, QMenu, QAction

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.annotations import AnnotationBase, MaskAnnotation


class ClickablePixmapItem(QGraphicsObject):
    """
    A QGraphicsObject that displays a QPixmap and emits signals when interacted with.
    It holds a reference to an Annotation instance.
    """
    class_label_changed = pyqtSignal(dict, int)

    def __init__(self, annotation: AnnotationBase, pixmap: QPixmap, coord_pos: Tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation = annotation
        self.pixmap = pixmap
        self.coord_pos = coord_pos
        self.class_id = annotation.class_id
        self.model_prediction = annotation.model_prediction

        self.setAcceptHoverEvents(True)
        self.hovered = False
        self.selected = False
        self.scale_factor = 1.0

        screen = QApplication.primaryScreen()
        self.dpi_scaling = screen.logicalDotsPerInch() / 96.0
        font_size = int(12 * self.dpi_scaling)
        self.font = QFont("Arial", font_size)
        self.font_metrics = QFontMetrics(self.font)
        self.label_height = self.font_metrics.height()

        self.human_prefix = "Human: "
        self.model_prefix = "Model: "

    # ----- API -------------------------------------------------------
    def setScaleFactor(self, scale: float):
        self.scale_factor = scale
        self.update()

    def set_crop_class(self, class_id: int):
        self.annotation.class_id = class_id
        self.class_id = class_id
        self.class_label_changed.emit(self.annotation.to_dict(), self.class_id)
        self.update()

    # ----- geometry --------------------------------------------------
    def boundingRect(self):
        w = self.pixmap.width() * self.scale_factor
        h = self.pixmap.height() * self.scale_factor
        return QRectF(0, -self.label_height, w, h + self.label_height)

    # ----- drawing ---------------------------------------------------
    def paint(self, painter: QPainter, option, widget):
        """Draw pixmap, border, labels, and ✓/✗."""
        scene = self.scene()
        # 1) Draw the scaled image
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.save()
        painter.scale(self.scale_factor, self.scale_factor)
        painter.drawPixmap(0, 0, self.pixmap)
        painter.restore()

        # 2) Compute scaled dimensions
        pw = self.pixmap.width() * self.scale_factor
        ph = self.pixmap.height() * self.scale_factor

        if scene and getattr(scene, "overlays_visible", True):
            if isinstance(self.annotation, MaskAnnotation) and self.annotation.mask is not None:
                arr = (self.annotation.mask > 0).astype(np.uint8) * 255
                h, w = arr.shape
                qimg = QImage(arr.data, w, h, QImage.Format_Grayscale8)
                mask_pix = QPixmap.fromImage(qimg).scaled(pw, ph)
                painter.setOpacity(0.5)
                painter.drawPixmap(0, 0, mask_pix)
                painter.setOpacity(1.0)
            else:
                x0 = self.coord_pos[0] * self.scale_factor
                y0 = self.coord_pos[1] * self.scale_factor
                r = max(8.0, min(pw, ph) // 40)

                painter.setPen(QPen(Qt.green, 2))
                painter.drawEllipse(QPointF(x0, y0), r, r)

                centre_r = max(1, r // 6)  # keep it visible but small    # <<<
                painter.save()  # <<<
                painter.setPen(Qt.NoPen)  # <<<
                painter.setBrush(Qt.black)  # solid fill, matches outline   # <<<
                painter.drawEllipse(QPointF(x0, y0), centre_r, centre_r)  # <<<
                painter.restore()

                painter.setPen(QPen(Qt.black, 1))
                painter.drawLine(x0, 0, x0, y0 - r)
                painter.drawLine(x0, y0 + r, x0, ph)
                painter.drawLine(0, y0, x0 - r, y0)
                painter.drawLine(x0 + r, y0, pw, y0)

        pen = QPen(Qt.darkGray if self.hovered else Qt.black)
        pen.setWidth(4 if (self.selected or self.hovered or self.annotation.is_manual) else 1)
        painter.setPen(pen)
        painter.drawRect(QRectF(0, 0, pw, ph))

        # 4) Human label (top-left)
        if self.class_id == -1:
            human_col = Qt.black
        elif self.class_id == -2:
            human_col = Qt.darkYellow
        elif self.class_id == -3:
            human_col = Qt.magenta
        elif self.class_id in CLASS_COMPONENTS:
            human_col = Qt.darkGreen
        else:
            human_col = Qt.red

        painter.save()
        painter.setFont(self.font)
        painter.setPen(human_col)
        painter.drawText(
            0,
            -self.font_metrics.descent(),
            self.human_prefix + self.get_human_label_text()
        )
        painter.restore()

        # 5) Model prediction + ✓/✗ glyph (top-right)
        if self.model_prediction:
            model_cid = next(
                (cid for cid, nm in CLASS_COMPONENTS.items() if nm == self.model_prediction),
                None
            )
            agree = (model_cid == self.class_id)
            glyph = "✓" if agree else "✗"

            painter.save()
            painter.setFont(self.font)
            painter.setPen(Qt.darkGreen if agree else Qt.red)

            text = self.model_prefix + self.model_prediction
            gw = self.font_metrics.width(glyph)
            tw = self.font_metrics.width(text)

            x_text = pw - tw
            x_glyph = x_text - gw - 4
            y = -self.font_metrics.descent()

            painter.drawText(x_glyph, y, glyph)
            painter.drawText(x_text, y, text)
            painter.restore()

    # ----- helpers / events ------------------------------------------
    def get_human_label_text(self):
        if self.class_id == -2:
            return "Unsure"
        if self.class_id in (-1, None):
            return "Unlabelled"
        if self.class_id == -3:
            return "Artefact"
        return CLASS_COMPONENTS.get(self.class_id, "ERROR")

    def hoverEnterEvent(self, e):
        self.hovered = True;
        self.update();
        super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e):
        self.hovered = False;
        self.update();
        super().hoverLeaveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.selected = not self.selected
            self.update()

    def contextMenuEvent(self, event):
        self.scene().context_menu_open = True
        menu = QMenu()
        for index, (cid, cname) in enumerate(CLASS_COMPONENTS.items()):
            act = QAction(f"{index}. {cname}", menu)
            act.triggered.connect(partial(self.set_crop_class, cid))
            menu.addAction(act)
        menu.addSeparator()
        for cid, title in [(-2, "Unsure (?)"), (-1, "Unlabel (-)"), (-3, "Artifact (!)")]:
            act = QAction(title, menu)
            act.triggered.connect(partial(self.set_crop_class, cid))
            menu.addAction(act)
        menu.exec_(event.screenPos())
        self.scene().context_menu_open = False
        self.scene().views()[0].parentWidget().setFocus(Qt.OtherFocusReason)

    def _view(self):
        views = self.scene().views() if self.scene() else []
        return views[0] if views else None
