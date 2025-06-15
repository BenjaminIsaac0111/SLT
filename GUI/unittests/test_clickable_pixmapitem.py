import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
    QMenu,
)
from PyQt5.QtGui import QPixmap, QMouseEvent
from PyQt5.QtCore import Qt, QPoint, QPointF

from GUI.views.ClickablePixmapItem import ClickablePixmapItem
from GUI.models.Annotation import Annotation
from GUI.configuration.configuration import CLASS_COMPONENTS


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def make_item(qapp):
    scene = QGraphicsScene()
    parent = QWidget()
    QGraphicsView(scene, parent)
    ann = Annotation(
        image_index=0,
        filename="img",
        coord=(0, 0),
        logit_features=np.array([], dtype=np.float32),
        uncertainty=0.5,
        mask_rle=None,
        mask_shape=None,
    )
    pixmap = QPixmap(10, 10)
    item = ClickablePixmapItem(annotation=ann, pixmap=pixmap, coord_pos=(2, 3))
    scene.addItem(item)
    return item, ann, scene, parent


def test_bounding_rect_scales_with_factor(qapp):
    item, ann, scene, parent = make_item(qapp)
    item.setScaleFactor(2.0)
    rect = item.boundingRect()
    assert rect.width() == pytest.approx(20.0)
    assert rect.height() == pytest.approx(20.0 + item.label_height)


def test_get_human_label_text_variants(qapp):
    item, ann, scene, parent = make_item(qapp)
    item.class_id = -2
    assert item.get_human_label_text() == "Unsure"
    item.class_id = -1
    assert item.get_human_label_text() == "Unlabelled"
    item.class_id = -3
    assert item.get_human_label_text() == "Artefact"
    item.class_id = next(iter(CLASS_COMPONENTS.keys()))
    assert item.get_human_label_text() == CLASS_COMPONENTS[item.class_id]
    item.class_id = 999
    assert item.get_human_label_text() == "ERROR"


def test_hover_and_selection_states(qapp):
    item, ann, scene, parent = make_item(qapp)
    item.hoverEnterEvent(None)
    assert item.hovered is True
    item.hoverLeaveEvent(None)
    assert item.hovered is False

    evt = QMouseEvent(
        QMouseEvent.MouseButtonPress,
        QPoint(1, 1),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    item.mousePressEvent(evt)
    assert item.selected is True
    item.mousePressEvent(evt)
    assert item.selected is False


def test_set_crop_class_emits_signal(qapp):
    item, ann, scene, parent = make_item(qapp)
    received = []

    def handler(data, cid):
        received.append((data, cid))

    item.class_label_changed.connect(handler)
    item.set_crop_class(2)
    assert ann.class_id == 2
    assert item.class_id == 2
    assert received and received[0][1] == 2
    assert received[0][0]["class_id"] == 2


def test_context_menu_event_triggers_actions(qapp, monkeypatch):
    item, ann, scene, parent = make_item(qapp)
    scene.context_menu_open = False

    def fake_exec_(self, pos):
        # Trigger first and last actions
        self.actions()[0].trigger()
        self.actions()[-1].trigger()
    monkeypatch.setattr(QMenu, "exec_", fake_exec_)

    event = type("E", (), {"screenPos": lambda self: QPointF(0, 0)})()
    item.contextMenuEvent(event)

    assert scene.context_menu_open is False
    assert ann.class_id == -3  # last triggered action
    assert item._view() is not None


class DummyPainter:
    """Minimal QPainter stub recording draw operations."""

    def __init__(self):
        self.calls = []

    def setRenderHint(self, *args):
        self.calls.append(("setRenderHint", args))

    def save(self):
        self.calls.append(("save",))

    def scale(self, sx, sy):
        self.calls.append(("scale", sx, sy))

    def drawPixmap(self, x, y, pixmap):
        self.calls.append(("drawPixmap", x, y, pixmap.size()))

    def restore(self):
        self.calls.append(("restore",))

    def setPen(self, pen):
        if hasattr(pen, "color"):
            color = pen.color().name()
            width = pen.width()
        else:
            color = int(pen)
            width = None
        self.calls.append(("setPen", color, width))

    def drawEllipse(self, *args):
        self.calls.append(("drawEllipse", args))

    def drawLine(self, *args):
        self.calls.append(("drawLine", args))

    def drawRect(self, rect):
        self.calls.append(("drawRect", rect.width(), rect.height()))

    def setFont(self, font):
        self.calls.append(("setFont", font.family(), font.pointSize()))

    def drawText(self, *args):
        text = args[-1]
        self.calls.append(("drawText", text))


def test_paint_draws_overlays_by_default(qapp):
    item, _, scene, _ = make_item(qapp)
    painter = DummyPainter()
    item.paint(painter, None, None)
    assert any(c[0] == "drawEllipse" for c in painter.calls)
    assert sum(1 for c in painter.calls if c[0] == "drawLine") == 4


def test_paint_skips_overlays_when_hidden(qapp):
    item, _, scene, _ = make_item(qapp)
    scene.overlays_visible = False
    painter = DummyPainter()
    item.paint(painter, None, None)
    assert not any(c[0] == "drawEllipse" for c in painter.calls)
    assert not any(c[0] == "drawLine" for c in painter.calls)


def test_paint_border_width_selected(qapp):
    item, _, _, _ = make_item(qapp)
    item.selected = True
    painter = DummyPainter()
    item.paint(painter, None, None)
    widths = [c[2] for c in painter.calls if c[0] == "setPen"]
    assert 4 in widths


def test_paint_prediction_glyphs(qapp):
    item, _, _, _ = make_item(qapp)
    cid = next(iter(CLASS_COMPONENTS.keys()))
    item.set_crop_class(cid)
    item.model_prediction = CLASS_COMPONENTS[cid]
    painter = DummyPainter()
    item.paint(painter, None, None)
    texts = [c[1] for c in painter.calls if c[0] == "drawText"]
    assert "✓" in texts

    painter = DummyPainter()
    other_cid = (cid + 1) % len(CLASS_COMPONENTS)
    item.model_prediction = CLASS_COMPONENTS[other_cid]
    item.paint(painter, None, None)
    texts = [c[1] for c in painter.calls if c[0] == "drawText"]
    assert "✗" in texts
