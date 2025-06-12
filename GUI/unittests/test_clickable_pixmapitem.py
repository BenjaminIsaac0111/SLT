import os
import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QWidget, QMenu
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
    view = QGraphicsView(scene, parent)
    ann = Annotation(
        image_index=0,
        filename="img",
        coord=(0, 0),
        logit_features=np.array([], dtype=np.float32),
        uncertainty=0.5,
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

    evt = QMouseEvent(QMouseEvent.MouseButtonPress, QPoint(1, 1), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    item.mousePressEvent(evt)
    assert item.selected is True
    item.mousePressEvent(evt)
    assert item.selected is False


def test_set_crop_class_emits_signal(qapp):
    item, ann, scene, parent = make_item(qapp)
    received = []
    item.class_label_changed.connect(lambda data, cid: received.append((data, cid)))
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

