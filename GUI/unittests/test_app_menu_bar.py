import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PyQt5.QtWidgets import QApplication, QFileDialog

from GUI.views.AppMenuBar import AppMenuBar


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_new_project_action_emits_signal(qapp, monkeypatch, tmp_path):
    bar = AppMenuBar()
    expected = str(tmp_path / "data.h5")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (expected, ""))

    received = []
    bar.request_new_project.connect(lambda p: received.append(p))
    bar._pick_new_project_data()
    assert received == [expected]


def test_populate_recent_menu(qapp, monkeypatch):
    bar = AppMenuBar()
    monkeypatch.setattr(
        "GUI.models.RecentProjectsDB.get_recent_paths", lambda limit=5: ["p1.slt", "p2.slt"]
    )

    bar._populate_recent_menu()
    labels = [a.text() for a in bar._recent_menu.actions()]
    assert labels == ["p1.slt", "p2.slt"]

    received = []
    bar.request_load_project.connect(lambda p: received.append(p))
    bar._recent_menu.actions()[0].trigger()
    assert received == ["p1.slt"]
