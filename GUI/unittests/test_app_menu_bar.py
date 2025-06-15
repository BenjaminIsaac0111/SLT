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
