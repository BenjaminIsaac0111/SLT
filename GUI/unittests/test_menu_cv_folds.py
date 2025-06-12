from PyQt5.QtWidgets import QFileDialog, QApplication, QInputDialog
import pytest
from gui_main import AppMenuBar


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_action_menu_emits_create_folds_signal(qapp, monkeypatch):
    mb = AppMenuBar()
    emitted = []
    mb.request_create_folds.connect(lambda d, o, s: emitted.append((d, o, s)))
    dirs = iter(["/in", "/out"])
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: next(dirs))
    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: (5, True))
    mb._pick_folds_dirs()
    assert emitted == [("/in", "/out", 5)]

