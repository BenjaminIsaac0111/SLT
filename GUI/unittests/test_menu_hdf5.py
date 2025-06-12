from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog
import pytest
from gui_main import AppMenuBar


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_action_menu_emits_build_hdf5_signal(qapp, monkeypatch):
    mb = AppMenuBar()
    emitted = []
    mb.request_build_hdf5.connect(lambda *args: emitted.append(args))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "/data")
    files = iter(["train.csv", "model.h5", "out.h5"])
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (next(files), ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("out.h5", ""))
    ints = iter([(10, True), (4, True)])
    monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: next(ints))
    mb._pick_hdf5_args()
    assert emitted == [("/data", "train.csv", "model.h5", "out.h5", 10, 4)]
