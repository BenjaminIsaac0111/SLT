import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PyQt5.QtWidgets import QApplication

from GUI.views.CrossValidationDialog import CrossValidationDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_get_config_returns_values(qapp, tmp_path):
    dlg = CrossValidationDialog()
    dlg.img_edit.setText(str(tmp_path))
    dlg.out_edit.setText(str(tmp_path / "out"))
    dlg.sp_folds.setValue(5)
    dlg.exec_ = lambda: CrossValidationDialog.Accepted
    cfg = dlg.get_config()
    assert cfg == (str(tmp_path), str(tmp_path / "out"), 5)
