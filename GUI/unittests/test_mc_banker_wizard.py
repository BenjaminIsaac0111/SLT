import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import yaml
import pytest
from PyQt5.QtWidgets import QApplication

from GUI.views.MCBankerWizard import MCBankerWizard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_wizard_creates_config(tmp_path, qapp, monkeypatch):
    wiz = MCBankerWizard()
    wiz.select_page.rb_create.setChecked(True)

    model_file = tmp_path / "model.h5"
    model_file.touch()
    monkeypatch.setattr(
        MCBankerWizard,
        "_infer_model_spec",
        staticmethod(lambda path: ([16, 16, 1], 3)),
    )
    monkeypatch.setattr("GUI.models.MCConfigDB.save_config", lambda cfg: None)

    wiz.create_page.model_path.setEditText(str(model_file))
    wiz.create_page.data_dir.setText(str(tmp_path))
    wiz.create_page.file_list.setText(str(tmp_path / "files.txt"))
    wiz.create_page.output_dir.setText(str(tmp_path / "out"))
    wiz.create_page.mc_iter.setValue(5)

    wiz.exec_ = lambda: MCBankerWizard.Accepted
    path = wiz.get_config_path()
    assert path
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["BATCH_SIZE"] == 1
    assert cfg["SHUFFLE_BUFFER_SIZE"] == 256
    assert cfg["INPUT_SIZE"] == [16, 16, 1]
    assert cfg["OUT_CHANNELS"] == 3
    assert cfg["OUTPUT_DIR"].endswith("out")

