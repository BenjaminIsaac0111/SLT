import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
    wiz.create_page.output_file.setText(str(tmp_path / "out.h5"))
    wiz.create_page.mc_iter.setValue(5)
    wiz.create_page.temperature.setValue(1.5)
    wiz.create_page.unc_type.setCurrentText("variance")
    wiz.create_page.subset.setValue(10)

    wiz.exec_ = lambda: MCBankerWizard.Accepted
    cfg = wiz.get_config()
    assert cfg
    assert cfg["BATCH_SIZE"] == 1
    assert cfg["SHUFFLE_BUFFER_SIZE"] == 256
    assert cfg["INPUT_SIZE"] == [16, 16, 1]
    assert cfg["OUT_CHANNELS"] == 3
    assert cfg["OUTPUT_FILE"].endswith("out.h5")
    assert cfg["UNCERTAINTY_TYPE"] == "variance"
    assert cfg["MC_N_ITER"] == 5
    assert cfg["TEMPERATURE"] == 1.5
    assert cfg["N_SAMPLES"] == 10


def test_wizard_loads_recent_config(tmp_path, qapp, monkeypatch):
    sample_cfg = {
        "MODEL_DIR": str(tmp_path),
        "MODEL_NAME": "model.h5",
        "DATA_DIR": str(tmp_path / "data"),
        "FILE_LIST": str(tmp_path / "files.txt"),
        "OUTPUT_FILE": str(tmp_path / "out.h5"),
        "MC_N_ITER": 7,
        "UNCERTAINTY_TYPE": "bald",
        "TEMPERATURE": 2.0,
        "N_SAMPLES": 4,
        "INPUT_SIZE": [32, 32, 1],
        "OUT_CHANNELS": 2,
    }

    monkeypatch.setattr(
        "GUI.models.MCConfigDB.get_recent_configs", lambda limit=10: [sample_cfg]
    )
    monkeypatch.setattr("GUI.models.MCConfigDB.save_config", lambda cfg: None)

    wiz = MCBankerWizard()

    wiz.create_page.recent.setCurrentIndex(1)

    assert wiz.create_page.model_path.currentText().endswith("model.h5")
    assert wiz.create_page.data_dir.text().endswith("data")
    assert wiz.create_page.file_list.text().endswith("files.txt")
    assert wiz.create_page.output_file.text().endswith("out.h5")
    assert wiz.create_page.mc_iter.value() == 7
    assert wiz.create_page.unc_type.currentText().lower() == "bald"
    assert wiz.create_page.temperature.value() == 2.0
    assert wiz.create_page.subset.value() == 4
    assert wiz.create_page.in_size.text() == "32x32x1"
    assert wiz.create_page.out_channels.text() == "2"



