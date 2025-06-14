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


def test_wizard_creates_config(tmp_path, qapp):
    wiz = MCBankerWizard()
    wiz.select_page.rb_create.setChecked(True)

    wiz.create_page.model_dir.setText(str(tmp_path))
    wiz.create_page.model_name.setText("model.h5")
    wiz.create_page.data_dir.setText(str(tmp_path))
    wiz.create_page.train_list.setText(str(tmp_path / "train.txt"))
    wiz.create_page.batch_size.setValue(2)
    wiz.create_page.shuffle_buf.setValue(4)
    wiz.create_page.out_channels.setValue(3)
    wiz.create_page.in_h.setValue(16)
    wiz.create_page.in_w.setValue(16)
    wiz.create_page.in_c.setValue(1)
    wiz.create_page.mc_iter.setValue(5)

    wiz.exec_ = lambda: MCBankerWizard.Accepted
    path = wiz.get_config_path()
    assert path
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["MODEL_DIR"] == str(tmp_path)
    assert cfg["BATCH_SIZE"] == 2
    assert cfg["INPUT_SIZE"] == [16, 16, 1]

