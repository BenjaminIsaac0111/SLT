import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

from GUI.views.TaskProgressWidget import MCBankerProgressWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_start_and_finish_reset_state(qapp, tmp_path):
    widget = MCBankerProgressWidget()
    outfile = tmp_path / "out.h5"
    widget.start(str(outfile), 10)
    assert widget.pause_btn.isEnabled()
    assert widget.pause_btn.text() == "Pause"
    assert not widget._paused

    # toggle pause then finish
    widget._toggle_pause()
    assert widget._paused
    assert widget.pause_btn.text() == "Resume"

    widget.finish()
    assert not widget._paused
    assert widget.pause_btn.text() == "Pause"
    assert not widget.pause_btn.isEnabled()

    # starting again should reset state
    widget.start(str(outfile), 5)
    assert widget.pause_btn.isEnabled()
    assert widget.pause_btn.text() == "Pause"
    assert not widget._paused
