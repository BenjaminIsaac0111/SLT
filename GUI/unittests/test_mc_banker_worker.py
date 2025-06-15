import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication

from GUI.workers.MCBankerWorker import MCBankerWorker


def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_worker_progress(monkeypatch):
    app = qapp()
    cfg = {"MODEL_DIR": ".", "MODEL_NAME": "m.h5", "DATA_DIR": ".", "FILE_LIST": "list.txt"}

    calls = []

    def fake_run_config(cfg, output_dir=None, resume=False, progress_cb=None, **k):
        for i in range(3):
            progress_cb(i + 1, 3)

    import importlib
    mcb = importlib.import_module("GUI.workers.MCBankerWorker")
    monkeypatch.setattr(
        "DeepLearning.inference.mc_banker_gui.run_config", fake_run_config
    )
    monkeypatch.setattr(mcb, "run_config", fake_run_config)

    worker = MCBankerWorker(cfg)
    worker.signals.progress.connect(lambda p, t: calls.append((p, t)))
    worker.signals.finished.connect(lambda ok: calls.append(ok))
    worker.run()

    assert calls[:3] == [(1, 3), (2, 3), (3, 3)]
    assert calls[-1] is True

