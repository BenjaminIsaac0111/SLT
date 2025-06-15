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


def test_worker_stop_while_paused(monkeypatch):
    app = qapp()
    cfg = {"MODEL_DIR": ".", "MODEL_NAME": "m.h5", "DATA_DIR": ".", "FILE_LIST": "list.txt"}

    events = []

    def fake_run_config(cfg, output_dir=None, resume=False, progress_cb=None, should_abort=None, should_pause=None, **k):
        import time
        for i in range(3):
            while should_pause():
                if should_abort():
                    events.append("stopped")
                    return
                time.sleep(0.01)
            if should_abort():
                events.append("stopped")
                return
            if progress_cb:
                progress_cb(i + 1, 3)
        
    import importlib, threading, time
    mcb = importlib.import_module("GUI.workers.MCBankerWorker")
    mp = "DeepLearning.inference.mc_banker_gui"
    monkeypatch.setattr(f"{mp}.run_config", fake_run_config)
    monkeypatch.setattr(f"{mp}._prepare_config", lambda c, **k: c)
    monkeypatch.setattr(f"{mp}._infer_model_spec", lambda p: ([1, 1, 1], 1))
    monkeypatch.setattr(mcb, "run_config", fake_run_config)

    worker = MCBankerWorker(cfg)
    worker.pause()
    worker.signals.finished.connect(lambda ok: events.append(ok))

    th = threading.Thread(target=worker.run)
    th.start()
    time.sleep(0.05)
    worker.cancel()
    th.join(timeout=1)
    assert not th.is_alive()
    assert "stopped" in events

