import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

from GUI.controllers.JobScheduler import JobScheduler
from GUI.views.TasksTab import TasksTab
from GUI.models import JobDB
from PyQt5.QtCore import QRunnable

class DummyPool:
    def __init__(self):
        self.runnable = None
    def start(self, runnable: QRunnable):
        self.runnable = runnable

class DummyTask(QRunnable):
    def run(self):
        pass

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_tasks_tab_refresh(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(JobDB, "_DB_PATH", tmp_path / "jobs.db")
    pool = DummyPool()
    scheduler = JobScheduler(pool=pool)
    tab = TasksTab(scheduler)

    t = DummyTask()
    scheduler.schedule_job("job", t)
    tab.refresh_jobs()
    assert tab.job_list.count() == 1
    widget = tab.job_list.itemWidget(tab.job_list.item(0))
    assert "running" in widget.meta.text()

    assert pool.runnable is not None
    pool.runnable.run()
    qapp.processEvents()
    tab.refresh_jobs()

    widget = tab.job_list.itemWidget(tab.job_list.item(0))
    assert "completed" in widget.meta.text()

