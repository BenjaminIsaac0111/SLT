import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Optional

from PyQt5.QtCore import QRunnable

from GUI.controllers.JobScheduler import JobScheduler
from GUI.models import JobDB


class DummyPool:
    def __init__(self):
        self.runnable: Optional[QRunnable] = None

    def start(self, runnable: QRunnable):
        # Defer execution until explicitly triggered in the test
        self.runnable = runnable


class DummyTask(QRunnable):
    def __init__(self):
        super().__init__()
        self.count = 0

    def run(self):
        self.count += 1


class PausableTask(DummyTask):
    def __init__(self):
        super().__init__()
        self.paused = False

    def pause(self):
        self.paused = True

    def resume_task(self):
        self.paused = False


def test_run_two_jobs_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(JobDB, "_DB_PATH", tmp_path / "jobs.db")
    pool = DummyPool()
    scheduler = JobScheduler(pool=pool)

    t1 = DummyTask()
    t2 = DummyTask()
    jid1 = scheduler.schedule_job("j1", t1)
    # execute first queued job
    assert pool.runnable is not None
    pool.runnable.run()

    jid2 = scheduler.schedule_job("j2", t2)
    assert pool.runnable is not None
    pool.runnable.run()

    jobs = JobDB.list_jobs(2)
    statuses = {j["id"]: j["status"] for j in jobs}
    assert statuses[jid1] == "completed"
    assert statuses[jid2] == "completed"
    assert jobs[0]["started_at"] and jobs[0]["finished_at"]
    assert t1.count == 1 and t2.count == 1


def test_pause_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(JobDB, "_DB_PATH", tmp_path / "jobs.db")
    pool = DummyPool()
    scheduler = JobScheduler(pool=pool)

    t = PausableTask()
    jid = scheduler.schedule_job("job", t)
    scheduler.pause_current()
    assert t.paused
    scheduler.resume_current()
    assert not t.paused

    assert pool.runnable is not None
    pool.runnable.run()

    jobs = JobDB.list_jobs(1)
    assert jobs[0]["id"] == jid
    assert jobs[0]["status"] == "completed"


