import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from GUI.models import JobDB


def test_add_and_list_jobs(tmp_path, monkeypatch):
    db_file = tmp_path / "jobs.db"
    monkeypatch.setattr(JobDB, "_DB_PATH", db_file)

    jid1 = JobDB.add_job("job1", {"a": 1})
    jid2 = JobDB.add_job("job2", {"b": 2})

    jobs = JobDB.list_jobs(2)
    assert jobs[0]["id"] == jid2
    assert jobs[1]["id"] == jid1
    assert jobs[0]["status"] == "queued"

    JobDB.update_status(jid1, "running")
    jobs = JobDB.list_jobs(2)
    running = next(j for j in jobs if j["id"] == jid1)
    assert running["status"] == "running"
    assert running["started_at"]

    JobDB.update_status(jid1, "completed")
    jobs = JobDB.list_jobs(2)
    finished = next(j for j in jobs if j["id"] == jid1)
    assert finished["finished_at"]


