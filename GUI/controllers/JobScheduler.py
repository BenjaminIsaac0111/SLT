from __future__ import annotations

"""Simple queued job scheduler using QThreadPool."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from GUI.models import JobDB


class _RunnableWrapper(QRunnable):
    """Internal wrapper that notifies scheduler when done."""

    def __init__(self, job_id: int, runnable: QRunnable, scheduler: "JobScheduler") -> None:
        super().__init__()
        self.job_id = job_id
        self.runnable = runnable
        self.scheduler = scheduler

    def run(self) -> None:  # pragma: no cover - executed in worker thread
        try:
            self.runnable.run()
            self.scheduler._job_done(self.job_id, True)
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("Job %s failed: %s", self.job_id, exc)
            self.scheduler._job_done(self.job_id, False)


class JobScheduler(QObject):
    """Queue jobs and run them one at a time."""

    job_started = pyqtSignal(int)
    job_finished = pyqtSignal(int, bool)

    def __init__(self, pool: Optional[QThreadPool] = None) -> None:
        super().__init__()
        self._pool = pool or QThreadPool.globalInstance()
        self._queue: List[Tuple[int, QRunnable]] = []
        self._current: Optional[Tuple[int, QRunnable]] = None
        self._paused_jobs: set[int] = set()

    # ------------------------------------------------------------------
    def schedule_job(self, name: str, runnable: QRunnable, config: Optional[Dict[str, Any]] = None) -> int:
        """Add *runnable* to the queue and start if idle."""
        job_id = JobDB.add_job(name, config)
        self._queue.append((job_id, runnable))
        if self._current is None:
            self._start_next()
        return job_id

    def _start_next(self) -> None:
        if not self._queue:
            return
        job_id, runnable = self._queue.pop(0)
        self._current = (job_id, runnable)
        JobDB.update_status(job_id, "running")
        self.job_started.emit(job_id)
        wrapper = _RunnableWrapper(job_id, runnable, self)
        self._pool.start(wrapper)

    def _job_done(self, job_id: int, ok: bool) -> None:
        if job_id in self._paused_jobs:
            self._paused_jobs.remove(job_id)
        else:
            JobDB.update_status(job_id, "completed" if ok else "failed")
        self.job_finished.emit(job_id, ok)
        self._current = None
        self._start_next()

    # ------------------------------------------------------------------
    def pause_current(self) -> None:
        if self._current is None:
            return
        job_id, runnable = self._current
        if hasattr(runnable, "cancel"):
            runnable.cancel()  # type: ignore[attr-defined]
            JobDB.update_status(job_id, "paused")
            self._paused_jobs.add(job_id)

    def resume_job(self, job_id: int, runnable: QRunnable) -> None:
        """Resume a paused job by re-queuing a new *runnable*."""
        JobDB.update_status(job_id, "queued")
        self._queue.insert(0, (job_id, runnable))
        if self._current is None:
            self._start_next()

    def cancel_current(self) -> None:
        if self._current is None:
            return
        job_id, runnable = self._current
        if hasattr(runnable, "cancel"):
            runnable.cancel()  # type: ignore[attr-defined]
            JobDB.update_status(job_id, "canceled")

    def move_to_front(self, job_id: int) -> None:
        """Move a queued job to the front if present."""
        for i, item in enumerate(self._queue):
            if item[0] == job_id:
                self._queue.insert(0, self._queue.pop(i))
                break

