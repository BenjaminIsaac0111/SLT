from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QWidget, QVBoxLayout

from .JobItemWidget import JobItemWidget


class TasksTab(QWidget):
    """Tab displaying background task progress and queued jobs."""

    def __init__(self, scheduler=None, parent=None):
        super().__init__(parent)
        self.scheduler = scheduler
        layout = QVBoxLayout(self)

        self.job_list = QListWidget()
        layout.addWidget(self.job_list)
        layout.addStretch(1)

        self.widgets = {}

        if self.scheduler is not None:
            self.scheduler.job_finished.connect(self.refresh_jobs)

        self.refresh_jobs()

    def _resume_job(self, job_id: int, config: dict) -> None:
        """Create a new worker from *config* and reschedule."""
        if not self.scheduler:
            return
        from GUI.workers import MCBankerWorker

        worker = MCBankerWorker(config, resume=True)
        worker.signals.progress.connect(self.widgets[job_id].update_progress)
        worker.signals.finished.connect(
            lambda ok, jid=job_id: self.widgets[jid].set_status(
                "completed" if ok else "failed"
            )
        )
        self.scheduler.resume_job(job_id, worker)

    # --------------------------------------------------------------
    def _remove_job(self, job_id: int) -> None:
        item_indexes = [i for i in range(self.job_list.count())]
        for idx in item_indexes:
            widget = self.job_list.itemWidget(self.job_list.item(idx))
            if getattr(widget, "job_id", None) == job_id:
                self.job_list.takeItem(idx)
                break
        self.widgets.pop(job_id, None)

    # --------------------------------------------------------------
    def add_job(self, job_id: int, name: str, config: dict, worker=None):
        """Create a new list item for *job_id* and optionally connect *worker*."""
        item = QListWidgetItem()
        widget = JobItemWidget(job_id, name, config)
        if worker is not None and hasattr(worker, "signals"):
            sigs = worker.signals
            if hasattr(sigs, "progress"):
                sigs.progress.connect(widget.update_progress)  # type: ignore
            if hasattr(sigs, "finished"):
                sigs.finished.connect(lambda ok, jid=job_id: widget.set_status("completed" if ok else "failed"))

        widget.request_pause.connect(
            lambda jid=job_id: self.scheduler.pause_current() if self.scheduler else None
        )
        widget.request_resume.connect(
            lambda jid=job_id: self._resume_job(jid, widget.config)
        )
        widget.request_cancel.connect(
            lambda jid=job_id: self.scheduler.cancel_current() if self.scheduler else None
        )
        widget.request_remove.connect(lambda jid=job_id: self._remove_job(jid))

        self.job_list.insertItem(0, item)
        self.job_list.setItemWidget(item, widget)
        self.widgets[job_id] = widget

    # --------------------------------------------------------------
    def refresh_jobs(self):
        """Reload job list from the database."""
        from GUI.models import JobDB

        jobs = JobDB.list_jobs(20)
        seen = set()
        for job in reversed(jobs):
            jid = job["id"]
            seen.add(jid)
            if jid not in self.widgets:
                self.add_job(jid, job["name"], job["config"])
            self.widgets[jid].set_status(job["status"])

        for jid in list(self.widgets):
            if jid not in seen:
                self._remove_job(jid)


