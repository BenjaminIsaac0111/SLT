from __future__ import annotations

"""Widget listing scheduled jobs with progress controls."""

from typing import Dict

from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QWidget, QVBoxLayout

from .JobItemWidget import JobItemWidget


class TasksTab(QWidget):
    """Tab displaying background tasks."""

    def __init__(self, scheduler=None, parent=None):
        super().__init__(parent)
        self.scheduler = scheduler
        layout = QVBoxLayout(self)

        self.job_list = QListWidget()
        layout.addWidget(self.job_list)
        layout.addStretch(1)

        self.refresh_jobs()

    # --------------------------------------------------------------
    def add_active_job(self, job_id: int, name: str, config: Dict, worker) -> None:
        """Add a running job to the list and connect progress signals."""
        widget = JobItemWidget(job_id, name, config)
        item = QListWidgetItem()
        item.setSizeHint(widget.sizeHint())
        self.job_list.addItem(item)
        self.job_list.setItemWidget(item, widget)
        if self.scheduler is not None:
            widget.request_pause.connect(self.scheduler.pause_current)
            widget.request_resume.connect(self.scheduler.resume_current)
            widget.request_cancel.connect(lambda jid=job_id: self.scheduler.cancel_job(jid))
        widget.request_delete.connect(lambda jid=job_id: self._delete_job(jid))
        worker.signals.progress.connect(widget.update_progress)

    # --------------------------------------------------------------
    def refresh_jobs(self) -> None:
        """Reload job list from the database."""
        from GUI.models import JobDB

        self.job_list.clear()
        for job in JobDB.list_jobs(50):
            item = QListWidgetItem()
            widget = JobItemWidget(job["id"], job["name"], job["config"])
            widget.update_status(job["status"], job["started_at"], job["finished_at"])
            if self.scheduler is not None:
                widget.request_pause.connect(self.scheduler.pause_current)
                widget.request_resume.connect(self.scheduler.resume_current)
                widget.request_delete.connect(lambda jid=job["id"]: self._delete_job(jid))
                widget.request_cancel.connect(lambda jid=job["id"]: self.scheduler.cancel_job(jid))
            item.setSizeHint(widget.sizeHint())
            self.job_list.addItem(item)
            self.job_list.setItemWidget(item, widget)

    # --------------------------------------------------------------
    def _delete_job(self, job_id: int) -> None:
        from GUI.models import JobDB

        JobDB.delete_job(job_id)
        self.refresh_jobs()


