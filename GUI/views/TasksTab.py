from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QWidget, QVBoxLayout

from .TaskProgressWidget import MCBankerProgressWidget


class TasksTab(QWidget):
    """Tab displaying background task progress and queued jobs."""

    def __init__(self, scheduler=None, parent=None):
        super().__init__(parent)
        self.scheduler = scheduler
        layout = QVBoxLayout(self)

        self.job_list = QListWidget()
        layout.addWidget(self.job_list)

        self.mc_widget = MCBankerProgressWidget()
        layout.addWidget(self.mc_widget)
        layout.addStretch(1)

        if self.scheduler is not None:
            self.scheduler.job_started.connect(self.refresh_jobs)
            self.scheduler.job_finished.connect(self.refresh_jobs)

        self.refresh_jobs()

    # --------------------------------------------------------------
    def refresh_jobs(self):
        """Reload job list from the database."""
        from GUI.models import JobDB

        self.job_list.clear()
        for job in JobDB.list_jobs(20):
            item = QListWidgetItem(f"{job['id']}: {job['name']} [{job['status']}]")
            self.job_list.addItem(item)

