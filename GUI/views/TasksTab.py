from PyQt5.QtWidgets import QWidget, QVBoxLayout

from .TaskProgressWidget import MCBankerProgressWidget


class TasksTab(QWidget):
    """Tab displaying background task progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.mc_widget = MCBankerProgressWidget()
        layout.addWidget(self.mc_widget)
        layout.addStretch(1)

