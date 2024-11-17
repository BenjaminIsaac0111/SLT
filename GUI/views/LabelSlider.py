from PyQt5.QtCore import pyqtSignal, Qt, QSize
from PyQt5.QtWidgets import QWidget, QSlider, QSizePolicy, QVBoxLayout


class LabeledSlider(QWidget):
    valueChanged = pyqtSignal(int)

    def __init__(self, minimum=0, maximum=10, interval=1, orientation=Qt.Horizontal, parent=None):
        super().__init__(parent)
        self.slider = QSlider(orientation)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setTickInterval(interval)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.slider.valueChanged.connect(self.valueChanged.emit)

        # Create the main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.slider)

        self.labels = []  # Store label references for later adjustment
        self.minimum = minimum
        self.maximum = maximum
        self.interval = interval

    def resizeEvent(self, event):
        """
        Recalculate label positions when the widget is resized.
        """
        super().resizeEvent(event)

    def sizeHint(self):
        """
        Provides a recommended size for the widget.
        """
        slider_size = self.slider.sizeHint()
        label_height = max((label.sizeHint().height() for label in self.labels), default=0)
        total_height = slider_size.height() + label_height + 5  # Add spacing
        return QSize(slider_size.width(), total_height)

    # Expose QSlider methods
    def setValue(self, value):
        self.slider.setValue(value)

    def value(self):
        return self.slider.value()

    def setMinimum(self, value):
        self.minimum = value
        self.slider.setMinimum(value)

    def setMaximum(self, value):
        self.maximum = value
        self.slider.setMaximum(value)

    def setTickInterval(self, value):
        self.interval = value
        self.slider.setTickInterval(value)

    def setOrientation(self, orientation):
        self.slider.setOrientation(orientation)
