import logging
from functools import partial
from typing import List, Dict, Optional

from PyQt5.QtCore import QRectF, pyqtSignal, Qt, QEvent, QSize, pyqtSlot
from PyQt5.QtGui import QPen, QPainter, QPixmap, QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QWidget, QComboBox, QPushButton, QVBoxLayout, QLabel,
    QProgressBar, QSpinBox, QSlider, QGraphicsView,
    QGraphicsScene, QMenu, QGridLayout, QFileDialog,
    QGroupBox, QSizePolicy, QSplitter, QGraphicsObject, QAction, QGraphicsItem, QApplication,
    QHBoxLayout, QCheckBox
)

from GUI.models.Annotation import Annotation

# Define class components globally
CLASS_COMPONENTS = {
    0: 'Non-Informative',
    1: 'Tumour',
    2: 'Stroma',
    3: 'Necrosis',
    4: 'Vessel',
    5: 'Inflammation',
    6: 'Tumour-Lumen',
    7: 'Mucin',
    8: 'Muscle'
}


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


class ClickablePixmapItem(QGraphicsObject):
    """
    A QGraphicsObject that displays a QPixmap and emits signals when interacted with.
    It holds a reference to an Annotation instance.
    """
    class_label_changed = pyqtSignal(dict, int)  # Emits (annotation_dict, class_id)

    def __init__(self, annotation: Annotation, pixmap: QPixmap, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation = annotation
        self.pixmap = pixmap
        self.class_id = annotation.class_id
        self.setAcceptHoverEvents(True)
        self.hovered = False
        self.selected = False
        self.scale_factor = 1.0  # Initialize scale factor

        # Calculate font size based on screen DPI
        screen = QApplication.primaryScreen()
        logical_dpi = screen.logicalDotsPerInch()
        standard_dpi = 96.0  # Standard Windows DPI
        self.dpi_scaling = logical_dpi / standard_dpi
        self.base_font_size = 12  # Base font size

        # Set up font and metrics
        font_size = int(self.base_font_size * self.dpi_scaling)
        self.font = QFont("Arial", font_size)
        self.font_metrics = QFontMetrics(self.font)
        self.label_height = self.font_metrics.height()

    def setScaleFactor(self, scale):
        """
        Sets the scale factor for the image.
        """
        self.scale_factor = scale
        self.update()  # Trigger a repaint

    def set_crop_class(self, class_id: int):
        self.annotation.class_id = class_id
        self.class_id = class_id
        self.class_label_changed.emit(self.annotation.to_dict(), self.class_id)
        self.update()  # Redraw the item to reflect the new label

    def boundingRect(self):
        """
        Adjusts the bounding rectangle to include the scaled image and the label above it.
        """
        pixmap_width = self.pixmap.width() * self.scale_factor
        pixmap_height = self.pixmap.height() * self.scale_factor
        return QRectF(0, -self.label_height, pixmap_width, pixmap_height + self.label_height)

    def paint(self, painter: QPainter, option, widget):
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Draw the scaled pixmap
        painter.save()
        painter.scale(self.scale_factor, self.scale_factor)
        painter.drawPixmap(0, 0, self.pixmap)
        painter.restore()

        # Draw border around the scaled image
        pixmap_width = self.pixmap.width() * self.scale_factor
        pixmap_height = self.pixmap.height() * self.scale_factor
        pen = QPen(Qt.black if not self.hovered else Qt.blue)
        pen.setWidth(2 if self.selected or self.hovered else 1)
        painter.setPen(pen)
        painter.drawRect(QRectF(0, 0, pixmap_width, pixmap_height))

        # Draw label text directly above the image, aligned with its top edge
        painter.save()
        # Position the label above the image in the item's local coordinates
        label_x = 0  # Align text with the left edge of the image
        label_y = -self.label_height + self.font_metrics.ascent()  # Offset to position text above image
        painter.translate(label_x, label_y)
        painter.setFont(self.font)
        painter.setPen(Qt.black)

        # Display "Unsure" for class_id -2, "Unlabelled" for -1, or the regular class name
        if self.class_id == -2:
            label_text = "Unsure"
        elif self.class_id == -1:
            label_text = "Unlabelled"
        else:
            label_text = CLASS_COMPONENTS.get(self.class_id, "Unlabelled")

        painter.drawText(0, 0, label_text)
        painter.restore()

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selected = not self.selected  # Toggle selection
            self.update()

    def contextMenuEvent(self, event):
        self.scene().context_menu_open = True  # Set the flag in the scene
        self.menu = QMenu()

        # Add class actions
        for class_id, class_name in CLASS_COMPONENTS.items():
            action = QAction(class_name, self.menu)
            action.triggered.connect(partial(self.set_crop_class, class_id))
            self.menu.addAction(action)

        # Add a separator to distinguish the special actions
        self.menu.addSeparator()

        # Add "Unsure" option with class_id -2
        unsure_action = QAction("Unsure (?)", self.menu)
        unsure_action.triggered.connect(partial(self.set_crop_class, -2))
        self.menu.addAction(unsure_action)

        # Add "Unlabel" option with class_id -1
        unlabel_action = QAction("Unlabel (-)", self.menu)
        unlabel_action.triggered.connect(partial(self.set_crop_class, -1))
        self.menu.addAction(unlabel_action)

        self.menu.exec_(event.screenPos())
        self.scene().context_menu_open = False  # Unset the flag


class ClusteredCropsView(QWidget):
    """
    ClusteredCropsView is responsible for displaying clustered crops in a grid layout,
    allowing users to interact with each crop (e.g., label them).
    """

    # Define signals to communicate with the controller
    request_clustering = pyqtSignal()
    sample_cluster = pyqtSignal(int)
    sampling_parameters_changed = pyqtSignal(int, int)  # Emits (cluster_id, crops_per_cluster)
    bulk_label_changed = pyqtSignal(int)  # Emits the class_id for all visible crops
    crop_label_changed = pyqtSignal(dict, int)  # Emits (annotation_dict, class_id)
    save_project_state_requested = pyqtSignal()
    export_annotations_requested = pyqtSignal()
    # Signals for Save and Save As actions
    save_project_requested = pyqtSignal()
    save_project_as_requested = pyqtSignal()

    load_project_state_requested = pyqtSignal(str)  # Emits the filename of the project state to load
    restore_autosave_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.zoom_level = 5  # Initial zoom level
        self.context_menu_open = False  # Initialize the flag
        self.selected_crops: List[dict] = []  # Store sampled crops for dynamic rearrangement
        self.auto_next_cluster = False
        self.init_ui()

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.graphics_view.viewport().installEventFilter(self)

    # noinspection PyAttributeOutsideInit
    def init_ui(self):
        """
        Initializes the user interface components with control panels and embedded keyboard shortcuts.
        """
        # Create a QSplitter to divide control panel and crop view
        self.splitter = QSplitter(Qt.Horizontal)

        # Left Panel (Control Panel)
        control_panel = QWidget()
        control_panel_layout = QVBoxLayout(control_panel)  # Main layout for control panel

        ##### Clustering Controls Group #####
        clustering_group = QGroupBox("Clustering")
        clustering_layout = QVBoxLayout()

        self.clustering_button = QPushButton("Start Clustering")
        self.clustering_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clustering_button.clicked.connect(self.request_clustering.emit)
        self.clustering_button.setFocusPolicy(Qt.NoFocus)
        clustering_layout.addWidget(self.clustering_button)

        # Clustering Progress Bar
        self.clustering_progress_bar = QProgressBar()
        self.clustering_progress_bar.setValue(0)
        self.clustering_progress_bar.setVisible(False)
        clustering_layout.addWidget(self.clustering_progress_bar)

        clustering_group.setLayout(clustering_layout)
        control_panel_layout.addWidget(clustering_group)

        ##### Cluster Navigation Group with Hints #####
        cluster_navigation_group = QGroupBox("Cluster Navigation")
        cluster_navigation_layout = QVBoxLayout()

        # Auto-Next Cluster Checkbox
        self.auto_next_checkbox = QCheckBox("Auto Next Cluster")
        self.auto_next_checkbox.setChecked(self.auto_next_cluster)
        self.auto_next_checkbox.stateChanged.connect(self.on_auto_next_toggle)
        self.auto_next_checkbox.setFocusPolicy(Qt.NoFocus)
        cluster_navigation_layout.addWidget(self.auto_next_checkbox)

        # Cluster Selection Controls with Hints
        cluster_selection_layout = QHBoxLayout()
        self.prev_cluster_button = QPushButton("Previous")
        self.prev_cluster_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.prev_cluster_button.clicked.connect(self.on_prev_cluster)
        self.prev_cluster_button.setEnabled(False)
        self.prev_cluster_button.setFocusPolicy(Qt.NoFocus)

        self.cluster_combo = QComboBox()
        self.cluster_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cluster_combo.currentIndexChanged.connect(self.on_cluster_selected)
        self.cluster_combo.setFocusPolicy(Qt.NoFocus)

        self.next_cluster_button = QPushButton("Next")
        self.next_cluster_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.next_cluster_button.clicked.connect(self.on_next_cluster)
        self.next_cluster_button.setEnabled(False)
        self.next_cluster_button.setFocusPolicy(Qt.NoFocus)

        cluster_selection_layout.addWidget(self.prev_cluster_button)
        cluster_selection_layout.addWidget(self.cluster_combo)
        cluster_selection_layout.addWidget(self.next_cluster_button)

        cluster_navigation_layout.addLayout(cluster_selection_layout)

        # Keyboard hints for navigation
        navigation_hint = QLabel("Shortcuts: Enter - Next Cluster, Backspace - Previous Cluster")
        cluster_navigation_layout.addWidget(navigation_hint)
        cluster_navigation_group.setLayout(cluster_navigation_layout)
        control_panel_layout.addWidget(cluster_navigation_group)

        ##### Sampling Parameters Group #####
        sampling_group = QGroupBox("Sampling Parameters")
        sampling_layout = QVBoxLayout()
        sampling_layout.addWidget(QLabel("Number of Crops per Cluster:"))

        self.crops_spinbox = QSpinBox()
        self.crops_spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.crops_spinbox.setRange(1, 1000)
        self.crops_spinbox.setValue(100)
        self.crops_spinbox.valueChanged.connect(self.on_crops_changed)
        self.crops_spinbox.setFocusPolicy(Qt.NoFocus)
        sampling_layout.addWidget(self.crops_spinbox)

        sampling_group.setLayout(sampling_layout)
        control_panel_layout.addWidget(sampling_group)

        ##### Zoom Controls Group #####
        # Zoom Controls Group
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QVBoxLayout()

        # Use the updated LabeledSlider
        self.zoom_slider = LabeledSlider(minimum=0, maximum=10, interval=1)
        self.zoom_slider.setValue(self.zoom_level)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)

        zoom_hint = QLabel("Use Ctrl + Mouse Wheel to zoom in/out")
        zoom_layout.addWidget(zoom_hint)

        zoom_group.setLayout(zoom_layout)
        control_panel_layout.addWidget(zoom_group)

        ##### Class Labels Group with Keyboard Hints #####
        class_labels_group = QGroupBox("Class Labels")
        class_labels_layout = QGridLayout()
        row, col = 0, 0

        # Separate Layout for Special Actions (Unlabel and Unsure)
        special_actions_layout = QHBoxLayout()

        # Unlabel Button with Hint
        unlabel_button = QPushButton("Unlabel (-)")
        unlabel_button.clicked.connect(partial(self.on_class_button_clicked, -1))
        unlabel_button.setFocusPolicy(Qt.NoFocus)
        unlabel_button.setStyleSheet("background-color: #f08080;")  # Light Coral
        special_actions_layout.addWidget(unlabel_button)

        # Unsure Button with Hint
        unsure_button = QPushButton("Unsure (?)")
        unsure_button.clicked.connect(partial(self.on_class_button_clicked, -2))
        unsure_button.setFocusPolicy(Qt.NoFocus)
        unsure_button.setStyleSheet("background-color: #ffa500;")  # Orange
        special_actions_layout.addWidget(unsure_button)

        # Add the special actions layout spanning all columns
        class_labels_layout.addLayout(special_actions_layout, row, 0, 1, 3)
        row += 1

        # Class Buttons with Keyboard Hints
        for class_id, class_name in CLASS_COMPONENTS.items():
            button_text = f"{class_name} ({class_id})"
            button = QPushButton(button_text)
            button.clicked.connect(partial(self.on_class_button_clicked, class_id))
            button.setFocusPolicy(Qt.NoFocus)
            class_labels_layout.addWidget(button, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1

        # Keyboard shortcut hint for class labeling
        class_hint = QLabel("Shortcuts: 1-9 for Class Labels, Minus (-) to Unlabel, ? to Unsure")
        class_labels_layout.addWidget(class_hint, row + 1, 0, 1, 3)  # Span across all columns
        class_labels_group.setLayout(class_labels_layout)
        control_panel_layout.addWidget(class_labels_group)

        ##### File Operations Group #####
        file_operations_group = QGroupBox("File Operations")
        file_operations_layout = QVBoxLayout()

        self.load_project_button = QPushButton("Load Project")
        self.load_project_button.clicked.connect(self.on_load_project_state)
        self.load_project_button.setFocusPolicy(Qt.NoFocus)
        file_operations_layout.addWidget(self.load_project_button)

        self.save_project_button = QPushButton("Save")
        self.save_project_button.clicked.connect(self.save_project_requested.emit)
        self.save_project_button.setFocusPolicy(Qt.NoFocus)
        file_operations_layout.addWidget(self.save_project_button)

        self.save_project_as_button = QPushButton("Save As...")
        self.save_project_as_button.clicked.connect(self.save_project_as_requested.emit)
        self.save_project_as_button.setFocusPolicy(Qt.NoFocus)
        file_operations_layout.addWidget(self.save_project_as_button)

        self.export_annotations_button = QPushButton("Export Annotations")
        self.export_annotations_button.clicked.connect(self.on_export_annotations)
        self.export_annotations_button.setFocusPolicy(Qt.NoFocus)
        file_operations_layout.addWidget(self.export_annotations_button)

        self.restore_autosave_button = QPushButton("Restore Autosave")
        self.restore_autosave_button.clicked.connect(self.on_restore_autosave)
        self.restore_autosave_button.setFocusPolicy(Qt.NoFocus)
        file_operations_layout.addWidget(self.restore_autosave_button)

        file_operations_group.setLayout(file_operations_layout)
        control_panel_layout.addWidget(file_operations_group)

        # Add a spacer to push everything to the top
        control_panel_layout.addStretch()

        # Labeling Statistics Group
        statistics_group = QGroupBox("Labeling Statistics")
        statistics_layout = QVBoxLayout()

        # Total Annotations
        self.total_annotations_label = QLabel("Total Annotations: 0")
        statistics_layout.addWidget(self.total_annotations_label)

        # Total Labeled Annotations
        self.total_labeled_label = QLabel("Total Labeled Annotations: 0")
        statistics_layout.addWidget(self.total_labeled_label)

        # Class Counts
        self.class_counts_labels = {}
        for class_id in sorted(CLASS_COMPONENTS.keys()):
            class_name = CLASS_COMPONENTS[class_id]
            label = QLabel(f"{class_name}: 0")
            statistics_layout.addWidget(label)
            self.class_counts_labels[class_id] = label

        # Unlabeled and Unsure counts
        self.unlabeled_label = QLabel("Unlabeled (-1): 0")
        statistics_layout.addWidget(self.unlabeled_label)
        self.unsure_label = QLabel("Unsure (-2): 0")
        statistics_layout.addWidget(self.unsure_label)

        statistics_group.setLayout(statistics_layout)
        control_panel_layout.addWidget(statistics_group)

        # Add a spacer to push everything to the top
        control_panel_layout.addStretch()

        # Add the control panel to the splitter
        self.splitter.addWidget(control_panel)

        # Right Panel (Crop View)
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.graphics_view.setFocusPolicy(Qt.NoFocus)
        self.graphics_view.setMouseTracking(True)

        # Graphics Scene
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)

        # Crop Loading Progress Bar
        # Crop Loading Progress Bar
        self.crop_loading_progress_bar = QProgressBar(self.graphics_view.viewport())
        self.crop_loading_progress_bar.setFixedSize(300, 25)
        self.crop_loading_progress_bar.setAlignment(Qt.AlignCenter)
        self.crop_loading_progress_bar.setVisible(False)

        # Add the crop view to the splitter
        self.splitter.addWidget(self.graphics_view)

        self.splitter.setStretchFactor(0, 1)  # Control panel
        self.splitter.setStretchFactor(1, 3)  # Crop view
        self.splitter.splitterMoved.connect(self.arrange_crops)
        self.splitter.splitterMoved.connect(self.on_splitter_moved)

        # Set the layout for the main window
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)
        self.setWindowTitle("Clustered Crops Viewer")
        self.resize(1200, 800)

    def keyPressEvent(self, event):
        key = event.key()

        # Number keys for class selection (0 to 8)
        if Qt.Key_0 <= key <= Qt.Key_8:
            class_id = key - Qt.Key_0
            if class_id in CLASS_COMPONENTS:
                self.on_class_button_clicked(class_id)
                return  # Event handled

        # Minus key to unlabel
        if key == Qt.Key_Minus or key == Qt.Key_Underscore:
            self.on_class_button_clicked(None)
            return  # Event handled

        # "?" key for marking as Unsure (-2)
        if key == Qt.Key_Slash and event.modifiers() == Qt.ShiftModifier:
            self.on_class_button_clicked(-2)
            return  # Event handled

        # Enter key to go to the next cluster
        if key == Qt.Key_Return or key == Qt.Key_Enter:
            self.on_next_cluster()
            return

        # Backspace key to go to the previous cluster
        if key == Qt.Key_Backspace:
            self.on_prev_cluster()
            return

        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        """
        Intercepts wheel events on the graphics view's viewport to handle zooming without scrolling.
        """
        if obj == self.graphics_view.viewport() and event.type() == QEvent.Wheel:
            if event.modifiers() == Qt.ControlModifier:
                # Handle zooming
                delta = event.angleDelta().y() / 120  # 120 is the standard step
                self.zoom_level += delta
                self.zoom_level = max(0, min(self.zoom_level, 10))  # Limit zoom_level between 0 and 10
                self.zoom_slider.setValue(self.zoom_level)
                self.arrange_crops()  # Update the display to apply the zoom
                event.accept()  # Accept the event to prevent further processing
                return True  # Event handled
            else:
                # Allow scrolling
                return False  # Do not filter the event
        return super().eventFilter(obj, event)

    def show_clustering_progress_bar(self):
        self.clustering_progress_bar.setValue(0)
        self.clustering_progress_bar.setVisible(True)

    def update_clustering_progress_bar(self, progress: int):
        self.clustering_progress_bar.setValue(progress)

    def hide_clustering_progress_bar(self):
        self.clustering_progress_bar.setVisible(False)

    def show_crop_loading_progress_bar(self):
        """
        Shows the progress bar in the center of the graphics view.
        """
        self.crop_loading_progress_bar.setValue(0)
        self.crop_loading_progress_bar.setVisible(True)
        QApplication.processEvents()  # Ensure UI is updated
        self.center_crop_loading_progress_bar()

    def center_crop_loading_progress_bar(self):
        """
        Centers the crop loading progress bar within the viewport of the graphics view.
        """
        viewport_rect = self.graphics_view.viewport().rect()
        x = (viewport_rect.width() - self.crop_loading_progress_bar.width()) // 2
        y = (viewport_rect.height() - self.crop_loading_progress_bar.height()) // 2
        self.crop_loading_progress_bar.move(x, y)

    def update_crop_loading_progress_bar(self, progress: int):
        """
        Updates the progress bar value.
        """
        if not self.crop_loading_progress_bar.isVisible():
            self.crop_loading_progress_bar.setVisible(True)
        self.crop_loading_progress_bar.setValue(progress)
        logging.debug(f"Progress updated to: {progress}%")

    def hide_crop_loading_progress_bar(self):
        """
        Hides the progress bar.
        """
        self.crop_loading_progress_bar.setVisible(False)

    def showEvent(self, event):
        super().showEvent(event)
        # Calculate sizes based on the actual window width
        total_width = self.width()
        control_panel_width = total_width // 3  # One-third of the total width
        graphics_view_width = total_width - control_panel_width
        self.splitter.setSizes([control_panel_width, graphics_view_width])

    def populate_cluster_selection(self, cluster_info: Dict[int, dict], selected_cluster_id: Optional[int] = None):
        """
        Populates the cluster selection ComboBox with cluster IDs and info.
        The clusters are displayed in the order they appear in the cluster_info.
        """
        self.cluster_combo.blockSignals(True)  # Block signals to prevent unwanted emissions
        self.cluster_combo.clear()
        for cluster_id in cluster_info:
            info = cluster_info[cluster_id]
            display_text = f"Cluster {cluster_id} - {info['num_annotations']} annotations"
            self.cluster_combo.addItem(display_text, cluster_id)

        if selected_cluster_id is not None:
            index = self.cluster_combo.findData(selected_cluster_id)
            if index != -1:
                self.cluster_combo.setCurrentIndex(index)
        self.cluster_combo.blockSignals(False)  # Re-enable signals

        logging.debug(f"Populated cluster selection with IDs: {list(cluster_info.keys())}")

        # Update the enabled state of the next and previous buttons
        has_clusters = self.cluster_combo.count() > 0
        current_index = self.cluster_combo.currentIndex()
        self.prev_cluster_button.setEnabled(has_clusters and current_index > 0)
        self.next_cluster_button.setEnabled(has_clusters and current_index < self.cluster_combo.count() - 1)

    def on_prev_cluster(self):
        """
        Selects the previous cluster in the ComboBox.
        """
        current_index = self.cluster_combo.currentIndex()
        if current_index > 0:
            self.cluster_combo.setCurrentIndex(current_index - 1)
        self.setFocus()  # Ensure the main widget regains focus

    def on_next_cluster(self):
        """
        Selects the next cluster in the ComboBox.
        """
        current_index = self.cluster_combo.currentIndex()
        if current_index < self.cluster_combo.count() - 1:
            self.cluster_combo.setCurrentIndex(current_index + 1)
        self.setFocus()  # Ensure the main widget regains focus

    def on_cluster_selected(self, index: int):
        """
        Emits a signal to sample crops from the selected cluster.

        :param index: The index of the selected cluster in the ComboBox.
        """
        cluster_id = self.cluster_combo.itemData(index)
        if cluster_id is not None:
            logging.debug(f"Cluster selected: {cluster_id}")
            self.sample_cluster.emit(cluster_id)

        # Update the enabled state of the next and previous buttons
        self.prev_cluster_button.setEnabled(index > 0)
        self.next_cluster_button.setEnabled(index < self.cluster_combo.count() - 1)

    def on_splitter_moved(self, pos, index):
        """
        Slot called when the splitter is moved.
        """
        self.arrange_crops()
        if self.crop_loading_progress_bar.isVisible():
            self.center_crop_loading_progress_bar()

    def on_auto_next_toggle(self, state):
        self.auto_next_cluster = bool(state)

    def on_crops_changed(self, value: int):
        """
        Emits a signal when the number of crops per cluster changes.

        :param value: The new number of crops per cluster.
        """
        current_cluster_id = self.get_selected_cluster_id()
        if current_cluster_id is not None:
            logging.debug(f"Number of crops per cluster changed to: {value}")
            self.sampling_parameters_changed.emit(current_cluster_id, value)

    def on_class_button_clicked(self, class_id: Optional[int]):
        """
        Assigns a class to all visible crops and emits a bulk change signal.
        """
        if class_id == -2:
            logging.info("Classifying current cluster as unsure.")
        elif class_id == -1:
            logging.info("Unlabeling current cluster.")
        else:
            logging.info(f"Assigning class {class_id} to the current cluster.")

        # Label all visible crops in bulk
        self.label_all_visible_crops(class_id)

        # Move to the next cluster if auto-advance is enabled
        if self.auto_next_cluster:
            self.on_next_cluster()

        self.setFocus()

    def label_all_visible_crops(self, class_id: Optional[int]):
        """
        Labels all the currently visible crops with the given class_id.
        If class_id is None, unlabels them.
        """
        for crop in self.selected_crops:
            # Update the crop's class_id without emitting individual signals
            annotation: Annotation = crop['annotation']
            annotation.class_id = class_id if class_id is not None else -1

        # Emit a bulk signal to indicate that all visible crops were updated
        self.bulk_label_changed.emit(class_id)

        # Refresh UI once after updating all crops
        self.arrange_crops()

    def get_selected_cluster_id(self) -> Optional[int]:
        """
        Returns the currently selected cluster ID.
        """
        index = self.cluster_combo.currentIndex()
        if index != -1:
            return int(self.cluster_combo.itemData(index))
        return None

    def get_cluster_id_list(self) -> List[int]:
        """
        Returns the list of cluster IDs in the order they appear in the dropdown.
        """
        cluster_id_list = []
        for index in range(self.cluster_combo.count()):
            cluster_id = int(self.cluster_combo.itemData(index))
            cluster_id_list.append(cluster_id)
        return cluster_id_list

    def display_sampled_crops(self, sampled_crops: List[dict]):
        """
        Displays the sampled crops in a grid layout within the graphics scene.

        :param sampled_crops: A list of dictionaries containing 'annotation', 'processed_crop', and 'coord_pos'.
        """
        self.selected_crops = sampled_crops
        self.arrange_crops()

    def update_progress(self, progress: int):
        """
        Updates the crop loading progress bar with the current progress value.

        :param progress: Progress percentage (0-100).
        """
        if not self.crop_loading_progress_bar.isVisible():
            self.crop_loading_progress_bar.setVisible(True)
        self.crop_loading_progress_bar.setValue(progress)
        logging.debug(f"Progress updated to: {progress}%")

    @pyqtSlot(dict)
    def update_labeling_statistics(self, statistics):
        total_annotations = statistics['total_annotations']
        total_labeled = statistics['total_labeled']
        class_counts = statistics['class_counts']

        self.total_annotations_label.setText(f"Total Annotations: {total_annotations}")
        self.total_labeled_label.setText(f"Total Labeled Annotations: {total_labeled}")

        # Update class counts
        for class_id, count in class_counts.items():
            if class_id in CLASS_COMPONENTS:
                label = self.class_counts_labels.get(class_id)
                if label:
                    class_name = CLASS_COMPONENTS[class_id]
                    label.setText(f"{class_name} ({class_id}): {count}")
            elif class_id == -1:
                self.unlabeled_label.setText(f"Unlabeled (-1): {count}")
            elif class_id == -2:
                self.unsure_label.setText(f"Unsure (-2): {count}")

    def reset_progress(self):
        """
        Resets and shows the progress bar.
        """
        self.crop_loading_progress_bar.setValue(0)
        self.crop_loading_progress_bar.setVisible(True)
        logging.debug("Progress bar reset.")

    def hide_progress_bar(self):
        """
        Hides the crop loading progress bar.
        """
        self.crop_loading_progress_bar.setVisible(False)
        logging.debug("Progress bar hidden.")

    def arrange_crops(self):
        if getattr(self.scene, 'context_menu_open', False):
            return  # Do not rearrange crops if the context menu is open
        self.scene.clear()

        if not self.selected_crops:
            logging.info("No sampled crops to display.")
            return

        # Get the viewport size
        viewport_width = self.graphics_view.viewport().width()
        base_size = 150  # Base size for images at zoom level 0
        scale_factor = pow(1.2, self.zoom_level)  # Exponential scale factor
        image_size = base_size * scale_factor

        num_columns = max(1, int(viewport_width // (image_size + 20)))  # Adjust spacing

        # Adjust image size to fill the width
        total_spacing = 20 * (num_columns + 1)
        if num_columns > 0:
            image_size = (viewport_width - total_spacing) / num_columns
        else:
            num_columns = 1
            image_size = viewport_width - total_spacing

        logging.debug(f"Calculated image size: {image_size}, Number of columns: {num_columns}")

        # Arrange images
        x_offset = 20
        y_offset = 20
        max_row_height = 0
        arranged_crops_count = 0  # Counter for arranged crops

        for idx, crop_data in enumerate(self.selected_crops):
            annotation: Annotation = crop_data['annotation']
            q_pixmap: QPixmap = crop_data['processed_crop']

            if q_pixmap.isNull():
                logging.warning(f"Invalid QPixmap for image index {annotation.image_index}. Skipping.")
                continue

            # Create pixmap item with reference to the Annotation instance
            pixmap_item = ClickablePixmapItem(annotation=annotation, pixmap=q_pixmap)
            pixmap_item.setFlag(QGraphicsItem.ItemIsSelectable, True)

            # Connect the signals
            pixmap_item.class_label_changed.connect(self.crop_label_changed.emit)

            # Position the crop image
            original_width = q_pixmap.width()
            if original_width == 0:
                logging.warning("Crop width is zero, skipping this image to prevent division by zero.")
                continue
            scale = image_size / original_width
            pixmap_item.setScaleFactor(scale)  # Use setScaleFactor instead of setScale

            # After setting the scale factor, get the bounding rect
            bounding_rect = pixmap_item.boundingRect()

            pixmap_item.setPos(x_offset, y_offset)
            self.scene.addItem(pixmap_item)
            arranged_crops_count += 1  # Increment counter

            x_offset += bounding_rect.width() + 20  # Use bounding_rect.width()
            max_row_height = max(max_row_height, bounding_rect.height())  # Use bounding_rect.height()

            if (idx + 1) % num_columns == 0:
                x_offset = 20
                y_offset += max_row_height + 20  # Adjust for both crop and label height
                max_row_height = 0

        # Assertion to ensure all crops are arranged
        expected_arranged = len(self.selected_crops)
        actual_arranged = arranged_crops_count
        assert actual_arranged == expected_arranged, (
            f"View arrangement mismatch: expected {expected_arranged} crops arranged, got {actual_arranged}"
        )
        logging.info(f"Arranged {actual_arranged} crops in the view.")

        # Adjust the scene rect
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        logging.debug("Scene rect set to include all items.")

        # Re-center the crop loading progress bar if it's visible
        if self.crop_loading_progress_bar.isVisible():
            self.center_crop_loading_progress_bar()

    def on_zoom_changed(self, value: int):
        """
        Handles the zoom level change from the slider.

        :param value: The new zoom level.
        """
        self.zoom_level = max(0, min(value, 10))  # Limit zoom_level between 0 and 10
        logging.debug(f"Zoom level changed to: {self.zoom_level}")
        self.arrange_crops()

    def resizeEvent(self, event):
        """
        Overrides the resize event to rearrange crops and reposition progress bars when the window size changes.
        """
        super().resizeEvent(event)
        logging.debug("Window resized. Rearranging crops.")
        self.arrange_crops()  # Ensure crops are arranged on window resize

        # Re-center the crop loading progress bar if it's visible
        if self.crop_loading_progress_bar.isVisible():
            self.center_crop_loading_progress_bar()

    def on_load_project_state(self):
        """
        Opens a file dialog to select a project state file to load.
        """
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Project State", "", "JSON Files (*.json);;All Files (*)",
            options=options
        )
        if filename:
            logging.debug(f"Selected project state file: {filename}")
            self.load_project_state_requested.emit(filename)
        else:
            logging.debug("No project state file selected.")

    def on_save_project_state(self):
        """
        Emits a signal when the 'Save Project' button is clicked.
        """
        self.save_project_state_requested.emit()

    def on_export_annotations(self):
        """
        Emits a signal when the 'Export Annotations' button is clicked.
        """
        self.export_annotations_requested.emit()

    def on_restore_autosave(self):
        """
        Emits a signal when the 'Restore Autosave' button is clicked.
        """
        self.restore_autosave_requested.emit()
