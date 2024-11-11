import logging
from functools import partial
from typing import List, Dict, Optional

from PyQt5.QtCore import QRectF, pyqtSignal, Qt, QEvent
from PyQt5.QtGui import QPen, QPainter, QPixmap, QFont
from PyQt5.QtWidgets import (
    QWidget, QComboBox, QPushButton, QVBoxLayout, QLabel,
    QProgressBar, QSpinBox, QSlider, QGraphicsView,
    QGraphicsScene, QMenu, QGridLayout, QFileDialog,
    QGroupBox, QSizePolicy, QSplitter, QGraphicsTextItem, QGraphicsObject, QAction, QGraphicsItem, QApplication,
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

        # Calculate font size based on screen DPI
        self.dpi_scaling = QApplication.primaryScreen().logicalDotsPerInch() / 96.0  # 96 DPI is standard
        self.base_font_size = 12  # Base font size that scales dynamically

        # Create the text item as a child
        self.text_item = QGraphicsTextItem(self)
        self.text_item.setDefaultTextColor(Qt.black)
        self.update_text_label()

    def set_crop_class(self, class_id: int):
        """
        Updates the class_id of the annotation and emits a signal.
        """
        self.annotation.class_id = class_id
        self.class_id = class_id
        self.class_label_changed.emit(self.annotation.to_dict(), self.class_id)
        self.update_text_label()
        self.update()

    def update_text_label(self):
        """
        Updates the text label with the current class name and scales the font size based on DPI, independent of zoom.
        """
        if self.text_item is not None:
            # Calculate dynamic font size
            font_size = int(self.base_font_size * self.dpi_scaling)
            font = QFont("Arial", font_size)
            self.text_item.setFont(font)

            # Set label text
            label_text = CLASS_COMPONENTS.get(self.class_id, "Unlabelled") if self.class_id != -1 else "Unlabelled"
            self.text_item.setPlainText(label_text)

            # Position the text above the image
            self.text_item.setPos(0, -20)  # Adjust as needed

    def boundingRect(self):
        return QRectF(0, 0, self.pixmap.width(), self.pixmap.height())

    def paint(self, painter: QPainter, option, widget):
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(0, 0, self.pixmap)

        # Draw border
        pen = QPen(Qt.black if not self.hovered else Qt.blue)
        pen.setWidth(2 if self.selected or self.hovered else 1)
        painter.setPen(pen)
        painter.drawRect(self.boundingRect())

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

        for class_id, class_name in CLASS_COMPONENTS.items():
            action = QAction(class_name, self.menu)
            action.triggered.connect(partial(self.set_crop_class, class_id))
            self.menu.addAction(action)

        unlabel_action = QAction("Unlabel", self.menu)
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
    class_selected_for_all = pyqtSignal(object)  # Emits the class_id for all visible crops
    crop_label_changed = pyqtSignal(dict, int)  # Emits (annotation_dict, class_id)
    save_project_state_requested = pyqtSignal()
    export_annotations_requested = pyqtSignal()
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

    def init_ui(self):
        """
        Initializes the user interface components.
        """
        # Create a QSplitter to divide control panel and crop view
        self.splitter = QSplitter(Qt.Horizontal)

        # Left Panel (Control Panel)
        control_panel = QWidget()
        control_panel_layout = QVBoxLayout(control_panel)

        # Clustering Controls Group
        clustering_group = QGroupBox("Clustering Controls")
        clustering_layout = QVBoxLayout()
        self.clustering_button = QPushButton("Start Clustering")
        self.clustering_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clustering_button.clicked.connect(self.request_clustering.emit)
        self.clustering_button.setFocusPolicy(Qt.NoFocus)  # Prevent button from stealing focus
        clustering_layout.addWidget(self.clustering_button)

        # Clustering Progress Bar
        self.clustering_progress_bar = QProgressBar()
        self.clustering_progress_bar.setValue(0)
        self.clustering_progress_bar.setVisible(False)  # Initially hidden
        clustering_layout.addWidget(self.clustering_progress_bar)
        clustering_group.setLayout(clustering_layout)
        control_panel_layout.addWidget(clustering_group)

        # Cluster Selection Group
        cluster_selection_group = QGroupBox("Cluster Selection")
        cluster_selection_layout = QVBoxLayout()
        # Checkbox for Auto-Next Cluster
        self.auto_next_checkbox = QCheckBox("Auto Next Cluster")
        self.auto_next_checkbox.setChecked(self.auto_next_cluster)  # Set initial state
        self.auto_next_checkbox.stateChanged.connect(self.on_auto_next_toggle)
        self.auto_next_checkbox.setFocusPolicy(Qt.NoFocus)  # Prevent checkbox from stealing focus
        cluster_selection_layout.addWidget(self.auto_next_checkbox)

        cluster_selection_layout.addWidget(QLabel("Select Cluster:"))

        # Create a horizontal layout for ComboBox and buttons
        cluster_selection_controls_layout = QHBoxLayout()

        self.prev_cluster_button = QPushButton("Previous")
        self.prev_cluster_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.prev_cluster_button.clicked.connect(self.on_prev_cluster)
        self.prev_cluster_button.setEnabled(False)  # Initially disabled
        self.prev_cluster_button.setFocusPolicy(Qt.NoFocus)  # Prevent button from stealing focus

        self.cluster_combo = QComboBox()
        self.cluster_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cluster_combo.currentIndexChanged.connect(self.on_cluster_selected)
        self.cluster_combo.setFocusPolicy(Qt.NoFocus)  # Prevent combobox from stealing focus

        self.next_cluster_button = QPushButton("Next")
        self.next_cluster_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.next_cluster_button.clicked.connect(self.on_next_cluster)
        self.next_cluster_button.setEnabled(False)  # Initially disabled
        self.next_cluster_button.setFocusPolicy(Qt.NoFocus)  # Prevent button from stealing focus

        # Add widgets to the controls layout
        cluster_selection_controls_layout.addWidget(self.prev_cluster_button)
        cluster_selection_controls_layout.addWidget(self.cluster_combo)
        cluster_selection_controls_layout.addWidget(self.next_cluster_button)

        # Add the controls layout to the cluster selection layout
        cluster_selection_layout.addLayout(cluster_selection_controls_layout)

        cluster_selection_group.setLayout(cluster_selection_layout)
        control_panel_layout.addWidget(cluster_selection_group)

        # Sampling Parameters Group
        sampling_group = QGroupBox("Sampling Parameters")
        sampling_layout = QVBoxLayout()
        sampling_layout.addWidget(QLabel("Number of Crops per Cluster:"))
        self.crops_spinbox = QSpinBox()
        self.crops_spinbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.crops_spinbox.setRange(1, 1000)
        self.crops_spinbox.setValue(100)
        self.crops_spinbox.valueChanged.connect(self.on_crops_changed)
        self.crops_spinbox.setFocusPolicy(Qt.NoFocus)  # Prevent spinbox from stealing focus
        sampling_layout.addWidget(self.crops_spinbox)
        sampling_group.setLayout(sampling_layout)
        control_panel_layout.addWidget(sampling_group)

        # Zoom Slider Group
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QVBoxLayout()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.zoom_slider.setMinimum(0)
        self.zoom_slider.setMaximum(10)
        self.zoom_slider.setValue(self.zoom_level)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        self.zoom_slider.setFocusPolicy(Qt.NoFocus)  # Prevent slider from stealing focus
        zoom_layout.addWidget(self.zoom_slider)
        zoom_group.setLayout(zoom_layout)
        control_panel_layout.addWidget(zoom_group)

        # Class Selection Buttons with Key Hints
        class_buttons_group = QGroupBox("Class Labels")
        class_buttons_layout = QGridLayout()
        row, col = 0, 0

        # Button for Unlabel with '-' key hint
        unlabel_button = QPushButton("Unlabel (-)")
        unlabel_button.clicked.connect(partial(self.on_class_button_clicked, None))
        unlabel_button.setFocusPolicy(Qt.NoFocus)  # Prevent button from stealing focus
        class_buttons_layout.addWidget(unlabel_button, row, col)
        col += 1

        # Add buttons for each class with number key hints
        for class_id, class_name in CLASS_COMPONENTS.items():
            button_text = f"{class_name} ({class_id})"
            button = QPushButton(button_text)
            button.clicked.connect(partial(self.on_class_button_clicked, class_id))
            button.setFocusPolicy(Qt.NoFocus)  # Prevent button from stealing focus
            class_buttons_layout.addWidget(button, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1
        class_buttons_group.setLayout(class_buttons_layout)

        # Add class buttons to control panel layout
        control_panel_layout.addWidget(class_buttons_group)

        # Add a spacer to ensure dynamic resizing
        control_panel_layout.addStretch()

        # Save and Load Buttons
        self.load_project_button = QPushButton("Load Project")
        self.load_project_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.load_project_button.clicked.connect(self.on_load_project_state)
        self.load_project_button.setFocusPolicy(Qt.NoFocus)  # Prevent button from stealing focus

        self.save_project_button = QPushButton("Save Project")
        self.save_project_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.save_project_button.clicked.connect(self.on_save_project_state)
        self.save_project_button.setFocusPolicy(Qt.NoFocus)  # Prevent button from stealing focus

        self.export_annotations_button = QPushButton("Export Annotations")
        self.export_annotations_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.export_annotations_button.clicked.connect(self.on_export_annotations)
        self.export_annotations_button.setFocusPolicy(Qt.NoFocus)  # Prevent button from stealing focus

        self.restore_autosave_button = QPushButton("Restore Autosave")
        self.restore_autosave_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.restore_autosave_button.clicked.connect(self.on_restore_autosave)
        self.restore_autosave_button.setFocusPolicy(Qt.NoFocus)  # Prevent button from stealing focus

        control_panel_layout.addWidget(self.load_project_button)
        control_panel_layout.addWidget(self.save_project_button)
        control_panel_layout.addWidget(self.export_annotations_button)
        control_panel_layout.addWidget(self.restore_autosave_button)

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
        self.graphics_view.setFocusPolicy(Qt.NoFocus)  # Ensure it can receive focus
        self.graphics_view.setMouseTracking(True)

        # Graphics Scene
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)

        # Crop Loading Progress Bar
        self.crop_loading_progress_bar = QProgressBar(self.graphics_view)
        self.crop_loading_progress_bar.setGeometry(
            (self.graphics_view.viewport().width() - 300) // 2,
            (self.graphics_view.viewport().height() - 25) // 2,
            300,
            25
        )  # Position and size within the view
        self.crop_loading_progress_bar.setAlignment(Qt.AlignCenter)
        self.crop_loading_progress_bar.setVisible(False)

        # Add the crop view to the splitter
        self.splitter.addWidget(self.graphics_view)

        self.splitter.setStretchFactor(0, 1)  # Control panel takes less space initially
        self.splitter.setStretchFactor(1, 3)  # Crop view takes up two-thirds initially
        self.splitter.splitterMoved.connect(self.arrange_crops)

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

        # Enter key to go to the next cluster
        if key == Qt.Key_Return or key == Qt.Key_Enter:
            self.on_next_cluster()
            return

        # Backspace key to go to the previous cluster
        if key == Qt.Key_Backspace:
            self.on_prev_cluster()
            return

        # Zoom with Ctrl + Mouse Wheel
        if key == Qt.Key_Plus or key == Qt.Key_Equal:
            self.zoom_level = min(self.zoom_level + 1, 10)
            self.zoom_slider.setValue(self.zoom_level)
            return
        elif key == Qt.Key_Minus:
            self.zoom_level = max(self.zoom_level - 1, 0)
            self.zoom_slider.setValue(self.zoom_level)
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
        viewport_rect = self.graphics_view.viewport().rect()
        x = (viewport_rect.width() - self.crop_loading_progress_bar.width()) // 2
        y = (viewport_rect.height() - self.crop_loading_progress_bar.height()) // 2
        self.crop_loading_progress_bar.move(x, y)
        self.crop_loading_progress_bar.setValue(0)
        self.crop_loading_progress_bar.setVisible(True)

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
        self.cluster_combo.blockSignals(False)  # Re-enable signals

        if selected_cluster_id is not None:
            index = self.cluster_combo.findData(selected_cluster_id)
            if index != -1:
                self.cluster_combo.setCurrentIndex(index)
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
        Assign the current cluster with the selected class, then move to the next cluster if auto-advance is enabled.
        """
        if class_id is None:
            logging.info("Unlabeling current cluster.")
        else:
            logging.info(f"Assigning class {class_id} to the current cluster.")

        # Label all visible crops
        self.label_all_visible_crops(class_id)

        # Emit the signal to inform the controller
        self.class_selected_for_all.emit(class_id)

        # Move to the next cluster if auto-advance is enabled
        if self.auto_next_cluster:
            self.on_next_cluster()

        self.setFocus()  # Ensure the main widget regains focus

    def label_all_visible_crops(self, class_id: Optional[int]):
        """
        Labels all the currently visible crops with the given class_id.
        If class_id is None, unlabels them.

        :param class_id: The class ID to assign, or None to unlabel.
        """
        for crop in self.selected_crops:
            # Update the crop's class_id
            annotation: Annotation = crop['annotation']
            annotation.class_id = class_id if class_id is not None else -1
            # Emit the signal
            self.crop_label_changed.emit(annotation.to_dict(), annotation.class_id)

        # Re-arrange the crops to reflect the updated labels
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
        """
        Arranges the sampled crops in a grid layout that adapts to the window size
        and supports zooming functionality. Images expand and contract to fill the space.
        """
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
            pixmap_item.setScale(scale)

            pixmap_item.setPos(x_offset, y_offset)
            self.scene.addItem(pixmap_item)
            arranged_crops_count += 1  # Increment counter

            x_offset += image_size + 20
            max_row_height = max(max_row_height, image_size)

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

    def on_zoom_changed(self, value: int):
        """
        Handles the zoom level change from the slider.

        :param value: The new zoom level.
        """
        self.zoom_level = max(-10, min(value, 10))  # Limit zoom_level between -10 and 10
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
            viewport_rect = self.graphics_view.viewport().rect()
            x = (viewport_rect.width() - self.crop_loading_progress_bar.width()) // 2
            y = (viewport_rect.height() - self.crop_loading_progress_bar.height()) // 2
            self.crop_loading_progress_bar.move(x, y)

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
