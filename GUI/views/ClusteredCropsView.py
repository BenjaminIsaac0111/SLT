# GUI/views/ClusteredCropsView.py

import logging
from functools import partial
from typing import List, Dict, Optional

from PyQt5.QtCore import QRectF, pyqtSignal, Qt, QEvent
from PyQt5.QtGui import QPen, QPainter, QPixmap, QFont, QWheelEvent
from PyQt5.QtWidgets import QGraphicsObject, QAction, QGraphicsItem
from PyQt5.QtWidgets import (
    QWidget, QComboBox, QPushButton, QVBoxLayout, QLabel,
    QProgressBar, QSpinBox, QSlider, QGraphicsView,
    QGraphicsScene, QMenu, QGridLayout, QFileDialog,
    QGroupBox, QSizePolicy, QSplitter, QGraphicsTextItem
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

        # Create the text item as a child
        self.text_item = QGraphicsTextItem(self)
        self.text_item.setDefaultTextColor(Qt.black)
        self.text_item.setFont(QFont("Arial", 12))  # Adjust font if needed
        self.update_text_label()

    def set_crop_class(self, class_id: int):
        """
        Updates the class_id of the annotation and emits a signal.
        """
        self.annotation.class_id = class_id if class_id != -1 else -1
        self.class_id = self.annotation.class_id
        self.class_label_changed.emit(self.annotation.to_dict(), self.class_id)
        self.update_text_label()
        self.update()

    def update_text_label(self):
        """
        Updates the text label with the current class name.
        """
        if self.text_item is not None:
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
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event)

    def show_context_menu(self, event):
        menu = QMenu()

        # Add actions for each class using functools.partial
        for class_id, class_name in CLASS_COMPONENTS.items():
            action = QAction(class_name, self)
            action.triggered.connect(partial(self.set_crop_class, class_id))
            menu.addAction(action)

        # Add Unlabel Option
        unlabel_action = QAction("Unlabel", self)
        unlabel_action.triggered.connect(partial(self.set_crop_class, -1))
        menu.addAction(unlabel_action)

        # Display the menu
        menu.exec_(event.screenPos())


class ClusteredCropsView(QWidget):
    """
    ClusteredCropsView is responsible for displaying clustered crops in a grid layout,
    allowing users to interact with each crop (e.g., label them).
    """

    # Define signals to communicate with the controller
    request_clustering = pyqtSignal()
    sample_cluster = pyqtSignal(int)
    sampling_parameters_changed = pyqtSignal(int, int)  # Emits (cluster_id, crops_per_cluster)
    class_selected = pyqtSignal(int, int)  # Emits (cluster_id, class_id)
    class_selected_for_all = pyqtSignal(object)  # Emits the class_id for all visible crops
    crop_label_changed = pyqtSignal(dict, int)  # Emits (annotation_dict, class_id)
    save_project_state_requested = pyqtSignal()
    export_annotations_requested = pyqtSignal()
    load_project_state_requested = pyqtSignal(str)  # Emits the filename of the project state to load
    restore_autosave_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.zoom_level = 5  # Initial zoom level
        self.selected_crops: List[dict] = []  # Store sampled crops for dynamic rearrangement
        self.init_ui()

    def init_ui(self):
        """
        Initializes the user interface components.
        """
        # Create a QSplitter to divide control panel and crop view
        splitter = QSplitter(Qt.Horizontal)

        # Left Panel (Control Panel)
        control_panel = QWidget()
        control_panel_layout = QVBoxLayout(control_panel)

        # Clustering Controls Group
        clustering_group = QGroupBox("Clustering Controls")
        clustering_layout = QVBoxLayout()
        self.clustering_button = QPushButton("Start Clustering")
        self.clustering_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clustering_button.clicked.connect(self.request_clustering.emit)
        clustering_layout.addWidget(self.clustering_button)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Initially hidden
        clustering_layout.addWidget(self.progress_bar)
        clustering_group.setLayout(clustering_layout)
        control_panel_layout.addWidget(clustering_group)

        # Cluster Selection Group
        cluster_selection_group = QGroupBox("Cluster Selection")
        cluster_selection_layout = QVBoxLayout()
        cluster_selection_layout.addWidget(QLabel("Select Cluster:"))
        self.cluster_combo = QComboBox()
        self.cluster_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cluster_combo.currentIndexChanged.connect(self.on_cluster_selected)
        cluster_selection_layout.addWidget(self.cluster_combo)
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
        sampling_layout.addWidget(self.crops_spinbox)
        sampling_group.setLayout(sampling_layout)
        control_panel_layout.addWidget(sampling_group)

        # Zoom Slider Group
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QVBoxLayout()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.zoom_slider.setMinimum(-10)
        self.zoom_slider.setMaximum(10)
        self.zoom_slider.setValue(self.zoom_level)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_group.setLayout(zoom_layout)
        control_panel_layout.addWidget(zoom_group)

        # Class Selection Buttons Group
        class_buttons_group = QGroupBox("Class Labels")
        class_buttons_layout = QGridLayout()
        row = 0
        col = 0

        # Add Unlabel Button
        unlabel_button = QPushButton("Unlabel")
        unlabel_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        unlabel_button.clicked.connect(partial(self.on_class_button_clicked, None))
        class_buttons_layout.addWidget(unlabel_button, row, col)

        col += 1
        for class_id, class_name in CLASS_COMPONENTS.items():
            button = QPushButton(class_name)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(partial(self.on_class_button_clicked, class_id))
            class_buttons_layout.addWidget(button, row, col)
            col += 1
            if col >= 3:  # Adjust number of buttons per row
                col = 0
                row += 1
        class_buttons_group.setLayout(class_buttons_layout)

        control_panel_layout.addWidget(class_buttons_group)

        # Add a spacer to ensure dynamic resizing
        control_panel_layout.addStretch()

        # Save and Load Buttons
        self.load_project_button = QPushButton("Load Project")
        self.load_project_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.load_project_button.clicked.connect(self.load_project_state_requested.emit)

        self.save_project_button = QPushButton("Save Project")
        self.save_project_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.save_project_button.clicked.connect(self.save_project_state_requested.emit)

        self.export_annotations_button = QPushButton("Export Annotations")
        self.export_annotations_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.export_annotations_button.clicked.connect(self.export_annotations_requested.emit)

        self.restore_autosave_button = QPushButton("Restore Autosave")
        self.restore_autosave_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.restore_autosave_button.clicked.connect(self.restore_autosave_requested.emit)

        control_panel_layout.addWidget(self.load_project_button)
        control_panel_layout.addWidget(self.save_project_button)
        control_panel_layout.addWidget(self.export_annotations_button)
        control_panel_layout.addWidget(self.restore_autosave_button)

        control_panel_layout.addStretch()

        # Add the control panel to the splitter
        splitter.addWidget(control_panel)

        # Right Panel (Crop View)
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.graphics_view.viewport().installEventFilter(self)
        self.graphics_view.setMouseTracking(True)

        # Graphics Scene
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)

        # Add the crop view to the splitter
        splitter.addWidget(self.graphics_view)

        # Set the initial splitter ratio
        splitter.setStretchFactor(0, 1)  # Control panel takes less space initially
        splitter.setStretchFactor(1, 5)  # Crop view takes more space initially

        # Set the layout for the main window
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.setWindowTitle("Clustered Crops Viewer")
        self.resize(1200, 800)  # Ensure a large initial window size

    def populate_cluster_selection(self, cluster_info: Dict[int, dict], selected_cluster_id: Optional[int] = None):
        """
        Populates the cluster selection ComboBox with cluster IDs and info.
        """
        self.cluster_combo.clear()
        for cluster_id, info in cluster_info.items():
            display_text = f"Cluster {cluster_id} - {info['num_annotations']} annotations"
            self.cluster_combo.addItem(display_text, cluster_id)
        if selected_cluster_id is not None:
            index = self.cluster_combo.findData(selected_cluster_id)
            if index != -1:
                self.cluster_combo.setCurrentIndex(index)

        # After populating, set the current index to the one matching selected_cluster_id
        index = self.cluster_combo.findData(selected_cluster_id)
        if index != -1:
            self.cluster_combo.setCurrentIndex(index)
        self.cluster_combo.blockSignals(False)
        logging.debug(f"Populated cluster selection with IDs: {list(cluster_info.keys())}")

    def on_cluster_selected(self, index: int):
        """
        Emits a signal to sample crops from the selected cluster.

        :param index: The index of the selected cluster in the ComboBox.
        """
        cluster_id = self.cluster_combo.itemData(index)
        if cluster_id is not None:
            logging.debug(f"Cluster selected: {cluster_id}")
            self.sample_cluster.emit(cluster_id)

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
        Handles the event when a class button is clicked to label all visible crops,
        or unlabel them if class_id is None.

        :param class_id: The class ID to assign, or None to unlabel.
        """
        if class_id is None:
            logging.info("Unlabeling all visible crops.")
        else:
            logging.info(f"Class button clicked: {class_id}")

        self.class_selected_for_all.emit(class_id)

    def on_restore_autosave(self):
        """
        Emits a signal when the 'Restore Autosave' button is clicked.
        """
        self.restore_autosave_requested.emit()

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
        Updates the progress bar with the current progress value.

        :param progress: Progress percentage (0-100).
        """
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        self.progress_bar.setValue(progress)
        logging.debug(f"Progress updated to: {progress}%")

    def reset_progress(self):
        """
        Resets and shows the progress bar.
        """
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        logging.debug("Progress bar reset.")

    def hide_progress_bar(self):
        """
        Hides the progress bar.
        """
        self.progress_bar.setVisible(False)
        logging.debug("Progress bar hidden.")

    def arrange_crops(self):
        """
        Arranges the sampled crops in a grid layout that adapts to the window size
        and supports zooming functionality. Images expand and contract to fill the space.
        """
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

        for idx, crop_data in enumerate(self.selected_crops):
            annotation: Annotation = crop_data['annotation']
            q_pixmap: QPixmap = crop_data['processed_crop']
            coord_pos: tuple = crop_data['coord_pos']

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

            x_offset += image_size + 20
            max_row_height = max(max_row_height, image_size)

            if (idx + 1) % num_columns == 0:
                x_offset = 20
                y_offset += max_row_height + 40  # Adjust for both crop and label height
                max_row_height = 0

            # Adjust the scene rect
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def on_crop_class_label_changed(self, annotation_dict: dict, class_id: int):
        """
        Handles when a class label is set for an individual crop.

        :param annotation_dict: The dictionary representation of the Annotation.
        :param class_id: The new class ID assigned.
        """
        self.crop_label_changed.emit(annotation_dict, class_id)

    def on_zoom_changed(self, value: int):
        """
        Handles the zoom level change from the slider.

        :param value: The new zoom level.
        """
        self.zoom_level = max(-10, min(value, 10))  # Limit zoom_level between -10 and 10
        logging.debug(f"Zoom level changed to: {self.zoom_level}")
        self.arrange_crops()

    def eventFilter(self, source, event):
        """
        Event filter to handle mouse wheel events for zooming.

        :param source: The source of the event.
        :param event: The event object.
        :return: True if the event is handled, False otherwise.
        """
        if event.type() == QEvent.Wheel and isinstance(event, QWheelEvent):
            modifiers = event.modifiers()
            if modifiers == Qt.ControlModifier:
                delta = event.angleDelta().y() / 120  # 120 is the standard step
                self.zoom_level += delta
                self.zoom_level = max(-10, min(self.zoom_level, 10))  # Limit zoom_level
                self.zoom_slider.setValue(self.zoom_level)
                return True
        return super().eventFilter(source, event)

    def resizeEvent(self, event):
        """
        Overrides the resize event to rearrange crops when the window size changes.

        :param event: The resize event.
        """
        super().resizeEvent(event)
        logging.debug("Window resized. Rearranging crops.")
        self.arrange_crops()

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
