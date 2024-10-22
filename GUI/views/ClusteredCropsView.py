import logging

from PyQt5.QtCore import pyqtSignal, Qt, QRectF, QEvent
from PyQt5.QtGui import QWheelEvent, QPainter, QPen
from PyQt5.QtWidgets import (
    QWidget, QComboBox, QPushButton, QVBoxLayout, QLabel,
    QProgressBar, QSpinBox, QSlider, QGraphicsView,
    QGraphicsScene, QGraphicsItem, QLineEdit, QMenu, QGridLayout, QFileDialog,
    QGraphicsObject, QGroupBox, QAction, QSizePolicy,
    QSplitter
)

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
    class_label_changed = pyqtSignal(dict, int)  # Emits (crop_data, class_id)
    crop_removed = pyqtSignal(dict)  # Emits crop_data

    def __init__(self, crop_data, pixmap, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_data = crop_data
        self.pixmap = pixmap
        self.setAcceptHoverEvents(True)
        self.hovered = False  # Track hover state
        self.selected = False  # Track selection state

    def boundingRect(self):
        return QRectF(0, 0, self.pixmap.width(), self.pixmap.height())

    def paint(self, painter, option, widget):
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
        # Add actions for each class
        for class_id, class_name in CLASS_COMPONENTS.items():
            action = QAction(class_name, self)
            action.triggered.connect(lambda checked, cid=class_id: self.set_crop_class(cid))
            menu.addAction(action)
        # Separator and Remove Option
        menu.addSeparator()
        remove_action = QAction("Remove from Cluster", self)
        remove_action.triggered.connect(self.remove_crop)
        menu.addAction(remove_action)
        # Display the menu
        menu.exec_(event.screenPos())

    def set_crop_class(self, class_id):
        self.crop_data['class_id'] = class_id
        self.class_label_changed.emit(self.crop_data, class_id)
        self.update()

    def remove_crop(self):
        self.crop_removed.emit(self.crop_data)


class ClusteredCropsView(QWidget):
    # Define signals to communicate with the controller
    request_clustering = pyqtSignal()
    sample_cluster = pyqtSignal(int)
    sampling_parameters_changed = pyqtSignal(int, int)  # Emits (cluster_id, crops_per_cluster)
    crop_clicked = pyqtSignal(dict)
    crop_removed = pyqtSignal(dict)
    cluster_labeled = pyqtSignal(int, str)
    class_selected = pyqtSignal(int, int)  # Emits (cluster_id, class_id)
    cluster_file_selected = pyqtSignal(str)  # Emits the filename of the selected cluster file
    save_annotations_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.zoom_level = 0  # Initial zoom level
        self.sampled_crops = []  # Store sampled crops for dynamic rearrangement
        self.init_ui()

    def init_ui(self):
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
        self.crops_spinbox.setRange(1, 100)
        self.crops_spinbox.setValue(10)
        self.crops_spinbox.valueChanged.connect(self.on_crops_changed)
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
        zoom_layout.addWidget(self.zoom_slider)
        zoom_group.setLayout(zoom_layout)
        control_panel_layout.addWidget(zoom_group)

        # Cluster Labeling Group
        labeling_group = QGroupBox("Cluster Labeling")
        labeling_layout = QVBoxLayout()
        self.cluster_label_edit = QLineEdit()
        self.cluster_label_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        labeling_layout.addWidget(QLabel("Cluster Label:"))
        labeling_layout.addWidget(self.cluster_label_edit)
        self.label_cluster_button = QPushButton("Set Label")
        self.label_cluster_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.label_cluster_button.clicked.connect(self.on_label_cluster)
        labeling_layout.addWidget(self.label_cluster_button)
        labeling_group.setLayout(labeling_layout)
        control_panel_layout.addWidget(labeling_group)

        # Class Selection Buttons Group
        class_buttons_group = QGroupBox("Class Labels")
        class_buttons_layout = QGridLayout()
        row = 0
        col = 0
        for class_id, class_name in CLASS_COMPONENTS.items():
            button = QPushButton(class_name)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(lambda checked, cid=class_id: self.on_class_button_clicked(cid))
            class_buttons_layout.addWidget(button, row, col)
            col += 1
            if col >= 2:  # Adjust number of buttons per row
                col = 0
                row += 1
        class_buttons_group.setLayout(class_buttons_layout)
        control_panel_layout.addWidget(class_buttons_group)

        # Add a spacer to ensure dynamic resizing
        control_panel_layout.addStretch()

        # Save and Load Buttons
        self.load_button = QPushButton("Load Cluster File")
        self.load_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.load_button.clicked.connect(self.on_load_cluster_file)
        self.save_annotations_button = QPushButton("Save Annotations")
        self.save_annotations_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.save_annotations_button.clicked.connect(self.on_save_annotations)
        control_panel_layout.addWidget(self.load_button)
        control_panel_layout.addWidget(self.save_annotations_button)

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

    def populate_cluster_selection(self, cluster_info, selected_cluster_id=None):
        """
        Populates the cluster selection ComboBox with available cluster IDs and their info.
        """
        self.cluster_combo.blockSignals(True)  # Prevent triggering selection signal during population
        if selected_cluster_id is None:
            selected_cluster_id = self.get_selected_cluster_id()
        self.cluster_combo.clear()
        for cid, info in cluster_info.items():
            num_annotations = info['num_annotations']
            num_images = info['num_images']
            label = info.get('label', '')
            if label:
                display_text = f"Cluster {cid} - '{label}' ({num_annotations} annotations from {num_images} images)"
            else:
                display_text = f"Cluster {cid} ({num_annotations} annotations from {num_images} images)"
            self.cluster_combo.addItem(display_text, cid)
        # After populating, set the current index to the one matching selected_cluster_id
        index = self.cluster_combo.findData(selected_cluster_id)
        if index != -1:
            self.cluster_combo.setCurrentIndex(index)
        self.cluster_combo.blockSignals(False)
        logging.debug(f"Populated cluster selection with IDs: {list(cluster_info.keys())}")

    def on_cluster_selected(self, index):
        """
        Emits a signal to sample crops from the selected cluster.
        """
        cluster_id = self.cluster_combo.itemData(index)
        if cluster_id is not None:
            logging.debug(f"Cluster selected: {cluster_id}")
            self.sample_cluster.emit(cluster_id)

    def on_crops_changed(self, value):
        """
        Emits a signal when the number of crops per cluster changes.
        """
        current_cluster_id = self.get_selected_cluster_id()
        if current_cluster_id is not None:
            logging.debug(f"Number of crops per cluster changed to: {value}")
            self.sampling_parameters_changed.emit(current_cluster_id, value)

    def on_class_button_clicked(self, class_id):
        """
        Handles the event when a class button is clicked.
        """
        logging.debug(f"Class button clicked: {class_id}")
        cluster_id = self.get_selected_cluster_id()
        if cluster_id is not None:
            self.class_selected.emit(cluster_id, class_id)
        else:
            logging.warning("No cluster is currently selected.")

    def get_selected_cluster_id(self):
        """
        Retrieves the currently selected cluster ID.
        """
        return self.cluster_combo.currentData()

    def display_sampled_crops(self, sampled_crops):
        """
        Displays the sampled crops in a grid layout within the graphics scene.
        """
        self.sampled_crops = sampled_crops
        self.arrange_crops()

    def update_progress(self, progress):
        """
        Updates the progress bar with the current progress value.
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

        if not self.sampled_crops:
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

        for idx, crop in enumerate(self.sampled_crops):
            if crop['crop'].isNull():
                logging.warning(f"Invalid QPixmap for image index {crop['image_index']}. Skipping.")
                continue

            pixmap_item = ClickablePixmapItem(crop, crop['crop'])
            pixmap_item.setFlag(QGraphicsItem.ItemIsSelectable, True)

            # Connect the signals
            pixmap_item.class_label_changed.connect(self.on_crop_class_label_changed)
            pixmap_item.crop_removed.connect(self.on_crop_removed)

            original_width = crop['crop'].width()
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
                y_offset += max_row_height + 20
                max_row_height = 0

        # Adjust the scene rect
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def on_crop_class_label_changed(self, crop_data, class_id):
        """
        Handles when a class label is set for an individual crop.
        """
        logging.debug(f"Crop {crop_data['image_index']} class label changed to {class_id}")
        # Optionally, update the visual representation to reflect the new class label
        # For example, you could change the border color based on the class

    def on_crop_removed(self, crop_data):
        """
        Handles when a crop is removed from the cluster.
        """
        logging.debug(f"Crop {crop_data['image_index']} removed from cluster")
        # Remove the crop from the sampled_crops list
        self.sampled_crops.remove(crop_data)
        # Rearrange crops
        self.arrange_crops()

    def on_zoom_changed(self, value):
        """
        Handles the zoom level change from the slider.
        """
        self.zoom_level = max(-10, min(value, 10))  # Limit zoom_level between -10 and 10
        logging.debug(f"Zoom level changed to: {self.zoom_level}")
        self.arrange_crops()

    def on_load_cluster_file(self):
        """
        Opens a file dialog to select a cluster file to load.
        """
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Open Cluster File", "", "JSON Files (*.json);;All Files (*)",
                                                  options=options)
        if filename:
            logging.debug(f"Selected cluster file: {filename}")
            self.cluster_file_selected.emit(filename)
        else:
            logging.debug("No cluster file selected.")

    def on_save_annotations(self):
        """
        Handles the event when the 'Save Annotations' button is clicked.
        """
        self.save_annotations_requested.emit()

    def eventFilter(self, source, event):
        """
        Event filter to handle mouse wheel events for zooming.
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
        """
        super().resizeEvent(event)
        logging.debug("Window resized. Rearranging crops.")
        self.arrange_crops()

    def on_crop_clicked(self, crop_data):
        """
        Handles the event when a crop is clicked.
        """
        pass

    def on_label_cluster(self):
        """
        Handles the event when the 'Set Label' button is clicked.
        """
        label = self.cluster_label_edit.text()
        cluster_id = self.get_selected_cluster_id()
        if cluster_id is not None and label:
            self.cluster_labeled.emit(cluster_id, label)
        else:
            logging.warning("Cluster ID or label is missing.")
