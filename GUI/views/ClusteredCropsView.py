import logging
from functools import partial
from typing import List, Dict, Optional

from PyQt5.QtCore import pyqtSignal, Qt, QEvent
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QComboBox, QPushButton, QVBoxLayout, QLabel,
    QProgressBar, QGraphicsView,
    QGraphicsScene, QGridLayout,
    QGroupBox, QSizePolicy, QSplitter, QGraphicsItem, QApplication,
    QHBoxLayout, QCheckBox, QScrollArea
)

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.Annotation import Annotation
from GUI.views.ClickablePixmapItem import ClickablePixmapItem
from GUI.views.LabelSlider import LabeledSlider


class ClusteredCropsView(QWidget):
    """
    ClusteredCropsView displays clustered crops in a grid layout,
    letting users interact with each crop (e.g., labeling).
    """

    # -------------------------------------------------------------------------
    #                                SIGNALS
    # -------------------------------------------------------------------------
    request_clustering = pyqtSignal()
    sample_cluster = pyqtSignal(int)
    sampling_parameters_changed = pyqtSignal(int, int)  # (cluster_id, crops_per_cluster)
    bulk_label_changed = pyqtSignal(int)  # class_id for all visible crops
    crop_label_changed = pyqtSignal(dict, int)  # (annotation_dict, class_id)
    save_project_state_requested = pyqtSignal()
    export_annotations_requested = pyqtSignal()
    save_project_requested = pyqtSignal()
    save_project_as_requested = pyqtSignal()
    load_project_state_requested = pyqtSignal()

    # -------------------------------------------------------------------------
    #                              INITIALIZATION
    # -------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.zoom_level = 5
        self.context_menu_open = False
        self.selected_crops: List[dict] = []
        self.auto_next_cluster = False

        # Main UI initialization
        self._init_ui()

        # Additional configuration
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.graphics_view.viewport().installEventFilter(self)

    def _init_ui(self):
        """
        Creates and arranges all UI components, including the splitter and panels.
        """
        self.splitter = QSplitter(Qt.Horizontal)

        # -- Left Panel: Control Panel (with scroll area)
        scroll_area = self._create_left_panel()
        scroll_area.setMinimumWidth(300)  # Ensures a minimum width for readability

        # -- Right Panel: Graphics View
        self._create_graphics_view()

        # Add widgets to the splitter
        self.splitter.addWidget(self.graphics_view)
        self.splitter.addWidget(scroll_area)
        self.splitter.splitterMoved.connect(self.on_splitter_moved)

        # Main Layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)
        self.setWindowTitle("Clustered Crops Viewer")
        self.resize(1200, 800)

    # -------------------------------------------------------------------------
    #                           LEFT PANEL CREATION
    # -------------------------------------------------------------------------
    def _create_left_panel(self) -> QScrollArea:
        """
        Creates the scrollable left panel containing all control widgets.
        """
        control_panel = QWidget()
        control_panel_layout = QVBoxLayout(control_panel)

        # --- Clustering Controls ---
        control_panel_layout.addWidget(self._create_clustering_controls())

        # --- Cluster Navigation ---
        control_panel_layout.addWidget(self._create_navigation_controls())

        # --- Zoom Controls ---
        control_panel_layout.addWidget(self._create_zoom_controls())

        # --- Class Labeling Controls ---
        control_panel_layout.addWidget(self._create_class_labels_group())

        # --- File Operations ---
        control_panel_layout.addWidget(self._create_file_operations_group())

        # Add a stretch before statistics
        control_panel_layout.addStretch()

        # --- Labeling Statistics ---
        control_panel_layout.addWidget(self._create_labeling_statistics_group())

        # Final stretch
        control_panel_layout.addStretch()

        # Wrap control panel in a QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(control_panel)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        return scroll_area

    def _create_clustering_controls(self) -> QGroupBox:
        """
        Builds the clustering controls (button + progress bars).
        """
        group = QGroupBox("Clustering")
        layout = QVBoxLayout()

        # Clustering Button
        self.clustering_button = QPushButton("Start Clustering")
        self.clustering_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clustering_button.clicked.connect(self.request_clustering.emit)
        self.clustering_button.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.clustering_button)

        # Clustering Progress Bar
        self.clustering_progress_bar = QProgressBar()
        self.clustering_progress_bar.setValue(0)
        self.clustering_progress_bar.setVisible(False)
        self.clustering_progress_bar.setFormat("Clustering: %p%")
        layout.addWidget(self.clustering_progress_bar)

        # Annotation Progress Bar
        self.annotation_progress_bar = QProgressBar()
        self.annotation_progress_bar.setValue(0)
        self.annotation_progress_bar.setVisible(False)
        self.annotation_progress_bar.setFormat("Extracting Annotations: %p%")
        layout.addWidget(self.annotation_progress_bar)

        group.setLayout(layout)
        return group

    def _create_navigation_controls(self) -> QGroupBox:
        """
        Builds the UI controls for navigating between clusters.
        """
        group = QGroupBox("Cluster Navigation")
        layout = QVBoxLayout()

        # Auto-Next checkbox
        self.auto_next_checkbox = QCheckBox("Auto Next Cluster")
        self.auto_next_checkbox.setChecked(self.auto_next_cluster)
        self.auto_next_checkbox.stateChanged.connect(self.on_auto_next_toggle)
        self.auto_next_checkbox.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.auto_next_checkbox)

        # Next / Previous Buttons + Combo
        nav_layout = QHBoxLayout()
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

        nav_layout.addWidget(self.prev_cluster_button)
        nav_layout.addWidget(self.cluster_combo)
        nav_layout.addWidget(self.next_cluster_button)
        layout.addLayout(nav_layout)

        # Navigation Hint
        navigation_hint = QLabel(
            "Shortcuts:\n"
            "Enter = Next Cluster\n"
            "Backspace = Previous Cluster\n"
            "Shift + T = Toggle Auto Next"
        )
        navigation_hint.setWordWrap(True)
        layout.addWidget(navigation_hint)

        group.setLayout(layout)
        return group

    def _create_zoom_controls(self) -> QGroupBox:
        """
        Builds the zoom slider and associated hint.
        """
        group = QGroupBox("Zoom")
        layout = QVBoxLayout()

        # LabeledSlider for zoom
        self.zoom_slider = LabeledSlider(minimum=0, maximum=10, interval=1)
        self.zoom_slider.setValue(self.zoom_level)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        layout.addWidget(self.zoom_slider)

        # Zoom Hint
        zoom_hint = QLabel("Use Ctrl + Mouse Wheel to zoom in/out")
        zoom_hint.setWordWrap(True)
        layout.addWidget(zoom_hint)

        group.setLayout(layout)
        return group

    def _create_class_labels_group(self) -> QGroupBox:
        """
        Builds the class labeling buttons (including special actions).
        """
        group = QGroupBox("Class Labels")
        layout = QGridLayout()

        # -- Special actions row --
        special_actions_layout = QHBoxLayout()
        unlabel_button = QPushButton("Unlabel (-)")
        unlabel_button.clicked.connect(partial(self.on_class_button_clicked, -1))
        unlabel_button.setFocusPolicy(Qt.NoFocus)
        unlabel_button.setStyleSheet("background-color: #f08080;")  # Light Coral
        special_actions_layout.addWidget(unlabel_button)

        unsure_button = QPushButton("Unsure (?)")
        unsure_button.clicked.connect(partial(self.on_class_button_clicked, -2))
        unsure_button.setFocusPolicy(Qt.NoFocus)
        unsure_button.setStyleSheet("background-color: #ffa500;")  # Orange
        special_actions_layout.addWidget(unsure_button)

        artifact_button = QPushButton("Artifact (!)")
        artifact_button.clicked.connect(partial(self.on_class_button_clicked, -3))
        artifact_button.setFocusPolicy(Qt.NoFocus)
        artifact_button.setStyleSheet("background-color: #d3d3d3;")  # Light Gray
        special_actions_layout.addWidget(artifact_button)

        layout.addLayout(special_actions_layout, 0, 0, 1, 3)

        # -- Class Buttons in a grid --
        row, col = 1, 0
        for class_id, class_name in CLASS_COMPONENTS.items():
            button_text = f"{class_name} ({class_id})"
            btn = QPushButton(button_text)
            btn.clicked.connect(partial(self.on_class_button_clicked, class_id))
            btn.setFocusPolicy(Qt.NoFocus)
            layout.addWidget(btn, row, col)
            col += 1
            if col >= 3:
                row += 1
                col = 0

        # -- "Agree with Model" button --
        agree_button = QPushButton("Agree with Model Predictions (Spacebar)")
        agree_button.clicked.connect(self.on_agree_with_model_clicked)
        agree_button.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        agree_button.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(agree_button, row + 1, 0, 1, 3)

        # -- Hint Label --
        hint_label = QLabel(
            "Shortcuts:\n"
            "1-9 = Class Labels\n"
            "- = Unlabel\n"
            "? = Unsure\n"
            "! = Artifact\n"
            "Spacebar = Agree with Model Predictions"
        )
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label, row + 2, 0, 1, 3)

        group.setLayout(layout)
        return group

    def _create_file_operations_group(self) -> QGroupBox:
        """
        Builds the file operations group (load, save, export).
        """
        group = QGroupBox("File Operations")
        layout = QVBoxLayout()

        self.load_project_button = QPushButton("Load Project")
        self.load_project_button.clicked.connect(self.on_load_project_state)
        self.load_project_button.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.load_project_button)

        self.save_project_button = QPushButton("Save")
        self.save_project_button.clicked.connect(self.save_project_requested.emit)
        self.save_project_button.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.save_project_button)

        self.save_project_as_button = QPushButton("Save As...")
        self.save_project_as_button.clicked.connect(self.save_project_as_requested.emit)
        self.save_project_as_button.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.save_project_as_button)

        self.export_annotations_button = QPushButton("Export Annotations")
        self.export_annotations_button.clicked.connect(self.on_export_annotations)
        self.export_annotations_button.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.export_annotations_button)

        group.setLayout(layout)
        return group

    def _create_labeling_statistics_group(self) -> QGroupBox:
        """
        Builds the labeling statistics group, showing counts and percentages.
        """
        group = QGroupBox("Labeling Statistics")
        layout = QVBoxLayout()

        # Dictionary to hold reference labels for updates
        self.class_counts_labels = {}

        # Class counts (for standard classes)
        for class_id in sorted(CLASS_COMPONENTS.keys()):
            class_name = CLASS_COMPONENTS[class_id]
            label = QLabel(f"{class_name}: 0")
            layout.addWidget(label)
            self.class_counts_labels[class_id] = label

        # Artifact, Unsure, Unlabeled
        self.artifact_label = QLabel("Artifact: 0")
        layout.addWidget(self.artifact_label)

        self.unsure_label = QLabel("Unsure: 0")
        layout.addWidget(self.unsure_label)

        self.unlabeled_label = QLabel("Unlabeled: 0")
        layout.addWidget(self.unlabeled_label)

        # Disagreement Statistics
        self.disagreement_label = QLabel("Disagreements: 0")
        layout.addWidget(self.disagreement_label)

        # Total Annotations + Labeled
        self.total_annotations_label = QLabel("Total Annotations: 0")
        layout.addWidget(self.total_annotations_label)

        self.total_labeled_label = QLabel("Total Labeled Annotations: 0")
        layout.addWidget(self.total_labeled_label)

        group.setLayout(layout)
        return group

    # -------------------------------------------------------------------------
    #                         RIGHT PANEL: GRAPHICS VIEW
    # -------------------------------------------------------------------------
    def _create_graphics_view(self):
        """
        Creates the QGraphicsView and associated scene/controls on the right panel.
        """
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.graphics_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.graphics_view.setFocusPolicy(Qt.NoFocus)
        self.graphics_view.setMouseTracking(True)

        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)

        self.crop_loading_progress_bar = QProgressBar(self.graphics_view.viewport())
        self.crop_loading_progress_bar.setFixedSize(300, 25)
        self.crop_loading_progress_bar.setAlignment(Qt.AlignCenter)
        self.crop_loading_progress_bar.setVisible(False)

    # -------------------------------------------------------------------------
    #                              EVENT HANDLING
    # -------------------------------------------------------------------------
    def keyPressEvent(self, event):
        """
        Handles keyboard shortcuts for labeling, navigation, etc.
        """
        key = event.key()

        # Number keys for class selection (0 to 8)
        if Qt.Key_0 <= key <= Qt.Key_8:
            class_id = key - Qt.Key_0
            if class_id in CLASS_COMPONENTS:
                self.on_class_button_clicked(class_id)
                return

        # Minus -> unlabel
        if key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.on_class_button_clicked(-1)
            return

        # "?" -> unsure (-2)
        if event.text() == '?':
            self.on_class_button_clicked(-2)
            return

        # "!" -> artifact (-3)
        if event.text() == '!':
            self.on_class_button_clicked(-3)
            return

        # Space -> agree with model predictions
        if key == Qt.Key_Space:
            self.on_agree_with_model_clicked()
            return

        # Shift + T -> toggle auto-next
        if key == Qt.Key_T and event.modifiers() == Qt.ShiftModifier:
            self.auto_next_cluster = not self.auto_next_cluster
            self.auto_next_checkbox.setChecked(self.auto_next_cluster)
            return

        # Enter -> next cluster
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.on_next_cluster()
            return

        # Backspace -> prev cluster
        if key == Qt.Key_Backspace:
            self.on_prev_cluster()
            return

        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        """
        Handles zoom via Ctrl + Mouse Wheel on the graphics view.
        """
        if obj == self.graphics_view.viewport() and event.type() == QEvent.Wheel:
            if event.modifiers() == Qt.ControlModifier:
                delta = event.angleDelta().y() / 120
                self.zoom_level = max(0, min(self.zoom_level + delta, 10))
                self.zoom_slider.setValue(self.zoom_level)
                self.arrange_crops()
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        """
        Rearranges crops and re-centers progress bars when window is resized.
        """
        super().resizeEvent(event)
        logging.debug("Window resized. Rearranging crops.")
        self.arrange_crops()
        if self.crop_loading_progress_bar.isVisible():
            self._center_crop_loading_progress_bar()

    def showEvent(self, event):
        """
        Sets initial splitter sizes when the window first shows.
        """
        super().showEvent(event)
        total_width = self.width()
        control_panel_width = total_width // 3
        graphics_view_width = total_width - control_panel_width
        self.splitter.setSizes([graphics_view_width, control_panel_width])

    # -------------------------------------------------------------------------
    #                              PROGRESS BARS
    # -------------------------------------------------------------------------
    def show_clustering_progress_bar(self):
        self.clustering_progress_bar.setVisible(True)

    def hide_clustering_progress_bar(self):
        self.clustering_progress_bar.setVisible(False)

    def update_clustering_progress_bar(self, progress: int):
        self.show_clustering_progress_bar()
        if progress == -1:
            # Indeterminate mode
            self.clustering_progress_bar.setRange(0, 0)
            self.clustering_progress_bar.setFormat("Clustering, please wait...")
        else:
            # Normal progress mode
            self.clustering_progress_bar.setRange(0, 100)
            self.clustering_progress_bar.setValue(progress)
            self.clustering_progress_bar.setFormat(f"Clustering: {progress}%")

    def update_annotation_progress_bar(self, progress: int):
        if not self.annotation_progress_bar.isVisible():
            self.annotation_progress_bar.setVisible(True)
        self.annotation_progress_bar.setValue(progress)
        self.annotation_progress_bar.setFormat(f"Extracting Annotations: {progress}%")

    def hide_annotation_progress_bar(self):
        self.annotation_progress_bar.setVisible(False)

    def show_crop_loading_progress_bar(self):
        self.crop_loading_progress_bar.setValue(0)
        self.crop_loading_progress_bar.setVisible(True)
        QApplication.processEvents()
        self._center_crop_loading_progress_bar()

    def hide_crop_loading_progress_bar(self):
        self.crop_loading_progress_bar.setVisible(False)

    def update_crop_loading_progress_bar(self, progress: int):
        if not self.crop_loading_progress_bar.isVisible():
            self.crop_loading_progress_bar.setVisible(True)
        self.crop_loading_progress_bar.setValue(progress)
        logging.debug(f"Progress updated to: {progress}%")

    def reset_progress(self):
        self.crop_loading_progress_bar.setValue(0)
        self.crop_loading_progress_bar.setVisible(True)
        logging.debug("Progress bar reset.")

    def hide_progress_bar(self):
        self.crop_loading_progress_bar.setVisible(False)
        logging.debug("Progress bar hidden.")

    def _center_crop_loading_progress_bar(self):
        viewport_rect = self.graphics_view.viewport().rect()
        x = (viewport_rect.width() - self.crop_loading_progress_bar.width()) // 2
        y = (viewport_rect.height() - self.crop_loading_progress_bar.height()) // 2
        self.crop_loading_progress_bar.move(x, y)

    # -------------------------------------------------------------------------
    #                         CLUSTER NAVIGATION
    # -------------------------------------------------------------------------
    def on_cluster_selected(self, index: int):
        cluster_id = self.cluster_combo.itemData(index)
        if cluster_id is not None:
            logging.debug(f"Cluster selected: {cluster_id}")
            self.sample_cluster.emit(cluster_id)

        # Update next/prev button states
        self.prev_cluster_button.setEnabled(index > 0)
        self.next_cluster_button.setEnabled(index < self.cluster_combo.count() - 1)

    def on_prev_cluster(self):
        current_index = self.cluster_combo.currentIndex()
        if current_index > 0:
            self.cluster_combo.setCurrentIndex(current_index - 1)
        self.setFocus()

    def on_next_cluster(self):
        current_index = self.cluster_combo.currentIndex()
        if current_index < self.cluster_combo.count() - 1:
            self.cluster_combo.setCurrentIndex(current_index + 1)
        self.setFocus()

    def on_auto_next_toggle(self, state):
        self.auto_next_cluster = bool(state)

    # -------------------------------------------------------------------------
    #                          ZOOM & SPLITTER
    # -------------------------------------------------------------------------
    def on_zoom_changed(self, value: int):
        self.zoom_level = max(0, min(value, 10))
        logging.debug(f"Zoom level changed to: {self.zoom_level}")
        self.arrange_crops()

    def on_splitter_moved(self, pos, index):
        self.arrange_crops()
        if self.crop_loading_progress_bar.isVisible():
            self._center_crop_loading_progress_bar()

    # -------------------------------------------------------------------------
    #                           DISPLAYING CROPS
    # -------------------------------------------------------------------------
    def display_sampled_crops(self, sampled_crops: List[dict]):
        """
        Displays the given sampled_crops in the scene.
        """
        self.selected_crops = sampled_crops
        self.arrange_crops()

    def arrange_crops(self):
        """
        Lays out the currently selected crops in a grid arrangement.
        """
        if getattr(self.scene, 'context_menu_open', False):
            return  # Skip rearranging if a context menu is open

        self.scene.clear()
        if not self.selected_crops:
            logging.info("No sampled crops to display.")
            return

        viewport_width = self.graphics_view.viewport().width()
        base_size = 150  # Base size at zoom = 0
        scale_factor = 1.2 ** self.zoom_level
        image_size = base_size * scale_factor

        # Determine number of columns
        total_spacing = 20 * 2  # left/right spacing
        # Give some margin per item
        approximate_item_width = image_size + 20
        num_columns = max(1, int((viewport_width - total_spacing) // approximate_item_width))

        # Adjust image_size so columns fill horizontally
        used_width_for_images = (viewport_width - (num_columns + 1) * 20)
        if num_columns > 0:
            image_size = used_width_for_images / num_columns

        logging.debug(f"Calculated image size: {image_size}, columns: {num_columns}")

        x_offset, y_offset = 20, 20
        max_row_height = 0
        arranged_count = 0

        for idx, crop_data in enumerate(self.selected_crops):
            annotation: Annotation = crop_data['annotation']
            pixmap: QPixmap = crop_data['processed_crop']

            if pixmap.isNull():
                logging.warning(f"Invalid QPixmap for image index {annotation.image_index}. Skipping.")
                continue

            pixmap_item = ClickablePixmapItem(annotation=annotation, pixmap=pixmap)
            pixmap_item.setFlag(QGraphicsItem.ItemIsSelectable, True)
            pixmap_item.class_label_changed.connect(self.crop_label_changed.emit)

            # Scale the pixmap to the target width
            original_width = pixmap.width()
            if original_width == 0:
                logging.warning("Crop width is zero, skipping image to prevent division by zero.")
                continue
            scale = image_size / original_width
            pixmap_item.setScaleFactor(scale)

            # Position the item
            bounding_rect = pixmap_item.boundingRect()
            pixmap_item.setPos(x_offset, y_offset)
            self.scene.addItem(pixmap_item)
            arranged_count += 1

            # Move offsets
            x_offset += bounding_rect.width() + 20
            max_row_height = max(max_row_height, bounding_rect.height())

            if (idx + 1) % num_columns == 0:
                x_offset = 20
                y_offset += max_row_height + 20
                max_row_height = 0

        # Validate the arrangement
        expected = len(self.selected_crops)
        assert arranged_count == expected, (
            f"View arrangement mismatch: expected {expected}, arranged {arranged_count}"
        )
        logging.info(f"Arranged {arranged_count} crops in the view.")

        # Resize scene to fit items
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

        # Re-center loading bar if needed
        if self.crop_loading_progress_bar.isVisible():
            self._center_crop_loading_progress_bar()

    # -------------------------------------------------------------------------
    #                           LABELING & STATS
    # -------------------------------------------------------------------------
    def on_class_button_clicked(self, class_id: Optional[int]):
        """
        Assigns the given class_id to all visible crops (bulk label).
        """
        if class_id == -3:
            logging.info("Classifying current cluster as artifact.")
        elif class_id == -2:
            logging.info("Classifying current cluster as unsure.")
        elif class_id == -1:
            logging.info("Unlabeling current cluster.")
        else:
            logging.info(f"Assigning class {class_id} to current cluster.")

        self._label_all_visible_crops(class_id)

        if self.auto_next_cluster:
            self.on_next_cluster()
        self.setFocus()

    def _label_all_visible_crops(self, class_id: Optional[int]):
        """
        Labels all visible crops in the scene with the given class_id.
        """
        if class_id is None:
            class_id = -1
        for crop in self.selected_crops:
            annotation: Annotation = crop['annotation']
            annotation.class_id = class_id

        self.bulk_label_changed.emit(class_id)
        self.arrange_crops()

    def on_agree_with_model_clicked(self):
        """
        Assigns each visible crop its model-predicted class, if available.
        """
        changes_made = False
        for crop_data in self.selected_crops:
            annotation = crop_data['annotation']
            if annotation.model_prediction is not None:
                model_class_id = self.get_class_id_from_model_prediction(annotation.model_prediction)
                if model_class_id is not None:
                    old_class = annotation.class_id
                    annotation.class_id = model_class_id
                    if annotation.class_id != old_class:
                        changes_made = True
                        self.crop_label_changed.emit(annotation.to_dict(), model_class_id)

        if changes_made:
            self.arrange_crops()

    def get_class_id_from_model_prediction(self, model_prediction: str) -> Optional[int]:
        """
        Maps a model-predicted class name back to its class_id, if found.
        """
        for cid, cname in CLASS_COMPONENTS.items():
            if cname == model_prediction:
                return cid
        return None

    def update_labeling_statistics(self, statistics: dict):
        """
        Updates the labeling statistics section with new data.
        """
        total_annotations = statistics['total_annotations']
        total_labeled = statistics['total_labeled']
        class_counts = statistics['class_counts']
        disagreement_count = statistics.get('disagreement_count', 0)
        agreement_percentage = statistics.get('agreement_percentage', 0.0)

        self.total_annotations_label.setText(f"Total Annotations: {total_annotations}")
        self.total_labeled_label.setText(f"Total Labeled Annotations: {total_labeled}")

        # Update class counts
        for cid, count in class_counts.items():
            if cid in CLASS_COMPONENTS:
                label = self.class_counts_labels.get(cid)
                if label:
                    cname = CLASS_COMPONENTS[cid]
                    label.setText(f"{cname} ({cid}): {count}")
            elif cid == -1:
                self.unlabeled_label.setText(f"Unlabeled: {count}")
            elif cid == -2:
                self.unsure_label.setText(f"Unsure: {count}")
            elif cid == -3:
                self.artifact_label.setText(f"Artifact: {count}")

        # Display global disagreement stats
        self.disagreement_label.setText(
            f"Disagreements: {disagreement_count} (Agreement: {agreement_percentage:.2f}%)"
        )

    # -------------------------------------------------------------------------
    #                         FILE & PROJECT SIGNALS
    # -------------------------------------------------------------------------
    def on_load_project_state(self):
        self.load_project_state_requested.emit()

    def on_save_project_state(self):
        self.save_project_state_requested.emit()

    def on_export_annotations(self):
        self.export_annotations_requested.emit()

    def get_selected_cluster_id(self) -> Optional[int]:
        """
        Returns the currently selected cluster ID in the combo box.
        """
        index = self.cluster_combo.currentIndex()
        if index != -1:
            return int(self.cluster_combo.itemData(index))
        return None

    def get_cluster_id_list(self) -> List[int]:
        """
        Returns the list of cluster IDs from the combo box in order.
        """
        ids = []
        for i in range(self.cluster_combo.count()):
            cid = self.cluster_combo.itemData(i)
            ids.append(cid)
        return ids

    def populate_cluster_selection(self, cluster_info: Dict[int, dict], selected_cluster_id: Optional[int] = None):
        """
        Populates the cluster combo with cluster IDs and info.
        """
        self.cluster_combo.blockSignals(True)
        self.cluster_combo.clear()

        for cid, info in cluster_info.items():
            num_annotations = info.get('num_annotations', '?')
            labeled_percentage = info.get('labeled_percentage', 0.0)
            avg_uncertainty = info.get('average_uncertainty', 0.0)
            cluster_label = info.get('label', '')

            text = (
                f"Cluster {cid} - {num_annotations} annotations, "
                f"{labeled_percentage:.2f}% assessed, "
                f"Avg Uncertainty: {avg_uncertainty:.2f}"
            )
            if cluster_label:
                text += f" - label: {cluster_label}"

            self.cluster_combo.addItem(text, cid)

        if selected_cluster_id is not None:
            index = self.cluster_combo.findData(selected_cluster_id)
            if index != -1:
                self.cluster_combo.setCurrentIndex(index)

        self.cluster_combo.blockSignals(False)

        # Update next/prev buttons
        has_clusters = (self.cluster_combo.count() > 0)
        current_index = self.cluster_combo.currentIndex()
        self.prev_cluster_button.setEnabled(has_clusters and current_index > 0)
        self.next_cluster_button.setEnabled(has_clusters and current_index < self.cluster_combo.count() - 1)
