import logging
from functools import partial
from typing import List, Dict, Optional

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QComboBox, QPushButton, QVBoxLayout, QLabel,
    QProgressBar, QGraphicsView, QGraphicsScene, QGridLayout,
    QGroupBox, QSizePolicy, QSplitter, QGraphicsItem, QApplication,
    QHBoxLayout, QScrollArea
)

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.Annotation import Annotation
from GUI.views.ClickablePixmapItem import ClickablePixmapItem
from GUI.views.LabelSlider import LabeledSlider

# ---------------------------------------------------------------------------
# Module-Level Constants
# ---------------------------------------------------------------------------
SPACING = 20
BASE_IMAGE_SIZE = 150
MIN_ZOOM = 0
MAX_ZOOM = 10


# ---------------------------------------------------------------------------
# Clustering Controls Widget
# ---------------------------------------------------------------------------
class ClusteringControlsWidget(QGroupBox):
    request_clustering = pyqtSignal()

    def __init__(self):
        super().__init__("Clustering")
        self._init_ui()

    def _init_ui(self):
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

        self.setLayout(layout)

    def update_clustering_progress(self, progress: int):
        self.clustering_progress_bar.setVisible(True)
        if progress == -1:
            self.clustering_progress_bar.setRange(0, 0)
            self.clustering_progress_bar.setFormat("Clustering, please wait...")
        else:
            self.clustering_progress_bar.setRange(0, 100)
            self.clustering_progress_bar.setValue(progress)
            self.clustering_progress_bar.setFormat(f"Clustering: {progress}%")

    def hide_clustering_progress(self):
        self.clustering_progress_bar.setVisible(False)

    def update_annotation_progress(self, progress: int):
        if not self.annotation_progress_bar.isVisible():
            self.annotation_progress_bar.setVisible(True)
        self.annotation_progress_bar.setValue(progress)
        self.annotation_progress_bar.setFormat(f"Extracting Annotations: {progress}%")

    def hide_annotation_progress(self):
        self.annotation_progress_bar.setVisible(False)


# ---------------------------------------------------------------------------
# Navigation Controls Widget
# ---------------------------------------------------------------------------
class NavigationControlsWidget(QGroupBox):
    sample_cluster = pyqtSignal(int)

    def __init__(self):
        super().__init__("Cluster Navigation")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Navigation Buttons and Combo Box
        nav_layout = QHBoxLayout()
        self.prev_cluster_button = QPushButton("Previous (Backspace)")
        self.prev_cluster_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.prev_cluster_button.clicked.connect(self.on_prev_cluster)
        self.prev_cluster_button.setEnabled(False)
        self.prev_cluster_button.setFocusPolicy(Qt.NoFocus)

        self.cluster_combo = QComboBox()
        self.cluster_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cluster_combo.currentIndexChanged.connect(self.on_cluster_selected)
        self.cluster_combo.setFocusPolicy(Qt.NoFocus)

        self.next_cluster_button = QPushButton("Next (Enter)")
        self.next_cluster_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.next_cluster_button.clicked.connect(self.on_next_cluster)
        self.next_cluster_button.setEnabled(False)
        self.next_cluster_button.setFocusPolicy(Qt.NoFocus)

        nav_layout.addWidget(self.prev_cluster_button)
        nav_layout.addWidget(self.cluster_combo)
        nav_layout.addWidget(self.next_cluster_button)
        layout.addLayout(nav_layout)

        self.setLayout(layout)

    def on_cluster_selected(self, index: int):
        cluster_id = self.cluster_combo.itemData(index)
        if cluster_id is not None:
            logging.debug(f"Cluster selected: {cluster_id}")
            self.sample_cluster.emit(cluster_id)
        # Update navigation button states
        self.prev_cluster_button.setEnabled(index > 0)
        self.next_cluster_button.setEnabled(index < self.cluster_combo.count() - 1)

    def on_prev_cluster(self):
        current_index = self.cluster_combo.currentIndex()
        if current_index > 0:
            self.cluster_combo.setCurrentIndex(current_index - 1)

    def on_next_cluster(self):
        current_index = self.cluster_combo.currentIndex()
        if current_index < self.cluster_combo.count() - 1:
            self.cluster_combo.setCurrentIndex(current_index + 1)

    def populate_clusters(self, cluster_info: Dict[int, dict], selected_cluster_id: Optional[int] = None):
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
        has_clusters = (self.cluster_combo.count() > 0)
        current_index = self.cluster_combo.currentIndex()
        self.prev_cluster_button.setEnabled(has_clusters and current_index > 0)
        self.next_cluster_button.setEnabled(has_clusters and current_index < self.cluster_combo.count() - 1)

    def get_selected_cluster_id(self) -> Optional[int]:
        index = self.cluster_combo.currentIndex()
        if index != -1:
            return int(self.cluster_combo.itemData(index))
        return None

    def get_cluster_id_list(self) -> List[int]:
        return [self.cluster_combo.itemData(i) for i in range(self.cluster_combo.count())]


# ---------------------------------------------------------------------------
# Zoom Controls Widget
# ---------------------------------------------------------------------------
class ZoomControlsWidget(QGroupBox):
    zoom_changed = pyqtSignal(int)

    def __init__(self, initial_zoom: int = 5):
        super().__init__("Zoom")
        self._init_ui(initial_zoom)

    def _init_ui(self, initial_zoom: int):
        layout = QVBoxLayout()
        self.zoom_slider = LabeledSlider(minimum=MIN_ZOOM, maximum=MAX_ZOOM, interval=1)
        self.zoom_slider.setValue(initial_zoom)
        self.zoom_slider.valueChanged.connect(self.zoom_changed.emit)
        layout.addWidget(self.zoom_slider)

        zoom_hint = QLabel("Use Ctrl + Mouse Wheel to zoom in/out")
        zoom_hint.setWordWrap(True)
        layout.addWidget(zoom_hint)

        self.setLayout(layout)

    def get_zoom_level(self) -> int:
        return self.zoom_slider.value()


# ---------------------------------------------------------------------------
# Class Labels Widget
# ---------------------------------------------------------------------------
class ClassLabelsWidget(QGroupBox):
    class_label_selected = pyqtSignal(int)
    agree_with_model = pyqtSignal()

    def __init__(self):
        super().__init__("Labelling Controls")
        self._init_ui()

    def _init_ui(self):
        layout = QGridLayout()

        # Special Actions Row
        special_actions_layout = QHBoxLayout()
        unlabel_button = QPushButton("Unlabel (-)")
        unlabel_button.clicked.connect(partial(self.class_label_selected.emit, -1))
        unlabel_button.setFocusPolicy(Qt.NoFocus)
        unlabel_button.setStyleSheet("background-color: #f08080;")
        special_actions_layout.addWidget(unlabel_button)

        unsure_button = QPushButton("Unsure (?)")
        unsure_button.clicked.connect(partial(self.class_label_selected.emit, -2))
        unsure_button.setFocusPolicy(Qt.NoFocus)
        unsure_button.setStyleSheet("background-color: #ffa500;")
        special_actions_layout.addWidget(unsure_button)

        artifact_button = QPushButton("Artifact (!)")
        artifact_button.clicked.connect(partial(self.class_label_selected.emit, -3))
        artifact_button.setFocusPolicy(Qt.NoFocus)
        artifact_button.setStyleSheet("background-color: #d3d3d3;")
        special_actions_layout.addWidget(artifact_button)

        layout.addLayout(special_actions_layout, 0, 0, 1, 3)

        # Class Buttons Grid
        row, col = 1, 0
        for class_id, class_name in CLASS_COMPONENTS.items():
            button_text = f"{class_name} ({class_id})"
            btn = QPushButton(button_text)
            btn.clicked.connect(partial(self.class_label_selected.emit, class_id))
            btn.setFocusPolicy(Qt.NoFocus)
            layout.addWidget(btn, row, col)
            col += 1
            if col >= 3:
                row += 1
                col = 0

        # "Agree with Model" Button
        agree_button = QPushButton("Agree with Model Predictions (Spacebar)")
        agree_button.clicked.connect(lambda: self.agree_with_model.emit())
        agree_button.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        agree_button.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(agree_button, row + 1, 0, 1, 3)

        # Hint Label
        hint_label = QLabel(
            "Shortcuts:\n"
            "Number Keys 1-9 = Main Class Labels\n"
            "- = Unlabel\n"
            "? = Unsure\n"
            "! = Artifact\n"
            "Spacebar = Agree with Model Predictions"
        )
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label, row + 2, 0, 1, 3)

        self.setLayout(layout)


# ---------------------------------------------------------------------------
# File Operations Widget
# ---------------------------------------------------------------------------
class FileOperationsWidget(QGroupBox):
    load_project_state_requested = pyqtSignal()
    save_project_requested = pyqtSignal()
    save_project_as_requested = pyqtSignal()
    export_annotations_requested = pyqtSignal()

    def __init__(self):
        super().__init__("File Operations")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.load_project_button = QPushButton("Load Project")
        self.load_project_button.clicked.connect(self.load_project_state_requested.emit)
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
        self.export_annotations_button.clicked.connect(self.export_annotations_requested.emit)
        self.export_annotations_button.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.export_annotations_button)

        self.setLayout(layout)


# ---------------------------------------------------------------------------
# Labeling Statistics Widget
# ---------------------------------------------------------------------------
class LabelingStatisticsWidget(QGroupBox):
    def __init__(self):
        super().__init__("Labeling Statistics")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.class_counts_labels = {}
        for class_id in sorted(CLASS_COMPONENTS.keys()):
            class_name = CLASS_COMPONENTS[class_id]
            label = QLabel(f"{class_name}: 0")
            layout.addWidget(label)
            self.class_counts_labels[class_id] = label

        self.artifact_label = QLabel("Artifact: 0")
        layout.addWidget(self.artifact_label)
        self.unsure_label = QLabel("Unsure: 0")
        layout.addWidget(self.unsure_label)
        self.unlabeled_label = QLabel("Unlabeled: 0")
        layout.addWidget(self.unlabeled_label)
        self.disagreement_label = QLabel("Disagreements: 0")
        layout.addWidget(self.disagreement_label)
        self.total_annotations_label = QLabel("Total Annotations: 0")
        layout.addWidget(self.total_annotations_label)
        self.total_labeled_label = QLabel("Total Labeled Annotations: 0")
        layout.addWidget(self.total_labeled_label)

        self.setLayout(layout)

    def update_statistics(self, statistics: dict):
        total_annotations = statistics['total_annotations']
        total_labeled = statistics['total_labeled']
        class_counts = statistics['class_counts']
        disagreement_count = statistics.get('disagreement_count', 0)
        agreement_percentage = statistics.get('agreement_percentage', 0.0)

        self.total_annotations_label.setText(f"Total Annotations: {total_annotations}")
        self.total_labeled_label.setText(f"Total Labeled Annotations: {total_labeled}")

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

        self.disagreement_label.setText(
            f"Disagreements: {disagreement_count} (Agreement: {agreement_percentage:.2f}%)"
        )


# ---------------------------------------------------------------------------
# Custom Graphics View for Crop Display and Zoom Handling
# ---------------------------------------------------------------------------
class CropGraphicsView(QGraphicsView):
    def __init__(self, main_view, parent=None):
        super().__init__(parent)
        self.main_view = main_view  # Store explicit reference to the main view
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setFocusPolicy(Qt.NoFocus)
        self.setMouseTracking(True)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            self.main_view.handle_zoom_wheel(event)
            event.accept()
        else:
            super().wheelEvent(event)


class AnnotationMethodWidget(QGroupBox):
    # Emit the selected method as a string, e.g., "Local Maxima" or "Equidistant Spots"
    annotation_method_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Annotation Method", parent)
        layout = QVBoxLayout(self)
        self.combo_box = QComboBox()
        self.combo_box.addItems(["Local Uncertainty Maxima", "Equidistant Spots"])
        self.combo_box.currentTextChanged.connect(self.annotation_method_changed.emit)
        layout.addWidget(self.combo_box)


# ---------------------------------------------------------------------------
# Main ClusteredCropsView Class
# ---------------------------------------------------------------------------
class ClusteredCropsView(QWidget):
    # Aggregate signals from child widgets
    request_clustering = pyqtSignal()
    annotation_method_changed = pyqtSignal(str)
    sample_cluster = pyqtSignal(int)
    sampling_parameters_changed = pyqtSignal(int, int)  # (cluster_id, crops_per_cluster)
    bulk_label_changed = pyqtSignal(int)  # class_id for all visible crops
    crop_label_changed = pyqtSignal(dict, int)  # (annotation_dict, class_id)
    save_project_state_requested = pyqtSignal()
    export_annotations_requested = pyqtSignal()
    save_project_requested = pyqtSignal()
    save_project_as_requested = pyqtSignal()
    load_project_state_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.zoom_level = 5
        self.selected_crops: List[dict] = []

        self._init_ui()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def _init_ui(self):
        # Main layout uses a splitter: left for graphics view, right for controls.
        self.splitter = QSplitter(Qt.Horizontal)

        # Left Panel: Control Panel (wrapped in a scroll area)
        left_panel = self._create_left_panel()
        left_panel.setMinimumWidth(300)

        # Right Panel: Graphics View
        self._create_graphics_view()

        self.splitter.addWidget(self.graphics_view)
        self.splitter.addWidget(left_panel)
        self.splitter.splitterMoved.connect(self.on_splitter_moved)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)
        self.setWindowTitle("Clustered Crops Viewer")
        self.resize(1200, 800)

    def _create_left_panel(self) -> QScrollArea:
        control_panel = QWidget()
        control_panel_layout = QVBoxLayout(control_panel)

        self.annotation_method_widget = AnnotationMethodWidget()
        self.annotation_method_widget.annotation_method_changed.connect(self.annotation_method_changed.emit)
        control_panel_layout.addWidget(self.annotation_method_widget)

        self.clustering_widget = ClusteringControlsWidget()
        self.clustering_widget.request_clustering.connect(self.request_clustering.emit)
        control_panel_layout.addWidget(self.clustering_widget)

        self.navigation_widget = NavigationControlsWidget()
        self.navigation_widget.sample_cluster.connect(self.sample_cluster.emit)
        control_panel_layout.addWidget(self.navigation_widget)

        self.zoom_widget = ZoomControlsWidget(initial_zoom=self.zoom_level)
        self.zoom_widget.zoom_changed.connect(self.on_zoom_changed)
        control_panel_layout.addWidget(self.zoom_widget)

        self.class_labels_widget = ClassLabelsWidget()
        self.class_labels_widget.class_label_selected.connect(self.on_class_button_clicked)
        self.class_labels_widget.agree_with_model.connect(self.on_agree_with_model_clicked)
        control_panel_layout.addWidget(self.class_labels_widget)

        self.file_ops_widget = FileOperationsWidget()
        self.file_ops_widget.load_project_state_requested.connect(self.on_load_project_state)
        self.file_ops_widget.save_project_requested.connect(self.save_project_requested.emit)
        self.file_ops_widget.save_project_as_requested.connect(self.save_project_as_requested.emit)
        self.file_ops_widget.export_annotations_requested.connect(self.on_export_annotations)
        control_panel_layout.addWidget(self.file_ops_widget)

        control_panel_layout.addStretch()

        self.statistics_widget = LabelingStatisticsWidget()
        control_panel_layout.addWidget(self.statistics_widget)

        control_panel_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(control_panel)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        return scroll_area

    def _create_graphics_view(self):
        self.graphics_view = CropGraphicsView(main_view=self)
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)

        self.crop_loading_progress_bar = QProgressBar(self.graphics_view.viewport())
        self.crop_loading_progress_bar.setFixedSize(300, 25)
        self.crop_loading_progress_bar.setAlignment(Qt.AlignCenter)
        self.crop_loading_progress_bar.setVisible(False)

    def handle_zoom_wheel(self, event):
        delta = event.angleDelta().y() / 120
        self.zoom_level = max(MIN_ZOOM, min(self.zoom_level + delta, MAX_ZOOM))
        self.zoom_widget.zoom_slider.setValue(self.zoom_level)
        self.arrange_crops()

    def keyPressEvent(self, event):
        key = event.key()
        if Qt.Key_0 <= key <= Qt.Key_8:
            class_id = key - Qt.Key_0
            if class_id in CLASS_COMPONENTS:
                self.on_class_button_clicked(class_id)
                return
        if key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.on_class_button_clicked(-1)
            return
        if event.text() == '?':
            self.on_class_button_clicked(-2)
            return
        if event.text() == '!':
            self.on_class_button_clicked(-3)
            return
        if key == Qt.Key_Space:
            self.on_agree_with_model_clicked()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.navigation_widget.on_next_cluster()
            return
        if key == Qt.Key_Backspace:
            self.navigation_widget.on_prev_cluster()
            return
        super().keyPressEvent(event)

    def on_zoom_changed(self, value: int):
        self.zoom_level = max(MIN_ZOOM, min(value, MAX_ZOOM))
        logging.debug(f"Zoom level changed to: {self.zoom_level}")
        self.arrange_crops()

    def on_splitter_moved(self, pos, index):
        self.arrange_crops()
        if self.crop_loading_progress_bar.isVisible():
            self._center_crop_loading_progress_bar()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        logging.debug("Window resized. Rearranging crops.")
        self.arrange_crops()
        if self.crop_loading_progress_bar.isVisible():
            self._center_crop_loading_progress_bar()

    def showEvent(self, event):
        super().showEvent(event)
        total_width = self.width()
        control_panel_width = total_width // 3
        graphics_view_width = total_width - control_panel_width
        self.splitter.setSizes([graphics_view_width, control_panel_width])

    def show_clustering_progress_bar(self):
        self.clustering_widget.clustering_progress_bar.setVisible(True)

    def hide_clustering_progress_bar(self):
        self.clustering_widget.clustering_progress_bar.setVisible(False)

    def update_clustering_progress_bar(self, progress: int):
        self.clustering_widget.update_clustering_progress(progress)

    def update_annotation_progress_bar(self, progress: int):
        self.clustering_widget.update_annotation_progress(progress)

    def hide_annotation_progress_bar(self):
        self.clustering_widget.hide_annotation_progress()

    def populate_cluster_selection(self, cluster_info: Dict[int, dict], selected_cluster_id: Optional[int] = None):
        self.navigation_widget.populate_clusters(cluster_info, selected_cluster_id)

    def get_selected_cluster_id(self) -> Optional[int]:
        return self.navigation_widget.get_selected_cluster_id()

    def _center_crop_loading_progress_bar(self):
        viewport_rect = self.graphics_view.viewport().rect()
        x = (viewport_rect.width() - self.crop_loading_progress_bar.width()) // 2
        y = (viewport_rect.height() - self.crop_loading_progress_bar.height()) // 2
        self.crop_loading_progress_bar.move(x, y)

    # --- Progress Bar Methods ---
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

    # --- Displaying Crops ---
    def display_sampled_crops(self, sampled_crops: List[dict]):
        self.selected_crops = sampled_crops
        self.arrange_crops()

    def arrange_crops(self):
        if getattr(self.scene, 'context_menu_open', False):
            return
        self.scene.clear()
        if not self.selected_crops:
            logging.info("No sampled crops to display.")
            return

        viewport_width = self.graphics_view.viewport().width()
        scale_factor = 1.2 ** self.zoom_level
        image_size = BASE_IMAGE_SIZE * scale_factor

        total_spacing = SPACING * 2
        approximate_item_width = image_size + SPACING
        num_columns = max(1, int((viewport_width - total_spacing) // approximate_item_width))

        used_width_for_images = (viewport_width - (num_columns + 1) * SPACING)
        if num_columns > 0:
            image_size = used_width_for_images / num_columns

        logging.debug(f"Calculated image size: {image_size}, columns: {num_columns}")

        x_offset, y_offset = SPACING, SPACING
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

            original_width = pixmap.width()
            if original_width == 0:
                logging.warning("Crop width is zero, skipping image to prevent division by zero.")
                continue
            scale = image_size / original_width
            pixmap_item.setScaleFactor(scale)

            bounding_rect = pixmap_item.boundingRect()
            pixmap_item.setPos(x_offset, y_offset)
            self.scene.addItem(pixmap_item)
            arranged_count += 1

            x_offset += bounding_rect.width() + SPACING
            max_row_height = max(max_row_height, bounding_rect.height())

            if (idx + 1) % num_columns == 0:
                x_offset = SPACING
                y_offset += max_row_height + SPACING
                max_row_height = 0

        expected = len(self.selected_crops)
        assert arranged_count == expected, (
            f"View arrangement mismatch: expected {expected}, arranged {arranged_count}"
        )
        logging.info(f"Arranged {arranged_count} crops in the view.")
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        if self.crop_loading_progress_bar.isVisible():
            self._center_crop_loading_progress_bar()

    # --- Labeling and Statistics ---
    def on_class_button_clicked(self, class_id: Optional[int]):
        if class_id == -3:
            logging.info("Classifying current cluster as artifact.")
        elif class_id == -2:
            logging.info("Classifying current cluster as unsure.")
        elif class_id == -1:
            logging.info("Unlabeling current cluster.")
        else:
            logging.info(f"Assigning class {class_id} to current cluster.")

        self._label_all_visible_crops(class_id)
        self.setFocus()

    def _label_all_visible_crops(self, class_id: Optional[int]):
        if class_id is None:
            class_id = -1
        for crop in self.selected_crops:
            annotation: Annotation = crop['annotation']
            annotation.class_id = class_id
        self.bulk_label_changed.emit(class_id)
        self.arrange_crops()

    def on_agree_with_model_clicked(self):
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
        for cid, cname in CLASS_COMPONENTS.items():
            if cname == model_prediction:
                return cid
        return None

    def update_labeling_statistics(self, statistics: dict):
        self.statistics_widget.update_statistics(statistics)

    # --- File & Project Signals ---
    def on_load_project_state(self):
        self.load_project_state_requested.emit()

    def on_export_annotations(self):
        self.export_annotations_requested.emit()
