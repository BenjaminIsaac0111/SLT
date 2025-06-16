from __future__ import annotations

import logging
from functools import partial
from typing import List, Dict, Optional

from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QComboBox, QPushButton, QVBoxLayout, QLabel,
    QProgressBar, QGraphicsView, QGraphicsScene, QGridLayout,
    QGroupBox, QSizePolicy, QSplitter, QGraphicsItem, QApplication,
    QHBoxLayout, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QCheckBox, QDialog, QDialogButtonBox
)

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.annotations import AnnotationBase
from GUI.models.export.ExportService import ExportOptions
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
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Create a container for the progress bar and overlaid label
        progress_container = QWidget()
        progress_container_layout = QGridLayout(progress_container)
        progress_container_layout.setContentsMargins(0, 0, 0, 0)
        progress_container_layout.setSpacing(0)

        # Clustering Progress Bar
        self.clustering_progress_bar = QProgressBar()
        self.clustering_progress_bar.setValue(0)
        self.clustering_progress_bar.setVisible(False)
        # Hide its own text since we will use the overlay label instead.
        self.clustering_progress_bar.setTextVisible(False)
        progress_container_layout.addWidget(self.clustering_progress_bar, 0, 0)

        # Overlay QLabel for displaying text
        self.clustering_progress_label = QLabel("Clustering Annotations")
        self.clustering_progress_label.setAlignment(Qt.AlignCenter)
        self.clustering_progress_label.setStyleSheet("background: transparent; color: black;")
        self.clustering_progress_label.setVisible(False)
        progress_container_layout.addWidget(self.clustering_progress_label, 0, 0)

        layout.addWidget(progress_container)

        # Annotation Progress Bar (unchanged)
        self.annotation_progress_bar = QProgressBar()
        self.annotation_progress_bar.setValue(0)
        self.annotation_progress_bar.setVisible(False)
        self.annotation_progress_bar.setFormat("Extracting Annotations: %p%")
        layout.addWidget(self.annotation_progress_bar)

        self.setLayout(layout)

    def update_clustering_progress(self, progress: int):
        self.clustering_progress_bar.setVisible(True)
        self.clustering_progress_label.setVisible(True)
        if progress == -1:
            # Use a determinate range but indicate waiting
            self.clustering_progress_bar.setRange(0, 0)
            self.clustering_progress_label.setText("Clustering annotations, please wait...")
        else:
            self.clustering_progress_bar.setRange(0, 100)
            self.clustering_progress_bar.setValue(progress)
            self.clustering_progress_label.setText(f"Clustering annotations: {progress}%")

    def hide_clustering_progress(self):
        self.clustering_progress_bar.setVisible(False)
        self.clustering_progress_label.setVisible(False)

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
    sample_cluster = pyqtSignal(int, bool)
    next_recommended_cluster_requested = pyqtSignal()

    def __init__(self):
        super().__init__("Navigation")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # ---------------- cluster chooser ----------------
        self.cluster_combo = QComboBox()
        self.cluster_combo.currentIndexChanged.connect(self.on_cluster_selected)
        layout.addWidget(self.cluster_combo)

        # ---------------- next-recommended ----------------
        self.next_recommended_button = QPushButton("Go to next recommended (⏎)")
        self.next_recommended_button.clicked.connect(self.on_next_recommended)
        self.next_recommended_button.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.next_recommended_button)

        # ---------------- workflow prefs -----------------
        self.auto_advance_checkbox = QCheckBox("Auto-advance after Agree")
        self.auto_advance_checkbox.setChecked(True)
        layout.addWidget(self.auto_advance_checkbox)


        self.cluster_combo.setFocusPolicy(Qt.NoFocus)
        self.next_recommended_button.setFocusPolicy(Qt.NoFocus)
        self.auto_advance_checkbox.setFocusPolicy(Qt.NoFocus)

        self.setLayout(layout)

        # Default navigation policy is greedy
        self.set_navigation_policy("greedy")

    def on_next_recommended(self):
        self.next_recommended_cluster_requested.emit()

    def set_navigation_policy(self, policy: str) -> None:
        """Update the button text to reflect the selected navigation policy."""
        mapping = {
            "greedy": "Go to next recommended (⏎)",
            "sequential": "Go to next sequential (⏎)",
            "random": "Go to next random (⏎)",
        }
        self.next_recommended_button.setText(mapping.get(policy, "Go to next (⏎)"))

    def on_cluster_selected(self, index: int):
        cluster_id = self.cluster_combo.itemData(index)
        if cluster_id is not None:
            logging.debug(f"Cluster selected: {cluster_id}")
            self.sample_cluster.emit(cluster_id, True)

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
            avg_adj_uncertainty = info.get('average_adjusted_uncertainty', 0.0)

            cluster_label = info.get('label', '')
            text = (
                f"Cluster {cid} - {num_annotations} annotations, "
                f"{labeled_percentage:.2f}% assessed, "
                f"Uncertainty: {avg_uncertainty:.8f} "
                f"Adj Uncertainty: {avg_adj_uncertainty:.8f}"
            )
            if cluster_label:
                text += f" - label: {cluster_label}"
            self.cluster_combo.addItem(text, cid)

        if selected_cluster_id is not None:
            index = self.cluster_combo.findData(selected_cluster_id)
            if index != -1:
                self.cluster_combo.setCurrentIndex(index)

        self.cluster_combo.blockSignals(False)

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

        unsure_button = QPushButton("Unsure (U)")
        unsure_button.clicked.connect(partial(self.class_label_selected.emit, -2))
        unsure_button.setFocusPolicy(Qt.NoFocus)
        unsure_button.setStyleSheet("background-color: #ffa500;")
        special_actions_layout.addWidget(unsure_button)

        artifact_button = QPushButton("Artifact (A)")
        artifact_button.clicked.connect(partial(self.class_label_selected.emit, -3))
        artifact_button.setFocusPolicy(Qt.NoFocus)
        artifact_button.setStyleSheet("background-color: #d3d3d3;")
        special_actions_layout.addWidget(artifact_button)

        layout.addLayout(special_actions_layout, 0, 0, 1, 3)

        # Class Buttons Grid
        row, col = 1, 0
        for class_id, class_name in CLASS_COMPONENTS.items():
            button_text = f"{class_name.upper()} ({class_id})"
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
            "U = Unsure\n"
            "A = Artifact\n"
            "Spacebar = Agree with Model Predictions"
        )
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label, row + 2, 0, 1, 3)

        self.setLayout(layout)


# ---------------------------------------------------------------------------
# Labeling Statistics Widget
# ---------------------------------------------------------------------------
class LabelingStatisticsWidget(QGroupBox):
    def __init__(self):
        super().__init__("Labeling Statistics")
        self._init_ui()
        # Make the widget expand to use available space.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _init_ui(self):
        layout = QVBoxLayout()
        self.table = QTableWidget()

        # ⇩⇩ now five columns
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Class (ID)", "Count", "Weight", "Mean Unc."]
        )

        self.table.setRowCount(len(CLASS_COMPONENTS))
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(self.table)

        self.summary_label = QLabel("")
        self.summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.summary_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.summary_label)
        self.setLayout(layout)

    def update_statistics_table(self, statistics: dict):
        primary_ids = sorted(CLASS_COMPONENTS.keys())
        for row, cid in enumerate(primary_ids):
            class_name = CLASS_COMPONENTS[cid]
            count = statistics["class_counts"].get(cid, 0)
            weight = statistics["class_weights"].get(cid)
            mu = statistics["class_mean_uncertainty"].get(cid)

            weight_str = f"{weight:.2f}" if weight is not None else "N/A"
            mu_str = f"{mu:.3f}" if mu is not None else "N/A"

            self.table.setItem(row, 0, QTableWidgetItem(f"{class_name} ({cid})"))
            self.table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.table.setItem(row, 2, QTableWidgetItem(weight_str))
            self.table.setItem(row, 3, QTableWidgetItem(mu_str))

        # Format summary for special classes and overall counts.
        special_counts = []
        for special_id, label in [(-1, "Unlabeled"), (-2, "Unsure"), (-3, "Artifact")]:
            cnt = statistics.get('class_counts', {}).get(special_id, 0)
            special_counts.append(f"{label}: {cnt}")
        total_annotations = statistics.get('total_annotations', 0)
        total_labeled = statistics.get('total_labeled', 0)
        disagreement_count = statistics.get('disagreement_count', 0)
        agreement_percentage = statistics.get('agreement_percentage', 0.0)

        summary_text = (
                f"Total Annotations: {total_annotations} | Total Labeled: {total_labeled}\n" +
                ", ".join(special_counts) + "\n" +
                f"Disagreements: {disagreement_count} (Agreement: {agreement_percentage:.2f}%)"
        )
        self.summary_label.setText(summary_text)


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


# ---------------------------------------------------------------------------
# Main ClusteredCropsView Class
# ---------------------------------------------------------------------------
class ClusteredCropsView(QWidget):
    request_clustering = pyqtSignal()
    annotation_method_changed = pyqtSignal(str)
    sample_cluster = pyqtSignal(int, bool)
    backtrack_requested = pyqtSignal()
    bulk_label_changed = pyqtSignal(int)
    crop_label_changed = pyqtSignal(dict, int)
    save_project_state_requested = pyqtSignal()
    export_annotations_requested = pyqtSignal(str)
    save_project_requested = pyqtSignal()
    save_project_as_requested = pyqtSignal(str)
    load_project_state_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.zoom_level = 5
        self.selected_crops: List[dict] = []
        self._nav_history: List[int] = []

        self._init_ui()
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def _init_ui(self):
        # Main layout uses a splitter: left for graphics view, right for controls.
        self.splitter = QSplitter(Qt.Horizontal)

        # ---- left: graphics view
        self._create_graphics_view()

        # ---- right: control panel
        right_panel = self._create_control_panel()
        right_panel.setMinimumWidth(300)

        self.splitter.addWidget(self.graphics_view)
        self.splitter.addWidget(right_panel)
        self.splitter.splitterMoved.connect(self.on_splitter_moved)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)
        self.setWindowTitle("Clustered Crops Viewer")
        self.resize(1200, 800)

    def _create_control_panel(self) -> QScrollArea:
        control_panel = QWidget()
        control_panel_layout = QVBoxLayout(control_panel)

        self.data_path_label = QLabel("")
        self.data_path_label.setWordWrap(True)
        control_panel_layout.addWidget(self.data_path_label)

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

        self.statistics_widget = LabelingStatisticsWidget()
        control_panel_layout.addWidget(self.statistics_widget, stretch=1)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(control_panel)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        return scroll_area

    def _create_graphics_view(self):
        self.graphics_view = CropGraphicsView(main_view=self)
        self.scene = QGraphicsScene(self)
        self.overlays_visible = True
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
        mods = event.modifiers()

        if Qt.Key_0 <= key <= Qt.Key_8:
            class_id = key - Qt.Key_0
            if class_id in CLASS_COMPONENTS:
                self.on_class_button_clicked(class_id)
                return

        if key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.on_class_button_clicked(-1)
            return

        if event.key() == Qt.Key_U:
            self.on_class_button_clicked(-2)
            return

        if event.key() == Qt.Key_A:
            self.on_class_button_clicked(-3)
            return

        # Force-agree  (Ctrl + Space)
        if key == Qt.Key_Space and (mods & Qt.ControlModifier):
            self.on_agree_with_model_clicked(force=True)
            return

        # Normal agree – *skip* locked crops
        if key == Qt.Key_Space:
            self.on_agree_with_model_clicked(force=False)
            return

        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.navigation_widget.on_next_recommended()
            return

        if key == Qt.Key_Backspace:
            self.backtrack_requested.emit()
            return

        if event.key() == Qt.Key_H:
            self.scene.overlays_visible = not getattr(self.scene,
                                                      "overlays_visible", True)
            self.scene.invalidate()
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
        QTimer.singleShot(0, self._set_splitter_sizes)

    def _set_splitter_sizes(self):
        total_width = self.width()
        control_panel_width = total_width // 3
        graphics_view_width = total_width - control_panel_width
        self.splitter.setSizes([graphics_view_width, control_panel_width])

    def show_clustering_progress_bar(self):
        self.clustering_widget.clustering_progress_bar.setVisible(True)

    def hide_clustering_progress_bar(self):
        self.clustering_widget.clustering_progress_bar.setVisible(False)
        self.clustering_widget.clustering_progress_label.setVisible(False)

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
            annotation: AnnotationBase = crop_data['annotation']
            pixmap: QPixmap = crop_data['processed_crop']

            if pixmap.isNull():
                logging.warning(f"Invalid QPixmap for image index {annotation.image_index}. Skipping.")
                continue

            pixmap_item = ClickablePixmapItem(annotation=annotation, pixmap=pixmap, coord_pos=crop_data['coord_pos'])
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
        if arranged_count != expected:
            logging.warning(
                "Arranged %d/%d crops—skipping assert to maintain UI stability.",
                arranged_count, expected
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
            ann: AnnotationBase = crop['annotation']
            ann.class_id = class_id
            ann.is_manual = (class_id != -1)  # mark only real labels as manual
            if class_id == -1:  # unlabel: restore
                ann.reset_uncertainty()
                ann.is_manual = False
        self.bulk_label_changed.emit(class_id)
        self.arrange_crops()

    def on_agree_with_model_clicked(self, *, force: bool = False):
        changes = False
        for crop in self.selected_crops:
            ann = crop["annotation"]
            if ann.is_manual and not force:
                continue
            cid = self.get_class_id_from_model_prediction(ann.model_prediction)
            if cid is not None and cid != ann.class_id:
                ann.class_id = cid
                ann.is_manual = False
                self.crop_label_changed.emit(ann.to_dict(), cid)
                changes = True

        if changes:
            self.arrange_crops()

        if self.navigation_widget.auto_advance_checkbox.isChecked():
            QTimer.singleShot(300, self.navigation_widget.on_next_recommended)

    @staticmethod
    def get_class_id_from_model_prediction(model_prediction: str) -> Optional[int]:
        for cid, cname in CLASS_COMPONENTS.items():
            if cname == model_prediction:
                return cid
        return None

    def set_recommended_cluster(self, cluster_id: int):
        self.navigation_widget.next_recommended_button.setText(f"Go to Recommended Cluster (ID: {cluster_id})")

    def update_labeling_statistics(self, statistics: dict):
        self.statistics_widget.update_statistics_table(statistics)

    # ------------------------------------------------------------------
    #  Display helpers
    # ------------------------------------------------------------------
    def set_data_path(self, path: str) -> None:
        """Update the label showing the active HDF5/SQLite file."""
        self.data_path_label.setText(f"Data file: {path}")

    # --- File & Project Signals ---
    def on_load_project_state(self):
        self.load_project_state_requested.emit()

    def on_export_annotations(self):
        self.export_annotations_requested.emit()

    def ask_export_options(
            self,
            flat_annos: list[tuple[int, AnnotationBase]],
    ) -> ExportOptions | None:
        """Return the user’s choice or *None* on cancel."""
        unlabeled = sum(
            1 for _, a in flat_annos if a.class_id in {None, -1, -2}
        )
        artefacts = sum(1 for _, a in flat_annos if a.class_id == -3)

        # optimisation: skip the dialog if nothing to ask
        if unlabeled == 0 and artefacts == 0:
            return ExportOptions(include_artifacts=True)

        dlg = ExportAnnotationsDialog(self, unlabeled, artefacts)
        return dlg.get_options()


class ExportAnnotationsDialog(QDialog):
    """
    Modal dialog that asks two questions:

        • Proceed despite un-labelled crops?
        • Include artefacts (class_id == -3) ?

    If the user clicks *Cancel* the caller gets `None`.
    Otherwise, an `ExportOptions` instance is returned.
    """

    def __init__(
            self,
            parent: QWidget | None,
            unlabeled_count: int,
            artifact_count: int,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export annotations")

        lay = QVBoxLayout(self)

        # --- Un-labelled warning -----------------------------------
        if unlabeled_count:
            label = QLabel(
                f"{unlabeled_count} annotations do not have a class label.\n"
                "Do you still want to continue with the export?"
            )
            label.setWordWrap(True)
            lay.addWidget(label)

        # --- Artefacts checkbox -------------------------------------
        self._ck_artifacts = QCheckBox(
            f"Include {artifact_count} artefact annotations"
        )
        if artifact_count:
            self._ck_artifacts.setChecked(True)
            lay.addWidget(self._ck_artifacts)
        else:
            self._ck_artifacts.hide()

        # --- OK / Cancel buttons ------------------------------------
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            orientation=Qt.Horizontal,
            parent=self,
        )
        lay.addWidget(btns)

        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    # ----------------------------------------------------------------
    #  Public helper --------------------------------------------------
    # ----------------------------------------------------------------
    def get_options(self) -> ExportOptions | None:
        if self.exec_() != QDialog.Accepted:
            return None
        return ExportOptions(include_artifacts=self._ck_artifacts.isChecked())
