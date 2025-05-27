#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
from PyQt5.QtCore import QObject, pyqtSlot, QTimer, QCoreApplication
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from GUI.configuration.configuration import PROJECT_EXT, MIME_FILTER, LEGACY_FILTER
from GUI.controllers.AnnotationClusteringController import AnnotationClusteringController
from GUI.controllers.ImageProcessingController import ImageProcessingController
from GUI.controllers.ProjectStateController import ProjectStateController
from GUI.models import StatePersistance
from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import BaseImageDataModel, create_image_data_model
from GUI.models.PointAnnotationGenerator import (
    LocalMaximaPointAnnotationGenerator,
    EquidistantPointAnnotationGenerator,
    CenterPointAnnotationGenerator,
)
from GUI.models.StatePersistance import ProjectState
from GUI.models.UncertaintyPropagator import propagate_for_annotations
from GUI.models.domain.ClusterSelection import make_selector
from GUI.unittests.debug_uncertainty_propargation import analyze_uncertainty, ProgressFilmer
from GUI.views.ClusteredCropsView import ClusteredCropsView
from GUI.views.ClusteringProgressDialog import ClusteringProgressDialog


class MainController(QObject):
    """
    Main orchestrator of the application, backend-agnostic data model
    """
    AUTOSAVE_IDLE_MS = 10_000
    _UNASSESSED_LABELS = {-1}

    def __init__(
            self,
            model: Optional[BaseImageDataModel],
            view: ClusteredCropsView
    ):
        super().__init__()
        self.feature_matrix = None
        self.image_data_model: Optional[BaseImageDataModel] = model
        self.view = view
        self._nav_history: list[int] = []
        self._current_cluster_id = None
        self.annotation_generator = LocalMaximaPointAnnotationGenerator()
        self._progress_dlg: Optional[ClusteringProgressDialog] = None

        # Controllers use BaseImageDataModel interface
        self.clustering_controller = AnnotationClusteringController(
            self.image_data_model, self.annotation_generator
        )
        self.cluster_selector = make_selector("greedy", self.clustering_controller)
        self.image_processing_controller = ImageProcessingController(self.image_data_model)
        self.project_state_controller = ProjectStateController(self.image_data_model)

        self._dirty = False  # ← any unsaved change?
        self._idle_timer = QTimer(singleShot=True)
        self._idle_timer.timeout.connect(self._autosave_if_dirty)

        self._filmer = ProgressFilmer(
            self.project_state_controller.get_frames_dir
        )

        self.connect_signals()
        QCoreApplication.instance().aboutToQuit.connect(self.cleanup)
        logging.info("MainController initialized.")

    def set_model(self, model: BaseImageDataModel):
        """
        Attach new data model (HDF5 or SQLite) and start autosave.
        """
        if self.image_data_model is model:
            return
        self.image_data_model = model
        self.clustering_controller.model = model
        self.image_processing_controller.model = model
        self.project_state_controller.model = model

        logging.info("Data model set: %s", model.data_path)

    def connect_signals(self):
        # --------------------------- View → Controller ------------------
        self.view.request_clustering.connect(self._begin_generate_annotations)
        self.view.sample_cluster.connect(self.on_select_cluster)
        self.view.annotation_method_changed.connect(self.on_label_generator_method_changed)
        self.view.bulk_label_changed.connect(self.on_bulk_label_changed)
        self.view.crop_label_changed.connect(self.on_crop_label_changed)
        self.view.save_project_state_requested.connect(self.save_project)
        self.view.export_annotations_requested.connect(self.export_annotations)
        self.view.load_project_state_requested.connect(self.load_project)
        self.view.save_project_requested.connect(self.save_project)
        self.view.save_project_as_requested.connect(self.save_project_as)
        self.view.navigation_widget.next_recommended_cluster_requested.connect(self.on_next_recommended_cluster)
        self.view.backtrack_requested.connect(self.on_backtrack_requested)

        # ---------------- ClusteringController → MainController ---------
        # No UI progress bars in the side-panel any more – handled by dialog
        self.clustering_controller.clusters_ready.connect(self.on_clusters_ready)
        self.clustering_controller.labeling_statistics_updated.connect(self.view.update_labeling_statistics)

        # ---------------- ImageProcessingController → View --------------
        self.image_processing_controller.crops_ready.connect(self.view.display_sampled_crops)
        self.image_processing_controller.progress_updated.connect(self.view.update_crop_loading_progress_bar)
        self.image_processing_controller.crop_loading_started.connect(self.view.show_crop_loading_progress_bar)
        self.image_processing_controller.crop_loading_finished.connect(self.view.hide_crop_loading_progress_bar)

        # ---------------- ProjectStateController callbacks --------------
        self.project_state_controller.autosave_finished.connect(self.on_autosave_finished)
        self.project_state_controller.project_loaded.connect(self.on_project_loaded)
        self.project_state_controller.project_saved.connect(self.on_project_saved)
        self.project_state_controller.save_failed.connect(self.on_save_failed)
        self.project_state_controller.load_failed.connect(self.on_load_failed)

    # -------------------------------------------------------------------------
    #                           CLUSTERING HANDLERS
    # -------------------------------------------------------------------------

    @pyqtSlot(str)
    def on_label_generator_method_changed(self, method: str):
        """
        Instantiates the appropriate annotation generator based on the selected method,
        and updates the clustering controller accordingly.
        """
        if method == "Local Maxima":
            self.annotation_generator = LocalMaximaPointAnnotationGenerator(
                filter_size=48, gaussian_sigma=4.0, use_gaussian=False
            )
            self._use_greedy_nav = True
        elif method == "Equidistant Spots":
            self.annotation_generator = EquidistantPointAnnotationGenerator(grid_spacing=48)
            self._use_greedy_nav = True
        elif method == "Image Centre":
            self.annotation_generator = CenterPointAnnotationGenerator()
            self._use_greedy_nav = False
        else:
            self.annotation_generator = LocalMaximaPointAnnotationGenerator(
                filter_size=48, gaussian_sigma=4.0, use_gaussian=False
            )
            self._use_greedy_nav = True
            logging.warning("Unknown annotation method selected; defaulting to Local Maxima.")
        self._seq_cursor = None
        self.clustering_controller.annotation_generator = self.annotation_generator
        logging.info("Annotation generator updated to method: %s", method)

        self._dirty = True
        self._idle_timer.start(self.AUTOSAVE_IDLE_MS)

    @pyqtSlot(dict)
    def on_clusters_ready(self, clusters: Dict[int, List[Annotation]]):
        """
        Handles when clusters are ready after clustering finishes.
        Updates the ImageProcessingController with the new clusters.
        """
        self.image_processing_controller.set_clusters(clusters)
        cluster_info = self.clustering_controller.generate_cluster_info()
        recommended_start_cluster_id = self.cluster_selector.select_next()
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=recommended_start_cluster_id)
        if recommended_start_cluster_id is not None:
            self.on_select_cluster(recommended_start_cluster_id, force=True)

    def _begin_generate_annotations(self):
        if self._progress_dlg:
            return

        self._progress_dlg = ClusteringProgressDialog(self.view)
        dlg = self._progress_dlg

        self.clustering_controller.annotation_progress.connect(
            lambda v: dlg.update_phase("Extracting", v)
        )
        self.clustering_controller.clustering_progress.connect(
            lambda v: dlg.update_phase("Clustering", v)
        )

        self.clustering_controller.clusters_ready.connect(self._finish_progress_dialog)
        self.clustering_controller.cancelled.connect(self._finish_progress_dialog)

        dlg.cancel_requested.connect(self.clustering_controller.cancel_clustering)
        dlg.cancel_requested.connect(self._finish_progress_dialog)

        dlg.show()
        self.clustering_controller.start_clustering()

    def _finish_progress_dialog(self):
        if not self._progress_dlg:
            return

        try:
            self.clustering_controller.annotation_progress.disconnect()
            self.clustering_controller.clustering_progress.disconnect()
            self.clustering_controller.clusters_ready.disconnect()
            self.clustering_controller.cancelled.disconnect()
        except TypeError:
            pass  # already disconnected

        self._progress_dlg.close()
        self._progress_dlg.deleteLater()
        self._progress_dlg = None

    @pyqtSlot(int, bool)
    def on_select_cluster(self, cluster_id: int, force: bool = False):
        """
        Handles the request to sample a cluster and display its crops.

        :param cluster_id: The ID of the cluster to sample.

        Args:
        cluster_id: The ID of the cluster to fetch.
            force: Force fetch.
        """
        if (
                not force
                and getattr(self, "_current_cluster_id", None) is not None
                and cluster_id != self._current_cluster_id
                and not self._visible_crops_complete()
        ):
            cmb = self.view.navigation_widget.cluster_combo
            cmb.blockSignals(True)
            idx = cmb.findData(self._current_cluster_id)

            if idx != -1:
                cmb.setCurrentIndex(idx)
                cmb.blockSignals(False)
                QMessageBox.information(
                    self.view,
                    "Finish labelling first",
                    ("Please label all crops currently in view\n"
                     "before moving to another cluster.")
                )
                return

        clusters = self.clustering_controller.get_clusters()
        annos = clusters.get(cluster_id, [])
        if not annos:
            logging.warning(f"No annotations found in cluster {cluster_id}.")
            self.view.display_sampled_crops([])
            return
        self.image_processing_controller.display_crops(annos)
        self._current_cluster_id = cluster_id

    # -------------------------------------------------------------------------
    #                       LABELING / ANNOTATION HANDLERS
    # -------------------------------------------------------------------------

    @pyqtSlot(int)
    def on_bulk_label_changed(self, class_id: int):
        selected_cluster_id = self.view.get_selected_cluster_id()
        clusters = self.clustering_controller.get_clusters()
        labeled_annotations = clusters.get(selected_cluster_id, [])

        if class_id == -1:
            for anno in labeled_annotations:
                anno.class_id = class_id
                anno.adjusted_uncertainty = anno.uncertainty
        else:
            for anno in labeled_annotations:
                anno.class_id = class_id
                anno.adjusted_uncertainty = 0.0

        # Propagate changes and update statistics
        self.propagate_labeling_changes()
        stats = self.clustering_controller.compute_labeling_statistics()
        self.view.update_labeling_statistics(stats)

        cluster_info = self.clustering_controller.generate_cluster_info()
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=selected_cluster_id)

        self._dirty = True
        self._idle_timer.start(self.AUTOSAVE_IDLE_MS)

    @pyqtSlot(dict, int)
    def on_crop_label_changed(self, crop_data: dict, class_id: int):
        """
        Handles when a class label is set for an individual crop.
        """
        cluster_id = int(crop_data["cluster_id"])
        image_index = int(crop_data["image_index"])
        coord = tuple(crop_data["coord"])

        clusters = self.clustering_controller.get_clusters()
        annotations = clusters.get(cluster_id, [])

        for anno in annotations:
            if anno.image_index == image_index and anno.coord == coord:
                anno.class_id = class_id
                if class_id == -1:  # user chose “Unlabel”
                    anno.is_manual = False
                    anno.reset_uncertainty()  # adjusted ← original prior
                    label_action = "un-labeled"
                else:
                    anno.is_manual = True  # manual lock
                    label_action = "labeled"

                logging.debug(
                    f"Annotation {image_index=} {coord=} {label_action} "
                    f"with class_id {class_id}"
                )
                break
        else:
            logging.warning(
                f"Annotation for image {image_index}, coord {coord} "
                f"not found in cluster {cluster_id}"
            )

        self.propagate_labeling_changes()
        stats = self.clustering_controller.compute_labeling_statistics()
        self.view.update_labeling_statistics(stats)

        cluster_info = self.clustering_controller.generate_cluster_info()
        self.view.populate_cluster_selection(cluster_info,
                                             selected_cluster_id=cluster_id)

        self._dirty = True
        self._idle_timer.start(self.AUTOSAVE_IDLE_MS)

    @pyqtSlot()
    def on_next_recommended_cluster(self):
        if not self._visible_crops_complete():
            QMessageBox.information(
                self.view,
                "Finish labelling first",
                ("Please label all crops currently in view\n"
                 + "before moving to another cluster.")
            )
            return
        current = self.view.get_selected_cluster_id()
        if current is not None:
            self._nav_history.append(current)

        next_cluster_id = self.cluster_selector.select_next(
            current_cluster_id=current
        )

        if next_cluster_id is not None:
            updated_cluster_info = self.clustering_controller.generate_cluster_info()
            self.view.populate_cluster_selection(updated_cluster_info, selected_cluster_id=next_cluster_id)
            self.on_select_cluster(next_cluster_id)
        else:
            logging.info("No next recommended cluster found.")
        self._filmer.maybe_record_frame(self._get_all_annotations())

    @pyqtSlot()
    def on_backtrack_requested(self):
        """Restore the previously displayed recommended cluster, if any."""
        if not self._nav_history:
            logging.debug("Backtrack: history empty; nothing to do.")
            return

        last = self._nav_history.pop()
        cluster_info = self.clustering_controller.generate_cluster_info()
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=last)
        self.on_select_cluster(last, force=True)

    def _get_all_annotations(self) -> List[Annotation]:
        """
        Retrieves all annotations from the current clusters.
        Returns:
            A list of Annotation objects from all clusters.
        """
        clusters = self.clustering_controller.get_clusters()
        return [anno for cluster in clusters.values() for anno in cluster]  # Flatten

    def _update_annotation_uncertainties(self, updated_uncertainties: np.ndarray):
        """
        Updates each annotation's adjusted_uncertainty attribute using updated_uncertainties.
        The annotations are retrieved from the current clusters.
        """
        all_annos = self._get_all_annotations()
        for anno, new_u in zip(all_annos, updated_uncertainties):
            anno.adjusted_uncertainty = new_u

    def set_feature_matrix(self, feature_matrix: np.ndarray):
        """
        Sets the feature matrix used for uncertainty propagation.
        """
        self.feature_matrix = feature_matrix
        logging.info("Feature matrix has been updated in the Main Controller.")

    def propagate_labeling_changes(self) -> None:
        """Re-compute adjusted_uncertainty for every annotation."""
        propagate_for_annotations(self._get_all_annotations())

    def debug_analyze_uncertainty_propagation(
            self,
            *,
            show_plot: bool = True,
            save_plot_to: Optional[str] = None,
    ) -> None:
        """Wrapper that plugs the detached utility into MainController."""
        stats_out = analyze_uncertainty(
            annotations=self._get_all_annotations(),
            show=show_plot,
            save_path=Path(save_plot_to) if save_plot_to else None,
        )
        logging.info("Uncertainty debug summary: %s", stats_out.to_log_string())

    # ---------- soft-lock helpers ------------------------------------

    def _visible_crops_complete(self) -> bool:
        """
        Return True if *every* crop currently displayed in the view
        has a label outside _UNASSESSED_LABELS.
        """
        for crop in self.view.selected_crops:  # ← populated by ClusteredCropsView
            if crop["annotation"].class_id in self._UNASSESSED_LABELS:
                return False
        return True

    # -------------------------------------------------------------------------
    #                           PROJECT STATE
    # -------------------------------------------------------------------------
    def get_current_state(self) -> StatePersistance.ProjectState:
        """Return a fully‑validated ProjectState instance."""
        if self.image_data_model is None:
            # an empty placeholder—won't be saved but keeps type consistent
            return StatePersistance.ProjectState(
                schema_version=2,
                data_backend="",
                data_path="",
                clusters={},
                cluster_order=[],
                selected_cluster_id=None,
            )

        backend = getattr(
            self.image_data_model,
            "backend",
            Path(self.image_data_model.data_path).suffix.lstrip(".").lower(),
        )

        clusters = self.clustering_controller.get_clusters()
        state = StatePersistance.ProjectState(
            schema_version=2,
            data_backend=backend,
            data_path=self.image_data_model.data_path,
            uncertainty="bald",
            clusters={
                str(cid): [a.to_dict() for a in annos]
                for cid, annos in clusters.items()
            },
            cluster_order=list(clusters),
            selected_cluster_id=self.view.get_selected_cluster_id(),
        )
        return state

    def autosave_project_state(self):
        """
        Initiates an autosave operation.
        """
        if self.image_data_model is None:
            logging.debug("Autosave skipped: model is not initialized.")
            return

        logging.debug("Autosave timer triggered.")
        project_state = self.get_current_state()
        self.project_state_controller.autosave_project_state(project_state)

    # -------------------------------------------------------------------------
    #                               SAVING / LOADING
    # -------------------------------------------------------------------------

    @pyqtSlot()
    def save_project(self):
        path = self.project_state_controller.get_current_save_path()
        if not path:
            return self.save_project_as()
        self.project_state_controller.save_project_state(self.get_current_state(), path)

    # -------------------------------------------------------------------
    @pyqtSlot()
    def save_project_as(self) -> None:
        """
        Prompt for a filename and write the current project state.
        Defaults to the modern *.slt* container (Zstandard‑compressed JSON).
        """
        file_path, chosen_filter = QFileDialog.getSaveFileName(
            self.view,
            "Save Project As",
            "",
            f"{MIME_FILTER};;{LEGACY_FILTER};;All Files (*)",
        )
        if not file_path:  # Cancel pressed
            return

        # ----------------------- normalise suffix -----------------------
        suffix = Path(file_path).suffix.lower()

        # user typed no suffix or an unknown one
        if suffix not in {PROJECT_EXT, ".json.gz"}:
            file_path += PROJECT_EXT
        elif suffix == "" and chosen_filter.startswith("Legacy"):
            file_path += ".json.gz"

        # -------------------------- save --------------------------------
        self.project_state_controller.set_current_save_path(file_path)
        self.project_state_controller.save_project_state(
            self.get_current_state(),
            file_path,  # controller decides codec via suffix
        )

    @pyqtSlot()
    def load_project(self) -> None:
        """
        Ask the user for a project file (.slt or legacy .json.gz) and hand it
        to ProjectStateController for asynchronous loading.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self.view,
            "Load Project",
            "",
            f"{MIME_FILTER};;{LEGACY_FILTER};;All Files (*)",
        )
        if not file_path:  # user pressed Cancel
            return

        # remember this path so subsequent Ctrl‑S writes to the same file
        self.project_state_controller.set_current_save_path(file_path)

        # hand off to controller; on success it emits project_loaded, which
        # we already connect to on_project_loaded
        self.project_state_controller.load_project_state(file_path)

    @pyqtSlot()
    def restore_autosave(self):
        """
        Restores the project state from the latest autosave file.
        """
        latest_autosave = self.project_state_controller.get_latest_autosave_file()
        if latest_autosave:
            self.project_state_controller.load_project_state(latest_autosave)
        else:
            QMessageBox.information(self.view, "No Autosave Found", "There are no autosave files to restore.")

    # -------------------------------------------------------------------------
    #                       SAVE / LOAD SLOTS AND CALLBACKS
    # -------------------------------------------------------------------------

    @pyqtSlot(bool)
    def on_autosave_finished(self, success: bool):
        """
        Handles the completion of the autosave operation.

        :param success: True if autosave was successful, False otherwise.
        """
        if success:
            logging.info("Autosave completed successfully.")
        else:
            logging.error("Autosave failed.")

    @pyqtSlot(ProjectState)
    def on_project_loaded(self, project_state: ProjectState):
        """
        Handles loading of the project state.

        :param project_state: The loaded project state.
        """
        self._initialize_model_if_needed(project_state)

        clusters_data = self._clusters_from_state(project_state)
        self.clustering_controller.clusters = clusters_data
        self.image_processing_controller.set_clusters(clusters_data)

        cluster_info = self.clustering_controller.generate_cluster_info()
        selected_cluster_id = project_state.selected_cluster_id
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=selected_cluster_id)

        if selected_cluster_id and selected_cluster_id in clusters_data:
            self.on_select_cluster(selected_cluster_id, force=True)
        else:
            first_cluster_id = next(iter(clusters_data), None)
            if first_cluster_id is not None:
                self.on_select_cluster(first_cluster_id)

        self.view.hide_progress_bar()
        stats = self.clustering_controller.compute_labeling_statistics()
        self.view.update_labeling_statistics(stats)

    @pyqtSlot(str)
    def on_project_saved(self, file_path: str):
        """
        Handles successful saving of the project.

        :param file_path: The file path where the project was saved.
        """
        QMessageBox.information(self.view, "Project Saved", f"Project state saved to {file_path}")

    @pyqtSlot(str)
    def on_save_failed(self, error_message: str):
        """
        Handles failure during saving the project.

        :param error_message: The error message describing the failure.
        """
        QMessageBox.critical(self.view, "Error", f"Failed to save project state: {error_message}")

    @pyqtSlot(str)
    def on_load_failed(self, error_message: str):
        """
        Handles failure during loading the project.

        :param error_message: The error message describing the failure.
        """
        QMessageBox.critical(self.view, "Error", f"Failed to load project state: {error_message}")

    # -------------------------------------------------------------------------
    #                               EXPORT
    # -------------------------------------------------------------------------

    def export_annotations(self):
        """
        Exports the labeled annotations in a final format for downstream use.
        Offers an option to include or exclude artifacts in the export.
        """
        clusters = self.clustering_controller.get_clusters()
        all_annotations = []
        annotations_without_class = False
        artifacts_present = False

        # Collect all annotations
        for cluster_id, annotations in clusters.items():
            for anno in annotations:
                if anno.class_id in [None, -1, -2]:
                    annotations_without_class = True
                elif anno.class_id == -3:
                    artifacts_present = True
                all_annotations.append((cluster_id, anno))

        # Check with user if unlabeled/unsure exist
        if annotations_without_class and not self._confirm_missing_labels():
            logging.info("User canceled exporting due to missing class labels.")
            return

        # Check with user if artifacts exist
        include_artifacts = self._confirm_include_artifacts() if artifacts_present else True

        # Build grouped annotations based on user choices
        grouped_annotations = self._build_grouped_annotations(all_annotations, include_artifacts)
        if not grouped_annotations:
            QMessageBox.warning(self.view, "No Annotations", "No annotations to export given your choices.")
            return

        # Prompt for export file
        export_file = self._prompt_export_file()
        if export_file:
            self._write_annotations_to_file(export_file, grouped_annotations)
        else:
            logging.debug("Export annotations action was canceled.")

    # -------------------------------------------------------------------------
    #                               CLEANUP
    # -------------------------------------------------------------------------

    def cleanup(self):
        """
        Cleans up resources before application exit.
        """
        logging.info("Cleaning up before application exit.")
        self.image_processing_controller.cleanup()
        self.clustering_controller.cleanup()
        self.project_state_controller.cleanup()

    # -------------------------------------------------------------------------
    #                       PRIVATE / HELPER METHODS
    # -------------------------------------------------------------------------

    def _assign_model_to_controllers(self, model: BaseImageDataModel):
        """
        Helper method to reassign the model to all sub-controllers.
        """
        self.image_data_model = model
        self.clustering_controller.model = model
        self.image_processing_controller.model = model
        self.project_state_controller.model = model

    def _initialize_model_if_needed(self, state: ProjectState):
        data_path = state.data_path  # « was  state["data_path"]
        if self.image_data_model and self.image_data_model.data_path == data_path:
            return
        self.set_model(create_image_data_model(state))

    def _clusters_from_state(
            self, state: ProjectState
    ) -> dict[int, list[Annotation]]:
        clusters: dict[int, list[Annotation]] = {
            int(cid): [Annotation.from_dict(a.dict()) for a in annos]
            for cid, annos in state.clusters.items()
        }

        # honour saved display order if present
        if state.cluster_order:
            clusters = self._reorder_clusters(state.cluster_order, clusters)

        return clusters

    @staticmethod
    def _reorder_clusters(cluster_order: List[int],
                          clusters_data: Dict[int, List[Annotation]]) -> OrderedDict:
        """
        Reorders a dictionary of clusters based on a given list of cluster IDs.
        """
        ordered_clusters_data = OrderedDict()
        for cid in cluster_order:
            if cid in clusters_data:
                ordered_clusters_data[cid] = clusters_data[cid]
            else:
                logging.warning(f"Cluster ID {cid} not found in annotations.")

        # Add any clusters that were not in cluster_order
        for cid in clusters_data:
            if cid not in ordered_clusters_data:
                ordered_clusters_data[cid] = clusters_data[cid]
        return ordered_clusters_data

    def _prompt_save_file(self) -> str:
        """
        Prompts the user to choose a location to save a project file.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self.view,
            "Save Project As",
            "",
            "Compressed JSON Files (*.json.gz);;All Files (*)",
            options=options
        )
        if file_path and not file_path.endswith('.json.gz'):
            file_path += '.json.gz'
        return file_path

    def _prompt_load_file(self) -> str:
        """
        Prompts the user to open a project file.
        """
        options = QFileDialog.Options()
        project_file, _ = QFileDialog.getOpenFileName(
            self.view,
            "Open Project",
            "",
            "Compressed JSON Files (*.json.gz);;JSON Files (*.json);;All Files (*)",
            options=options
        )
        return project_file

    def _prompt_export_file(self) -> str:
        """
        Prompts the user to select a file location for exporting annotations.
        """
        options = QFileDialog.Options()
        export_file, _ = QFileDialog.getSaveFileName(
            self.view,
            "Export Annotations",
            "",
            "JSON Files (*.json);;All Files (*)",
            options=options
        )
        return export_file

    @staticmethod
    def _confirm_missing_labels() -> bool:
        """
        Asks the user if they want to proceed if some annotations are unlabeled.
        """
        reply = QMessageBox.question(
            None,
            "Annotations Without Class Labels",
            ("Some annotations do not have class labels assigned. "
             "Do you want to proceed with exporting?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        return reply == QMessageBox.Yes

    @staticmethod
    def _confirm_include_artifacts() -> bool:
        """
        Asks the user if they want to include artifacts in the export.
        """
        reply = QMessageBox.question(
            None,
            "Include Artifacts?",
            "Artifacts have been detected. Do you want to include them in the export?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        return (reply == QMessageBox.Yes)

    @staticmethod
    def _build_grouped_annotations(
            all_annotations: List[tuple],
            include_artifacts: bool
    ) -> Dict[str, List[dict]]:
        """
        Builds the grouped annotations dictionary based on user choices.

        :param all_annotations: List of (cluster_id, Annotation) pairs.
        :param include_artifacts: Whether to include artifacts (class_id == -3).
        :return: A dictionary of filename -> list of annotation dicts.
        """
        grouped = {}
        for cluster_id, anno in all_annotations:
            # Skip unlabeled/unsure
            if anno.class_id is None or anno.class_id in [-1, -2]:
                continue

            # Skip artifacts if not included
            if anno.class_id == -3 and not include_artifacts:
                continue

            annotation_data = {
                'coord': [int(c) for c in anno.coord],
                'class_id': int(anno.class_id),
                'cluster_id': int(cluster_id)
            }
            grouped.setdefault(anno.filename, []).append(annotation_data)
        return grouped

    @staticmethod
    def _write_annotations_to_file(export_file: str, grouped_annotations: dict):
        """
        Writes the grouped annotations to the specified file in JSON format.
        """
        try:
            with open(export_file, 'w') as f:
                json.dump(grouped_annotations, f, indent=4)
            logging.info(f"Annotations exported to {export_file}")
            QMessageBox.information(None, "Export Successful", f"Annotations exported to {export_file}")
        except Exception as e:
            logging.error(f"Error exporting annotations: {e}")
            QMessageBox.critical(None, "Error", f"Failed to export annotations: {e}")

    # ---------------------------------------------------------------------
    #  AUTOSAVE — called by single‑shot timer when user is idle
    # ---------------------------------------------------------------------
    def _autosave_if_dirty(self):
        """Write a project snapshot if anything changed since last save."""
        if not self._dirty:
            return
        self._dirty = False
        project_state = self.get_current_state()
        self.project_state_controller.autosave_project_state(project_state)

    @pyqtSlot(str)
    def on_navigation_policy_changed(self, name: str):
        self.cluster_selector = make_selector(name, self.clustering_controller)
