#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import scipy
from PyQt5.QtCore import QObject, pyqtSlot, QTimer, QCoreApplication
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from GUI.configuration.configuration import PROJECT_EXT, MIME_FILTER, LEGACY_FILTER
from GUI.controllers.AnnotationClusteringController import AnnotationClusteringController
from GUI.controllers.ImageProcessingController import ImageProcessingController
from GUI.controllers.ProjectStateController import ProjectStateController
from GUI.models import StatePersistance
from GUI.models.Annotation import Annotation
# Use unified interface and factory
from GUI.models.ImageDataModel import BaseImageDataModel, create_image_data_model
from GUI.models.PointAnnotationGenerator import (
    LocalMaximaPointAnnotationGenerator,
    EquidistantPointAnnotationGenerator, CenterPointAnnotationGenerator,
)
from GUI.models.StatePersistance import ProjectState
from GUI.models.UncertaintyPropagator import propagate_for_annotations
from GUI.unittests.debug_uncertainty_propargation import analyze_uncertainty, ProgressFilmer
from GUI.views.ClusteredCropsView import ClusteredCropsView


class MainController(QObject):
    """
    Main orchestrator of the application, backend-agnostic data model
    """
    AUTOSAVE_IDLE_MS = 10_000

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
        self.annotation_generator = LocalMaximaPointAnnotationGenerator()
        self._use_greedy_nav: bool = True  # updated in on_label_generator_method_changed
        self._seq_cursor: Optional[int] = None  # stores the “current” cluster id for sequential mode

        # Controllers use BaseImageDataModel interface
        self.clustering_controller = AnnotationClusteringController(
            self.image_data_model, self.annotation_generator
        )
        self.image_processing_controller = ImageProcessingController(self.image_data_model)
        self.project_state_controller = ProjectStateController(self.image_data_model)

        self._dirty = False  # ← any unsaved change?
        self._idle_timer = QTimer(singleShot=True)
        self._idle_timer.timeout.connect(self._autosave_if_dirty)

        self._filmer = ProgressFilmer(
            self.project_state_controller.get_frames_dir()
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
        """
        Connects signals from the view and other controllers to the appropriate methods.
        """
        # --- View -> Controller ---
        self.view.request_clustering.connect(self.clustering_controller.start_clustering)
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
        # --- ClusteringController -> View ---
        self.clustering_controller.show_clustering_progress_bar.connect(self.view.show_clustering_progress_bar)
        self.clustering_controller.hide_clustering_progress_bar.connect(self.view.hide_clustering_progress_bar)
        self.clustering_controller.clustering_progress.connect(self.view.update_clustering_progress_bar)
        self.clustering_controller.annotation_progress.connect(self.view.update_annotation_progress_bar)
        self.clustering_controller.annotation_progress_finished.connect(self.view.hide_annotation_progress_bar)
        self.clustering_controller.clusters_ready.connect(self.on_clusters_ready)
        self.clustering_controller.labeling_statistics_updated.connect(self.view.update_labeling_statistics)

        # --- ImageProcessingController -> View ---
        self.image_processing_controller.crops_ready.connect(self.view.display_sampled_crops)
        self.image_processing_controller.progress_updated.connect(self.view.update_crop_loading_progress_bar)
        self.image_processing_controller.crop_loading_started.connect(self.view.show_crop_loading_progress_bar)
        self.image_processing_controller.crop_loading_finished.connect(self.view.hide_crop_loading_progress_bar)

        # --- ProjectStateController -> Slots ---
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
        first_cluster_id = list(cluster_info.keys())[0] if cluster_info else None
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=first_cluster_id)
        if first_cluster_id is not None:
            self.on_select_cluster(first_cluster_id)
        self.view.hide_clustering_progress_bar()

    @pyqtSlot(int)
    def on_select_cluster(self, cluster_id: int):
        """
        Handles the request to sample a cluster and display its crops.

        :param cluster_id: The ID of the cluster to sample.
        """
        clusters = self.clustering_controller.get_clusters()
        annotations = clusters.get(cluster_id, [])
        if not annotations:
            logging.warning(f"No annotations found in cluster {cluster_id}.")
            self.view.display_sampled_crops([])
            return
        self.image_processing_controller.display_crops(annotations, cluster_id)

    def handle_sampling_parameters_changed(self):
        """
        Handles the sampling parameters after debouncing.
        Refreshes the displayed crops based on the updated parameters.
        """
        logging.info("Handling debounced sampling parameter changes.")
        selected_cluster_id = self.view.get_selected_cluster_id()
        if selected_cluster_id is not None:
            self.on_select_cluster(selected_cluster_id)
        else:
            logging.warning("No cluster is currently selected. Clearing displayed crops.")
            self.view.display_sampled_crops([])

    # -------------------------------------------------------------------------
    #                       LABELING / ANNOTATION HANDLERS
    # -------------------------------------------------------------------------

    @pyqtSlot(int)
    def on_bulk_label_changed(self, class_id: int):
        selected_cluster_id = self.view.get_selected_cluster_id()
        clusters = self.clustering_controller.get_clusters()
        labeled_annotations = clusters.get(selected_cluster_id, [])

        if class_id == -1:
            for anno in labeled_annotations[:self.image_processing_controller.crops_per_cluster]:
                anno.class_id = class_id
                anno.adjusted_uncertainty = anno.uncertainty
        else:
            for anno in labeled_annotations[:self.image_processing_controller.crops_per_cluster]:
                anno.class_id = class_id
                anno.adjusted_uncertainty = 0.0

        # Propagate changes and update statistics
        self.propagate_labeling_changes()
        self.clustering_controller.compute_labeling_statistics()

        cluster_info = self.clustering_controller.generate_cluster_info()
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=selected_cluster_id)

        self.autosave_project_state()

    @pyqtSlot(dict, int)
    def on_crop_label_changed(self, crop_data: dict, class_id: int):
        """
        Handles when a class label is set for an individual crop.
        """
        cluster_id = int(crop_data['cluster_id'])
        image_index = int(crop_data['image_index'])
        coord = tuple(crop_data['coord'])

        clusters = self.clustering_controller.get_clusters()
        annotations = clusters.get(cluster_id, [])

        # Update the annotation matching the provided crop data.
        for anno in annotations:
            if anno.image_index == image_index and anno.coord == coord:
                anno.class_id = class_id
                label_action = "unlabeled" if class_id == -1 else "labeled"
                logging.debug(
                    f"Annotation for image {image_index}, coord {coord} {label_action} with class_id {class_id}"
                )
                break
        else:
            logging.warning(f"Annotation for image {image_index}, coord {coord} not found in cluster {cluster_id}")

        # Propagate uncertainty changes
        self.propagate_labeling_changes()

        # Recompute labeling statistics
        self.clustering_controller.compute_labeling_statistics()

        cluster_info = self.clustering_controller.generate_cluster_info()
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=cluster_id)

        self._dirty = True
        self._idle_timer.start(self.AUTOSAVE_IDLE_MS)

    @pyqtSlot()
    def on_next_recommended_cluster(self):
        current = self.view.get_selected_cluster_id()
        if current is not None:
            self._nav_history.append(current)
        if self._use_greedy_nav:
            next_cluster_id = self.select_next_cluster_based_on_greedy_metric()
        else:
            next_cluster_id = self._select_next_cluster_sequential(current)

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
        self.on_select_cluster(last)

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

    def select_next_cluster_based_on_greedy_metric(
            self, *, skip_current: bool = True
    ) -> Optional[int]:
        """
        Return the cluster ID with the highest composite z-score *excluding*
        (1) the cluster currently shown in the UI   and
        (2) clusters that are fully labelled.

        Parameters
        ----------
        skip_current : bool, default True
            If True, the cluster that is selected in the GUI at call-time will be
            omitted from consideration.  This prevents immediately revisiting the
            same cluster after the user has just labelled it.
        """
        clusters = self.clustering_controller.get_clusters()
        if not clusters:
            return None

        # -------------------------------------------------- #
        #  Identify cluster(s) to skip
        # -------------------------------------------------- #
        current_cid = self.view.get_selected_cluster_id() if skip_current else None

        # Remove clusters that are fully labelled
        def _has_unlabelled(annos):
            return any(a.class_id == -1 for a in annos)

        candidate_items = {
            cid: annos
            for cid, annos in clusters.items()
            if _has_unlabelled(annos) and cid != current_cid
        }
        if not candidate_items:
            logging.info("All remaining clusters are fully labelled or only current cluster left.")
            return None

        # -------------------------------------------------- #
        #  Global label stats for rarity & coverage terms
        # -------------------------------------------------- #
        labeled_annos = [
            a for annos in clusters.values()
            for a in annos if a.class_id not in (-1, -2, -3)
        ]
        label_counts = Counter(a.class_id for a in labeled_annos) or {0: 1}

        kd = None
        if labeled_annos:
            feats_lab = np.vstack([a.logit_features for a in labeled_annos])
            kd = scipy.spatial.cKDTree(feats_lab)

        # -------------------------------------------------- #
        #  Per-cluster raw metrics
        # -------------------------------------------------- #
        per_cluster = []  # (cid, U, D, R, C)

        for cid, annos in candidate_items.items():
            unlab = [a for a in annos if a.class_id == -1]
            lab = [a for a in annos if a.class_id not in (-1, -2, -3)]

            # mean adjusted uncertainty of *unlabelled* members
            U = float(np.mean([a.adjusted_uncertainty for a in unlab])) if unlab else 0.0

            # disagreement rate among labelled members
            if lab:
                disagreements = [
                    1
                    for a in lab
                    if (mp := self.clustering_controller.get_class_id_from_prediction(a.model_prediction)) is not None
                       and mp != a.class_id
                ]
                D = sum(disagreements) / len(lab)
            else:
                D = 0.0

            # rarity of majority class in the cluster
            if lab:
                maj_cls = Counter(a.class_id for a in lab).most_common(1)[0][0]
            else:  # no labels yet in this cluster → use first unlabelled prediction
                maj_cls = self.clustering_controller.get_class_id_from_prediction(unlab[0].model_prediction)
            R = 1.0 / (label_counts.get(maj_cls, 0) + 1)

            # distance to nearest labelled sample (coverage)
            if kd is not None:
                centre = np.mean([a.logit_features for a in annos], axis=0)
                C = kd.query(centre)[0]
            else:
                C = 0.0

            per_cluster.append((cid, U, D, R, C))

        # -------------------------------------------------- #
        #  z-score normalisation & selection
        # -------------------------------------------------- #
        mat = np.asarray([t[1:] for t in per_cluster], dtype=np.float32)  # shape (m,4)
        means = mat.mean(axis=0)
        stds = mat.std(axis=0)
        stds[stds == 0] = 1.0  # avoid divide-by-zero

        best_cid, best_score = None, -np.inf
        for cid, U, D, R, C in per_cluster:
            z = ((U - means[0]) / stds[0] +
                 (D - means[1]) / stds[1] +
                 (R - means[2]) / stds[2] +
                 (C - means[3]) / stds[3])
            if z > best_score:
                best_score, best_cid = z, cid

        logging.info(
            "Next recommended cluster (z-score rule, skipping %s): %s  [score %.3f]",
            current_cid, best_cid, best_score,
        )
        return best_cid

    def _select_next_cluster_sequential(self, current: Optional[int]) -> Optional[int]:
        """
        Returns the next cluster id in ascending order, wrapping around once.
        Keeps an internal cursor so repeated presses iterate deterministically.
        """
        cluster_ids = sorted(self.clustering_controller.get_clusters())
        if not cluster_ids:
            return None

        # Initialise cursor
        if self._seq_cursor is None or self._seq_cursor not in cluster_ids:
            self._seq_cursor = cluster_ids[0]
        else:
            # advance to next id
            idx = cluster_ids.index(self._seq_cursor)
            self._seq_cursor = cluster_ids[(idx + 1) % len(cluster_ids)]

        # Do not return the same cluster twice in a row
        if self._seq_cursor == current and len(cluster_ids) > 1:
            idx = cluster_ids.index(self._seq_cursor)
            self._seq_cursor = cluster_ids[(idx + 1) % len(cluster_ids)]

        return self._seq_cursor

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
            self.on_select_cluster(selected_cluster_id)
        else:
            first_cluster_id = next(iter(clusters_data), None)
            if first_cluster_id is not None:
                self.on_select_cluster(first_cluster_id)

        self.view.hide_progress_bar()
        self.clustering_controller.compute_labeling_statistics()

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
