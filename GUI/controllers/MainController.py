#!/usr/bin/env python3
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, List

from PyQt5.QtCore import QObject, pyqtSlot, QTimer, QCoreApplication, QThreadPool
from PyQt5.QtWidgets import QMessageBox

from GUI.controllers.AnnotationClusteringController import AnnotationClusteringController
from GUI.controllers.ImageProcessingController import ImageProcessingController
from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import BaseImageDataModel, create_image_data_model
from GUI.models.PointAnnotationGenerator import (
    LocalMaximaPointAnnotationGenerator,
    EquidistantPointAnnotationGenerator,
    CenterPointAnnotationGenerator,
)
from GUI.models.UncertaintyPropagator import propagate_for_annotations
from GUI.models.export.Options import ExportOptions
from GUI.models.export.Usecase import ExportAnnotationsUseCase
from GUI.models.io.IOService import ProjectIOService
from GUI.models.io.Persistence import ProjectState
from GUI.models.navigation.ClusterSelection import make_selector
from GUI.views.ClusteredCropsView import ClusteredCropsView
from GUI.views.ClusteringProgressDialog import ClusteringProgressDialog
from GUI.views.AnnotationPreviewDialog import AnnotationPreviewDialog
from GUI.workers.CrossValidationWorker import CrossValidationWorker


class MainController(QObject):
    """Top‑level coordinator between GUI view and processing back‑end."""

    AUTOSAVE_IDLE_MS = 10_000
    _UNASSESSED_LABELS = {-1}

    def __init__(self, model: Optional[BaseImageDataModel], view: ClusteredCropsView):
        super().__init__()
        self.image_data_model = model
        self.view = view

        # ---------- core sub‑controllers ---------------------------------
        self.annotation_generator = LocalMaximaPointAnnotationGenerator()
        self.clustering_controller = AnnotationClusteringController(model, self.annotation_generator)
        self.image_processing_controller = ImageProcessingController(model)
        self.io = ProjectIOService(data_anchor=Path(model.data_path) if model else None)
        self._export_usecase = ExportAnnotationsUseCase()

        self.cluster_selector = make_selector("greedy", self.clustering_controller)

        # ---------- GUI state -------------------------------------------
        self._progress_dlg: Optional[ClusteringProgressDialog] = None
        self._nav_history: list[int] = []
        self._current_cluster_id: Optional[int] = None
        self._current_ann_method = "Local Uncertainty Maxima"

        # ---------- dirty‑flag & idle‑flush -----------------------------
        self._dirty = False
        self._idle_timer = QTimer(singleShot=True)
        self._idle_timer.timeout.connect(self._autosave_if_dirty)

        self._connect_signals()
        QCoreApplication.instance().aboutToQuit.connect(self.cleanup)
        logging.info("MainController initialised.")

    # ==================================================================
    #  Signal wiring
    # ==================================================================
    def _connect_signals(self):
        # -------- view → controller -----------------------------------
        v = self.view
        v.request_clustering.connect(self._begin_generate_annotations)
        v.sample_cluster.connect(self.on_select_cluster)
        v.annotation_method_changed.connect(self.on_label_generator_method_changed)
        v.bulk_label_changed.connect(self.on_bulk_label_changed)
        v.crop_label_changed.connect(self.on_crop_label_changed)
        v.navigation_widget.next_recommended_cluster_requested.connect(self.on_next_cluster)
        v.backtrack_requested.connect(self.on_backtrack_requested)

        # these now come from AppMenuBar (connected in app_entry)
        v.save_project_requested.connect(self.save_project)  # quick save
        v.save_project_as_requested.connect(self.save_project_as)  # path:str
        v.load_project_state_requested.connect(self.load_project)  # path:str

        # -------- clustering → controller -----------------------------
        self.clustering_controller.clusters_ready.connect(self.on_clusters_ready)
        self.clustering_controller.labeling_statistics_updated.connect(v.update_labeling_statistics)

        # -------- image‑processing → view -----------------------------
        ip = self.image_processing_controller
        ip.crops_ready.connect(v.display_sampled_crops)
        ip.progress_updated.connect(v.update_crop_loading_progress_bar)
        ip.crop_loading_started.connect(v.show_crop_loading_progress_bar)
        ip.crop_loading_finished.connect(v.hide_crop_loading_progress_bar)

        # -------- I/O service callbacks -------------------------------
        self.io.project_saved.connect(self._on_project_saved)
        self.io.project_loaded.connect(self._on_project_loaded)
        self.io.save_failed.connect(self._on_save_failed)
        self.io.load_failed.connect(self._on_load_failed)
        self.io.autosave_finished.connect(self.on_autosave_finished)

    # ==================================================================
    #  Persistence API (slots expected by AppMenuBar)
    # ==================================================================
    @pyqtSlot()
    def save_project(self):
        if self.io.current_path is None:
            # View should open Save‑As dialog instead; ignore quietly.
            return
        self.io.save_async(self.get_current_state(), self.io.current_path)

    @pyqtSlot(str)
    def save_project_as(self, path: str):
        self.io.set_current_path(path)
        self.io.save_async(self.get_current_state(), path)

    @pyqtSlot(str)
    def load_project(self, path: str):
        self.io.load_async(path)

    # restore from autosave ---------------------------------------------------
    def restore_autosave(self):
        latest = self.io.latest_autosave()
        if latest:
            self.io.load_async(str(latest))
        else:
            QMessageBox.information(self.view, "No Autosave Found", "There are no autosave files to restore.")

    # ==================================================================
    #  Autosave idle‑timer
    # ==================================================================
    def _autosave_if_dirty(self):
        if not self._dirty:
            return
        self._dirty = False
        self.io.autosave_async(self.get_current_state())

    # slot for io.autosave_finished
    @pyqtSlot(bool)
    def on_autosave_finished(self, success: bool):
        logging.info("Autosave %s", "succeeded" if success else "failed")

    # ==================================================================
    #  Callbacks for io.project_{loaded|saved|failed}
    # ==================================================================
    def _on_project_saved(self, path: str):
        QMessageBox.information(self.view, "Project Saved", f"Project written to {path}")

    def _on_save_failed(self, err: str):
        QMessageBox.critical(self.view, "Save Error", err)

    def _on_load_failed(self, err: str):
        QMessageBox.critical(self.view, "Load Error", err)

    # ==================================================================
    #  Successful load handler (same logic as before, just renamed)
    # ==================================================================
    def _on_project_loaded(self, state: ProjectState):
        self._initialize_model_if_needed(state)
        clusters = self._clusters_from_state(state)
        self.clustering_controller.clusters = clusters
        self.image_processing_controller.set_clusters(clusters)
        self.on_label_generator_method_changed(state.annotation_method)

        mb = self.view.window().menuBar()
        if hasattr(mb, "set_checked_annotation_method"):
            mb.set_checked_annotation_method(state.annotation_method)

        cluster_info = self.clustering_controller.generate_cluster_info()
        sel = state.selected_cluster_id
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=sel)

        if sel and sel in clusters:
            self.on_select_cluster(sel, force=True)
        elif clusters:
            self.on_select_cluster(next(iter(clusters)), force=True)

        self.view.hide_progress_bar()
        stats = self.clustering_controller.compute_labeling_statistics()
        self.view.update_labeling_statistics(stats)

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

        self._current_ann_method = method

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
            force:
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
    def on_next_cluster(self):
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

    def propagate_labeling_changes(self) -> None:
        """Re-compute adjusted_uncertainty for every annotation."""
        propagate_for_annotations(self._get_all_annotations())

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
    def get_current_state(self) -> ProjectState:
        """Return a fully‑validated ProjectState instance."""
        if self.image_data_model is None:
            return ProjectState(
                schema_version=4,
                data_backend="",
                data_path="",
                clusters={},
                cluster_order=[],
                selected_cluster_id=None,
                annotation_method=""
            )

        backend = getattr(
            self.image_data_model,
            "backend",
            Path(self.image_data_model.data_path).suffix.lstrip(".").lower(),
        )

        clusters = self.clustering_controller.get_clusters()
        state = ProjectState(
            schema_version=4,
            data_backend=backend,
            data_path=self.image_data_model.data_path,
            uncertainty="bald",
            clusters={
                str(cid): [a.to_dict() for a in annos]
                for cid, annos in clusters.items()
            },
            cluster_order=list(clusters),
            selected_cluster_id=self.view.get_selected_cluster_id(),
            annotation_method=self._current_ann_method
        )
        return state

    # -------------------------------------------------------------------------
    #                       SAVE / LOAD SLOTS AND CALLBACKS
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    #                               EXPORT
    # -------------------------------------------------------------------------

    @pyqtSlot(str)
    def export_annotations(self, export_file: str):
        clusters = self.clustering_controller.get_clusters()
        flat = [(cid, a) for cid, annos in clusters.items() for a in annos]

        opts = self.view.ask_export_options(flat)  # returns None on cancel
        if opts is None:
            return

        try:
            count = self._export_usecase(
                flat,
                ExportOptions(include_artifacts=opts.include_artifacts),
                Path(export_file),
            )
        except ValueError as e:
            QMessageBox.warning(self.view, "Export Aborted", str(e))
        else:
            QMessageBox.information(
                self.view,
                "Export Successful",
                f"Exported {count} annotations to\n{export_file}",
            )

    # -----------------------------------------------------------------
    #               CROSS VALIDATION FOLD GENERATION
    # -----------------------------------------------------------------
    @pyqtSlot(str, str)
    def create_cv_folds(self, data_dir: str, out_dir: str) -> None:
        """Generate grouped cross-validation folds asynchronously."""
        worker = CrossValidationWorker(data_dir, out_dir)
        worker.signals.finished.connect(
            lambda: QMessageBox.information(
                self.view, "CV Folds", "Fold generation completed."
            )
        )
        worker.signals.error.connect(
            lambda err: QMessageBox.critical(self.view, "CV Fold Error", err)
        )
        QThreadPool.globalInstance().start(worker)

    # -----------------------------------------------------------------
    #                    ANNOTATION PREVIEW DIALOG
    # -----------------------------------------------------------------
    @pyqtSlot()
    def show_annotation_preview(self) -> None:
        """Open a dialog showing full images with annotation overlays."""
        clusters = self.clustering_controller.get_clusters()
        annotations = [
            a for annos in clusters.values() for a in annos if a.class_id != -1
        ]
        dlg = AnnotationPreviewDialog(self.image_data_model, annotations, self.view)
        dlg.exec_()

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

    # -------------------------------------------------------------------------
    #                       PRIVATE / HELPER METHODS
    # -------------------------------------------------------------------------

    def _initialize_model_if_needed(self, state: ProjectState):
        data_path = state.data_path
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

    @pyqtSlot(str)
    def on_navigation_policy_changed(self, name: str):
        self.cluster_selector = make_selector(name, self.clustering_controller)
        self.view.navigation_widget.set_navigation_policy(name)

    def set_model(self, model: BaseImageDataModel):
        """Attach a new data model (HDF5 or SQLite) to every sub‑controller."""
        if self.image_data_model is model:
            return
        self.image_data_model = model
        self.clustering_controller.model = model
        self.image_processing_controller.model = model
        # update IO service so frames directory fingerprint is stable
        self.io._tag = model.data_path  # _tag is only used for temp dir names
        logging.info("Data model set → %s", model.data_path)
