import json
import logging
from collections import OrderedDict
from typing import Optional, Dict, List

from PyQt5.QtCore import QObject, pyqtSlot, QTimer, QCoreApplication
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from GUI.controllers.AnnotationClusteringController import AnnotationClusteringController
from GUI.controllers.ImageProcessingController import ImageProcessingController
from GUI.controllers.ProjectStateController import ProjectStateController
from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.PointAnnotationGenerator import LocalMaximaPointAnnotationGenerator, EquidistantPointAnnotationGenerator
from GUI.views.ClusteredCropsView import ClusteredCropsView


class MainController(QObject):
    """
    GlobalClusterController acts as the main orchestrator of the application.
    It handles user interactions, connects the view with other controllers,
    and keeps the UI responsive by updating it based on signals.
    """

    def __init__(self, model: Optional[ImageDataModel], view: ClusteredCropsView):
        """
        Initializes the GlobalClusterController.

        :param model: An instance of the ImageDataModel.
        :param view: An instance of the ClusteredCropsView.
        """
        super().__init__()
        self.image_data_model = model
        self.view = view
        self.annotation_generator = LocalMaximaPointAnnotationGenerator()

        # Instantiate other controllers with the initial model
        self.clustering_controller = AnnotationClusteringController(self.image_data_model, self.annotation_generator)
        self.image_processing_controller = ImageProcessingController(self.image_data_model)
        self.project_state_controller = ProjectStateController(self.image_data_model)

        # Autosave timer initialization (do not start it yet)
        self.autosave_timer = QTimer()
        self.autosave_timer.setInterval(30000)  # 30 seconds
        self.autosave_timer.timeout.connect(self.autosave_project_state)

        # Connect signals and slots
        self.connect_signals()

        # Ensure cleanup on application exit
        QCoreApplication.instance().aboutToQuit.connect(self.cleanup)

    def set_model(self, model: ImageDataModel):
        """
        Sets the model and starts the autosave timer if not already active.

        :param model: The new ImageDataModel instance to set.
        """
        if self.image_data_model == model:
            logging.debug("Model is already set. Skipping reassignment.")
            return

        self._assign_model_to_controllers(model)

        if not self.autosave_timer.isActive():
            self.autosave_timer.start()
        logging.info("Model has been updated and autosave timer started.")

    def connect_signals(self):
        """
        Connects signals from the view and other controllers to the appropriate methods.
        """
        # --- View -> Controller ---
        self.view.request_clustering.connect(self.clustering_controller.start_clustering)
        self.view.sample_cluster.connect(self.on_sample_cluster)
        self.view.sampling_parameters_changed.connect(self.on_sampling_parameters_changed)
        self.view.annotation_method_changed.connect(self.on_annotation_method_changed)
        self.view.bulk_label_changed.connect(self.on_bulk_label_changed)
        self.view.crop_label_changed.connect(self.on_crop_label_changed)
        self.view.save_project_state_requested.connect(self.save_project)
        self.view.export_annotations_requested.connect(self.export_annotations)
        self.view.load_project_state_requested.connect(self.load_project)
        self.view.save_project_requested.connect(self.save_project)
        self.view.save_project_as_requested.connect(self.save_project_as)

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
    def on_annotation_method_changed(self, method: str):
        """
        Instantiates the appropriate annotation generator based on the selected method,
        and updates the clustering controller accordingly.
        """
        if method == "Local Maxima":
            self.annotation_generator = LocalMaximaPointAnnotationGenerator(
                filter_size=48, gaussian_sigma=4.0, use_gaussian=False
            )
        elif method == "Equidistant Spots":
            self.annotation_generator = EquidistantPointAnnotationGenerator(grid_spacing=48)
        else:
            self.annotation_generator = LocalMaximaPointAnnotationGenerator(
                filter_size=48, gaussian_sigma=4.0, use_gaussian=False
            )
            logging.warning("Unknown annotation method selected; defaulting to Local Maxima.")

        self.clustering_controller.annotation_generator = self.annotation_generator
        logging.info("Annotation generator updated to method: %s", method)

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
            self.on_sample_cluster(first_cluster_id)
        self.view.hide_clustering_progress_bar()

    @pyqtSlot(int)
    def on_sample_cluster(self, cluster_id: int):
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

    @pyqtSlot(int, int)
    def on_sampling_parameters_changed(self, cluster_id: int, crops_per_cluster: int):
        """
        Handles changes in sampling parameters by refreshing the displayed crops.

        :param cluster_id: The ID of the selected cluster.
        :param crops_per_cluster: The updated number of crops per cluster.
        """
        logging.info(f"Sampling parameters changed: "
                     f"Cluster {cluster_id}, {crops_per_cluster} crops per cluster.")
        self.image_processing_controller.set_crops_per_cluster(crops_per_cluster, cluster_id)

    def handle_sampling_parameters_changed(self):
        """
        Handles the sampling parameters after debouncing.
        Refreshes the displayed crops based on the updated parameters.
        """
        logging.info("Handling debounced sampling parameter changes.")
        selected_cluster_id = self.view.get_selected_cluster_id()
        if selected_cluster_id is not None:
            self.on_sample_cluster(selected_cluster_id)
        else:
            logging.warning("No cluster is currently selected. Clearing displayed crops.")
            self.view.display_sampled_crops([])

    # -------------------------------------------------------------------------
    #                       LABELING / ANNOTATION HANDLERS
    # -------------------------------------------------------------------------

    @pyqtSlot(int)
    def on_bulk_label_changed(self, class_id: int):
        """
        Handles a bulk update for all visible crops' labels.
        Refreshes UI and saves the project state.

        :param class_id: The new class ID to apply.
        """
        selected_cluster_id = self.view.get_selected_cluster_id()
        clusters = self.clustering_controller.get_clusters()
        annotations = clusters.get(selected_cluster_id, [])

        for anno in annotations[:self.image_processing_controller.crops_per_cluster]:
            anno.class_id = class_id

        cluster_info = self.clustering_controller.generate_cluster_info()
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=selected_cluster_id)

        self.autosave_project_state()
        self.clustering_controller.compute_labeling_statistics()

    @pyqtSlot(dict, int)
    def on_crop_label_changed(self, crop_data: dict, class_id: int):
        """
        Handles when a class label is set for an individual crop.

        :param crop_data: Dictionary containing crop information.
        :param class_id: The new class ID assigned.
        """
        cluster_id = int(crop_data['cluster_id'])
        image_index = int(crop_data['image_index'])
        coord = tuple(crop_data['coord'])

        clusters = self.clustering_controller.get_clusters()
        annotations = clusters.get(cluster_id, [])

        for anno in annotations:
            if anno.image_index == image_index and anno.coord == coord:
                anno.class_id = class_id
                label_action = "unlabeled" if class_id == -1 else "labeled"
                logging.debug(
                    f"Annotation for image {image_index}, coord {coord} "
                    f"{label_action} with class_id {class_id}"
                )
                break
        else:
            logging.warning(f"Annotation for image {image_index}, coord {coord} "
                            f"not found in cluster {cluster_id}")

        cluster_info = self.clustering_controller.generate_cluster_info()
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=cluster_id)

        self.autosave_project_state()
        self.clustering_controller.compute_labeling_statistics()

    # -------------------------------------------------------------------------
    #                           PROJECT STATE
    # -------------------------------------------------------------------------

    def get_current_state(self) -> dict:
        """
        Retrieves the current project state, including cluster and annotation data.
        """
        if self.image_data_model is None:
            logging.debug("Cannot get current state: model is not initialized.")
            return {}

        clusters = self.clustering_controller.get_clusters()
        project_state = {
            'hdf5_file_path': self.image_data_model.hdf5_file_path,
            'annotations': {},
            'cluster_order': list(clusters.keys()),
            'selected_cluster_id': self.view.get_selected_cluster_id(),
        }

        for cluster_id, annotations in clusters.items():
            for anno in annotations:
                annotation_data = anno.to_dict()
                project_state['annotations'].setdefault(anno.filename, []).append(annotation_data)

        return project_state

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
        """
        Saves the project state to the current file path.
        If the current file path is not set, prompts the user to choose a save location.
        """
        current_path = self.project_state_controller.get_current_save_path()
        if current_path:
            project_state = self.get_current_state()
            self.project_state_controller.save_project_state(project_state, current_path)
        else:
            self.save_project_as()

    @pyqtSlot()
    def save_project_as(self):
        """
        Prompts the user to select a location and file name, and then saves the project state.
        """
        file_path = self._prompt_save_file()
        if not file_path:
            logging.info("Save As action was canceled by the user.")
            return

        self.project_state_controller.set_current_save_path(file_path)
        project_state = self.get_current_state()
        self.project_state_controller.save_project_state(project_state, file_path)

    @pyqtSlot()
    def load_project(self):
        """
        Loads the project state from a saved file to resume the session.
        """
        project_file = self._prompt_load_file()
        if project_file:
            self.project_state_controller.load_project_state(project_file)
        else:
            logging.info("Load project action was canceled by the user.")

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

    @pyqtSlot(dict)
    def on_project_loaded(self, project_state: dict):
        """
        Handles loading of the project state.

        :param project_state: The loaded project state.
        """
        clusters_data = self._rebuild_clusters_from_annotations(project_state)
        self.clustering_controller.clusters = clusters_data
        self.image_processing_controller.set_clusters(clusters_data)

        hdf5_file_path = project_state.get('hdf5_file_path', None)
        if not hdf5_file_path:
            logging.error("No hdf5_file_path in project_state.")
            QMessageBox.critical(self.view, "Error", "Project state does not contain hdf5_file_path.")
            return

        self._initialize_model_if_needed(hdf5_file_path, project_state)

        cluster_info = self.clustering_controller.generate_cluster_info()
        selected_cluster_id = project_state.get('selected_cluster_id', None)
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=selected_cluster_id)

        if selected_cluster_id and selected_cluster_id in clusters_data:
            self.on_sample_cluster(selected_cluster_id)
        else:
            first_cluster_id = next(iter(clusters_data), None)
            if first_cluster_id is not None:
                self.on_sample_cluster(first_cluster_id)

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
        self.autosave_timer.stop()
        self.image_processing_controller.cleanup()
        self.clustering_controller.cleanup()
        self.project_state_controller.cleanup()

    # -------------------------------------------------------------------------
    #                       PRIVATE / HELPER METHODS
    # -------------------------------------------------------------------------

    def _assign_model_to_controllers(self, model: ImageDataModel):
        """
        Helper method to reassign the model to all sub-controllers.
        """
        self.image_data_model = model
        self.clustering_controller.model = model
        self.image_processing_controller.model = model
        self.project_state_controller.model = model

    def _initialize_model_if_needed(self, hdf5_file_path: str, project_state: dict):
        """
        Helper to initialize a new ImageDataModel if needed (or update if paths differ).
        """
        if (self.image_data_model is None or
                self.image_data_model.hdf5_file_path != hdf5_file_path):
            uncertainty_type = project_state.get('uncertainty_type', 'variance')
            image_data_model = ImageDataModel(
                hdf5_file_path=hdf5_file_path,
                uncertainty_type=uncertainty_type
            )
            self.set_model(image_data_model)

    def _rebuild_clusters_from_annotations(self, project_state: dict) -> Dict[int, List[Annotation]]:
        """
        Helper method to rebuild clusters dict from the annotations in the project state.
        """
        annotations_data = project_state.get('annotations', {})
        clusters_data = {}
        for filename, annotations_list in annotations_data.items():
            for annotation_dict in annotations_list:
                anno = Annotation.from_dict(annotation_dict)
                cluster_id = anno.cluster_id
                clusters_data.setdefault(cluster_id, []).append(anno)

        cluster_order = project_state.get('cluster_order', None)
        if cluster_order is not None:
            clusters_data = self._reorder_clusters(cluster_order, clusters_data)
        return clusters_data

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
