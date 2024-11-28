import json
import logging
import os
import tempfile
from collections import OrderedDict
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSlot, QTimer, QCoreApplication
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from GUI.controllers.ClusteringController import ClusteringController
from GUI.controllers.ImageProcessingController import ImageProcessingController
from GUI.controllers.ProjectStateController import ProjectStateController
from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.UncertaintyRegionSelector import UncertaintyRegionSelector
from GUI.views.ClusteredCropsView import ClusteredCropsView

TEMP_DIR = os.path.join(tempfile.gettempdir(), 'my_application_temp')
os.makedirs(TEMP_DIR, exist_ok=True)


class GlobalClusterController(QObject):
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
        self.model = model
        self.view = view
        self.region_selector = UncertaintyRegionSelector()

        # Instantiate other controllers
        self.clustering_controller = ClusteringController(self.model, self.region_selector)
        self.image_processing_controller = ImageProcessingController(self.model)
        self.project_state_controller = ProjectStateController(self.model)

        # Autosave timer initialization (do not start it yet)
        self.autosave_timer = QTimer()
        self.autosave_timer.setInterval(60000)  # 60 seconds
        self.autosave_timer.timeout.connect(self.autosave_project_state)

        # Connect signals and slots
        self.connect_signals()

        # Ensure cleanup on application exit
        QCoreApplication.instance().aboutToQuit.connect(self.cleanup)

    def set_model(self, model: ImageDataModel):
        """
        Sets the model and starts the autosave timer if not already active.
        """
        self.model = model
        self.clustering_controller.model = model
        self.image_processing_controller.model = model
        self.project_state_controller.model = model
        # Start the autosave timer if not already started
        if not self.autosave_timer.isActive():
            self.autosave_timer.start()

    def connect_signals(self):
        """
        Connects signals from the view and other controllers to the appropriate methods.
        """
        # View signals to controller methods
        self.view.request_clustering.connect(self.clustering_controller.start_clustering)
        self.view.sample_cluster.connect(self.on_sample_cluster)
        self.view.sampling_parameters_changed.connect(self.on_sampling_parameters_changed)
        self.view.bulk_label_changed.connect(self.on_bulk_label_changed)
        self.view.crop_label_changed.connect(self.on_crop_label_changed)
        self.view.save_project_state_requested.connect(self.save_project)
        self.view.export_annotations_requested.connect(self.export_annotations)
        self.view.load_project_state_requested.connect(self.load_project)
        self.view.restore_autosave_requested.connect(self.restore_autosave)
        self.view.save_project_requested.connect(self.save_project)
        self.view.save_project_as_requested.connect(self.save_project_as)

        # ClusteringController signals to view
        self.clustering_controller.clustering_started.connect(self.view.show_clustering_progress_bar)
        self.clustering_controller.clustering_progress.connect(self.view.update_clustering_progress_bar)
        self.clustering_controller.clusters_ready.connect(self.on_clusters_ready)
        self.clustering_controller.labeling_statistics_updated.connect(self.view.update_labeling_statistics)

        # ImageProcessingController signals to view
        self.image_processing_controller.crops_ready.connect(self.view.display_sampled_crops)
        self.image_processing_controller.progress_updated.connect(self.view.update_crop_loading_progress_bar)
        self.image_processing_controller.crop_loading_started.connect(self.view.show_crop_loading_progress_bar)
        self.image_processing_controller.crop_loading_finished.connect(self.view.hide_crop_loading_progress_bar)

        # ProjectStateController signals to methods
        self.project_state_controller.autosave_finished.connect(self.on_autosave_finished)
        self.project_state_controller.project_loaded.connect(self.on_project_loaded)
        self.project_state_controller.project_saved.connect(self.on_project_saved)
        self.project_state_controller.save_failed.connect(self.on_save_failed)
        self.project_state_controller.load_failed.connect(self.on_load_failed)

    @pyqtSlot(dict)
    def on_clusters_ready(self, clusters):
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
        logging.info(f"Sampling parameters changed: Cluster {cluster_id}, {crops_per_cluster} crops per cluster.")
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

        # Update class_id for all annotations in the displayed crops
        for anno in annotations[:self.image_processing_controller.crops_per_cluster]:
            anno.class_id = class_id

        # Update cluster info and UI
        cluster_info = self.clustering_controller.generate_cluster_info()
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=selected_cluster_id)

        # Autosave the project state
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

        # Update the class_id for the corresponding Annotation instance
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

        # Refresh the cluster info and UI
        cluster_info = self.clustering_controller.generate_cluster_info()
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=cluster_id)

        # Autosave the project state
        self.autosave_project_state()
        self.clustering_controller.compute_labeling_statistics()

    def get_current_state(self) -> dict:
        if self.model is None:
            logging.debug("Cannot get current state: model is not initialized.")
            return {}
        clusters = self.clustering_controller.get_clusters()
        project_state = {
            'hdf5_file_path': self.model.hdf5_file_path,
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
        if self.model is None:
            logging.debug("Autosave skipped: model is not initialized.")
            return
        logging.debug("Autosave timer triggered.")
        project_state = self.get_current_state()
        self.project_state_controller.autosave_project_state(project_state)

    @pyqtSlot()
    def save_project(self):
        """
        Saves the project state to the current file path.
        If the current file path is not set, prompts the user to choose a save location.
        """
        if self.project_state_controller.get_current_save_path():
            project_state = self.get_current_state()
            self.project_state_controller.save_project_state(
                project_state,
                self.project_state_controller.get_current_save_path()
            )
        else:
            self.save_project_as()

    @pyqtSlot()
    def save_project_as(self):
        """
        Prompts the user to select a location and file name, and then saves the project state.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self.view, "Save Project As", "", "Compressed JSON Files (*.json.gz);;All Files (*)", options=options
        )
        if file_path:
            if not file_path.endswith('.json.gz'):
                file_path += '.json.gz'
            self.project_state_controller.set_current_save_path(file_path)
            project_state = self.get_current_state()
            self.project_state_controller.save_project_state(project_state, file_path)
        else:
            logging.info("Save As action was canceled by the user.")

    @pyqtSlot()
    def load_project(self):
        """
        Loads the project state from a saved file to resume the session.
        """
        options = QFileDialog.Options()
        project_file, _ = QFileDialog.getOpenFileName(
            self.view, "Open Project", "", "Compressed JSON Files (*.json.gz);;JSON Files (*.json);;All Files (*)",
            options=options
        )
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
        # Reconstruct clusters from annotations
        annotations_data = project_state.get('annotations', {})
        clusters_data = {}
        for filename, annotations_list in annotations_data.items():
            for annotation_dict in annotations_list:
                anno = Annotation.from_dict(annotation_dict)
                cluster_id = anno.cluster_id
                clusters_data.setdefault(cluster_id, []).append(anno)

        # If 'cluster_order' is provided, reorder clusters_data
        cluster_order = project_state.get('cluster_order', None)
        if cluster_order is not None:
            ordered_clusters_data = OrderedDict()
            for cluster_id in cluster_order:
                if cluster_id in clusters_data:
                    ordered_clusters_data[cluster_id] = clusters_data[cluster_id]
                else:
                    logging.warning(f"Cluster ID {cluster_id} not found in annotations.")
            # Add any clusters that were not in cluster_order
            for cluster_id in clusters_data:
                if cluster_id not in ordered_clusters_data:
                    ordered_clusters_data[cluster_id] = clusters_data[cluster_id]
            clusters_data = ordered_clusters_data

        self.clustering_controller.clusters = clusters_data

        # Update ImageProcessingController with the clusters
        self.image_processing_controller.set_clusters(clusters_data)

        hdf5_file_path = project_state.get('hdf5_file_path', None)
        if hdf5_file_path is None:
            logging.error("No hdf5_file_path in project_state.")
            QMessageBox.critical(self.view, "Error", "Project state does not contain hdf5_file_path.")
            return

        # Initialize model if not already initialized or if hdf5_file_path is different
        if self.model is None or self.model.hdf5_file_path != hdf5_file_path:
            model = ImageDataModel(hdf5_file_path)
            self.set_model(model)
            # Update model in controllers
            self.clustering_controller.model = self.model
            self.image_processing_controller.model = self.model
            self.project_state_controller.model = self.model

        # Update cluster info and UI
        cluster_info = self.clustering_controller.generate_cluster_info()
        selected_cluster_id = project_state.get('selected_cluster_id', None)
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=selected_cluster_id)

        # Display crops of the selected cluster
        if selected_cluster_id is not None and selected_cluster_id in clusters_data:
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

    def export_annotations(self):
        """
        Exports the labeled annotations in a final format for downstream use.
        """
        clusters = self.clustering_controller.get_clusters()
        grouped_annotations = {}
        annotations_without_class = False

        for cluster_id, annotations in clusters.items():
            for anno in annotations:
                if anno.class_id is not None and anno.class_id != -1 and anno.class_id != -2:
                    annotation_data = {
                        'coord': [int(c) for c in anno.coord],
                        'class_id': int(anno.class_id),
                        'cluster_id': int(cluster_id)
                    }
                    grouped_annotations.setdefault(anno.filename, []).append(annotation_data)
                else:
                    annotations_without_class = True

        if annotations_without_class:
            reply = QMessageBox.question(
                self.view,
                "Annotations Without Class Labels",
                "Some annotations do not have class labels assigned. Do you want to proceed with exporting?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                logging.info("User canceled exporting due to missing class labels.")
                return

        if not grouped_annotations:
            QMessageBox.warning(self.view, "No Annotations", "There are no annotations to export.")
            return

        # Ask user where to save the export
        options = QFileDialog.Options()
        export_file, _ = QFileDialog.getSaveFileName(
            self.view, "Export Annotations", "", "JSON Files (*.json);;All Files (*)", options=options
        )
        if export_file:
            try:
                with open(export_file, 'w') as f:
                    json.dump(grouped_annotations, f, indent=4)
                logging.info(f"Annotations exported to {export_file}")
                QMessageBox.information(self.view, "Export Successful", f"Annotations exported to {export_file}")
            except Exception as e:
                logging.error(f"Error exporting annotations: {e}")
                QMessageBox.critical(self.view, "Error", f"Failed to export annotations: {e}")
        else:
            logging.debug("Export annotations action was canceled.")

    def cleanup(self):
        """
        Cleans up resources before application exit.
        """
        logging.info("Cleaning up before application exit.")
        self.autosave_timer.stop()
        self.image_processing_controller.cleanup()
        self.clustering_controller.cleanup()
        self.project_state_controller.cleanup()
