# controllers/GlobalClusterController.py

import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QTimer, QPoint, QCoreApplication
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from sklearn.cluster import AgglomerativeClustering

from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.ImageProcessor import ImageProcessor
from GUI.models.UncertaintyRegionSelector import UncertaintyRegionSelector
from GUI.views.ClusteredCropsView import ClusteredCropsView

# Create a dedicated temp directory for your application
TEMP_DIR = os.path.join(tempfile.gettempdir(), 'my_application_temp')
os.makedirs(TEMP_DIR, exist_ok=True)

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


class AutosaveWorker(QObject):
    save_finished = pyqtSignal(bool)
    save_project_state_signal = pyqtSignal(object, str)

    def __init__(self):
        super().__init__()
        self.save_project_state_signal.connect(self.save_project_state)

    @pyqtSlot(object, str)
    def save_project_state(self, project_state, file_path):
        """
        Saves the project state to the specified file path asynchronously.
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(project_state, f, indent=4)
            logging.info(f"Autosave completed successfully to {file_path}")
            self.save_finished.emit(True)  # Signal successful save
        except TypeError as e:
            logging.error(f"Serialization error during autosave: {e}")
            self.save_finished.emit(False)
        except Exception as e:
            logging.error(f"Error during async autosave: {e}")
            self.save_finished.emit(False)  # Signal failed save


class ClusteringWorker(QThread):
    """
    Worker thread for performing global clustering to keep the UI responsive.
    """
    clustering_finished = pyqtSignal(list)
    progress_updated = pyqtSignal(int)

    def __init__(self, model: ImageDataModel, labels_acquirer: UncertaintyRegionSelector, parent=None):
        super().__init__(parent)
        self.model = model
        self.labels_acquirer = labels_acquirer

    def process_image(self, idx):
        """
        Processes a single image and assigns a cluster_id to each Annotation.
        """
        data = self.model.get_image_data(idx)
        uncertainty_map = data.get('uncertainty', None)
        logits = data.get('logits', None)
        filename = data.get('filename', None)

        if uncertainty_map is None or logits is None or filename is None:
            logging.warning(f"Missing data for image index {idx}. Skipping.")
            return [], idx

        logit_features, dbscan_coords = self.labels_acquirer.generate_point_labels(
            uncertainty_map=uncertainty_map,
            logits=logits
        )

        annotations = [
            Annotation(
                filename=filename,
                coord=coord,
                logit_features=logit_feature,
                class_id=-1,  # Default to -1 (unlabelled)
                image_index=idx,
                uncertainty=uncertainty_map[coord[0], coord[1]],
                cluster_id=None  # Assign the cluster ID here
            )
            for coord, logit_feature in zip(dbscan_coords, logit_features)
        ]

        return annotations, idx

    def run(self):
        """
        Executes the clustering process with parallel image processing.
        """
        all_annotations = []  # List to collect annotations with image_index, coord, logit_features, etc.
        total_images = self.model.get_number_of_images()
        logging.info(f"Starting clustering on {total_images} images.")

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.process_image, idx): idx for idx in range(total_images)}

            for i, future in enumerate(as_completed(futures)):
                try:
                    annotations, idx = future.result()
                    all_annotations.extend(annotations)
                except Exception as e:
                    logging.error(f"Error processing image {futures[future]}: {e}")
                    continue

                # Emit progress update
                progress = int(((i + 1) / total_images) * 100)
                self.progress_updated.emit(progress)
                logging.debug(f"Processed image {idx + 1}/{total_images}. Progress: {progress}%")

        logging.info(f"Total annotations collected: {len(all_annotations)}")

        if not all_annotations:
            logging.warning("No annotations found across all images.")
            self.clustering_finished.emit({})
            return

        # Extract logit features for clustering
        try:
            logit_matrix = np.array([anno.logit_features for anno in all_annotations])
            logging.debug(f"Logit matrix shape: {logit_matrix.shape}")
        except Exception as e:
            logging.error(f"Error creating logit matrix: {e}")
            self.clustering_finished.emit({})
            return

        # Perform clustering
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.labels_acquirer.distance_threshold,
                linkage=self.labels_acquirer.linkage
            )
            clustering.fit(logit_matrix)
            labels = clustering.labels_
            logging.debug(f"Clustering labels generated: {labels}")
        except Exception as e:
            logging.error(f"Clustering failed: {e}")
            self.clustering_finished.emit({})
            return

        # Assign cluster IDs to each annotation
        for label, annotation in zip(labels, all_annotations):
            annotation.cluster_id = int(label)  # Directly assign cluster_id

        logging.info(f"Global clustering complete. Number of clusters: {len(set(clustering.labels_))}")
        self.clustering_finished.emit(all_annotations)


class ImageProcessingWorker(QObject):
    processing_finished = pyqtSignal(list)

    def __init__(self, sampled_annotations, model, image_processor):
        super().__init__()
        self.sampled_annotations = sampled_annotations
        self.model = model
        self.image_processor = image_processor

    @pyqtSlot()
    def process_images(self):
        sampled_crops = []

        for anno in self.sampled_annotations:
            result = self.process_single_annotation(anno)
            if result is not None:
                sampled_crops.append(result)

        self.processing_finished.emit(sampled_crops)

    def process_single_annotation(self, anno):
        cache_key = (anno.image_index, tuple(anno.coord))
        cached_result = self.image_processor.cache.get(cache_key)
        if cached_result:
            processed_crop, coord_pos = cached_result
        else:
            # Process the crop
            image_data = self.model.get_image_data(anno.image_index)
            image_array = image_data.get('image', None)
            coord = anno.coord
            if image_array is None:
                return None

            processed_crop, coord_pos = self.image_processor.extract_crop_data(
                image_array, coord, crop_size=512, zoom_factor=2
            )

            # Store in cache
            self.image_processor.cache.set(cache_key, (processed_crop, coord_pos))

        return {
            'annotation': anno,
            'processed_crop': processed_crop,
            'coord_pos': coord_pos
        }


class PrefetchWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, cluster_ids: List[int], clusters: Dict[int, List[Annotation]],
                 model: ImageDataModel, image_processor: ImageProcessor, crops_per_cluster: int):
        super().__init__()
        self.cluster_ids = cluster_ids
        self.clusters = clusters
        self.model = model
        self.image_processor = image_processor
        self.crops_per_cluster = crops_per_cluster
        self.is_running = False
        self._is_stopped = False

    @pyqtSlot()
    def run(self):
        """
        Prefetches data for the specified cluster IDs.
        """
        self.is_running = True
        for cluster_id in self.cluster_ids:
            if self._is_stopped:
                logging.info(f"Prefetching stopped. Exiting prefetch worker.")
                break
            annotations = self.clusters.get(cluster_id, [])
            num_samples = min(self.crops_per_cluster, len(annotations))
            selected = annotations[:num_samples]

            logging.info(f"Prefetching cluster {cluster_id}: {num_samples} annotations.")

            for anno in selected:
                if self._is_stopped:
                    logging.info(f"Prefetching stopped during annotation processing.")
                    break
                # Prefetch image data
                try:
                    cache_key = (anno.image_index, tuple(anno.coord))
                    if self.image_processor.cache.get(cache_key):
                        logging.debug(f"Annotation {anno} already prefetched. Skipping.")
                        continue  # Already prefetched

                    image_data = self.model.get_image_data(anno.image_index)
                    image_array = image_data.get('image', None)
                    coord = anno.coord
                    if image_array is None:
                        logging.warning(f"No image data found for image index {anno.image_index}.")
                        continue

                    # Process the crop
                    processed_crop, coord_pos = self.image_processor.extract_crop_data(
                        image_array, coord, crop_size=512, zoom_factor=2
                    )

                    # Store in cache
                    self.image_processor.cache.set(cache_key, (processed_crop, coord_pos))

                    logging.debug(f"Prefetched annotation {anno}.")

                except Exception as e:
                    logging.error(f"Error prefetching data for cluster {cluster_id}, annotation {anno}: {e}")
                    continue
        self.is_running = False
        self.finished.emit()

    def stop(self):
        """
        Stops the prefetching process.
        """
        self._is_stopped = True


class GlobalClusterController(QObject):
    """
    GlobalClusterController handles global clustering across all images and manages the presentation of sampled crops.
    """

    clustering_started = pyqtSignal()
    clustering_progress = pyqtSignal(int)
    clusters_ready = pyqtSignal(dict)

    def __init__(self, model: ImageDataModel, view: ClusteredCropsView):
        super().__init__()
        self.model = model
        self.view = view
        self.image_processor = ImageProcessor()
        self.region_selector = UncertaintyRegionSelector()
        self.is_saving = False

        self.prefetched_clusters = set()  # Keep track of prefetched clusters

        self.prefetch_timer = QTimer()
        self.prefetch_timer.setSingleShot(True)
        self.prefetch_timer.setInterval(100)  # Adjust the interval as needed
        self.prefetch_timer.timeout.connect(self.execute_prefetch)

        # Initialize attributes before restoring project state
        self.clusters: Dict[int, List[Annotation]] = {}
        self.cluster_labels: Dict[int, str] = {}

        self.crops_per_cluster = 100

        # Set the prefetch buffer size here
        self.prefetch_buffer_size = 10  # Adjust this value as needed

        self.debounce_timer = QTimer()
        self.debounce_timer.setInterval(300)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.handle_sampling_parameters_changed)

        # Initialize autosave worker and thread
        self.autosave_worker = AutosaveWorker()
        self.autosave_thread = QThread()
        self.autosave_worker.moveToThread(self.autosave_thread)
        self.autosave_worker.save_finished.connect(self.on_autosave_finished)
        self.autosave_thread.start()

        # Autosave file path within the dedicated temp directory
        self.temp_file_path = os.path.join(TEMP_DIR, 'project_autosave.json')
        logging.info(f"Temporary autosave file path: {self.temp_file_path}")

        # Set autosave interval to 60 seconds
        self.autosave_interval = 60000  # 60 seconds
        self.autosave_timer = QTimer()
        self.autosave_timer.setInterval(self.autosave_interval)
        self.autosave_timer.timeout.connect(self.autosave_project_state)
        self.autosave_timer.start()

        # Initialize threads to None
        self.image_thread: Optional[QThread] = None
        self.prefetch_thread: Optional[QThread] = None
        self.worker: Optional[ClusteringWorker] = None

        # Now, attempt to load the last autosave
        latest_autosave_file = self.get_latest_autosave_file()
        if latest_autosave_file:
            logging.info(f"Autosave file found: {latest_autosave_file}")
            # Prompt the user to restore the last autosave
            reply = QMessageBox.question(
                self.view,
                "Restore Autosave?",
                "An autosave file was found. Do you want to restore your last session?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.load_project_state(latest_autosave_file)
            else:
                logging.info("User chose not to restore the autosave.")
        else:
            logging.info("No autosave file found.")

        # Connect view signals
        self.connect_signals()

        QCoreApplication.instance().aboutToQuit.connect(self.cleanup)

    def connect_signals(self):
        """
        Connect signals from the view to controller methods.
        """
        self.view.request_clustering.connect(self.start_clustering)
        self.view.sample_cluster.connect(self.on_sample_cluster)
        self.view.sampling_parameters_changed.connect(self.on_selection_parameters_changed)
        self.view.class_selected.connect(self.on_class_selected)
        self.view.class_selected_for_all.connect(self.on_class_selected_for_all)
        self.view.crop_label_changed.connect(self.on_crop_label_changed)
        self.view.save_project_state_requested.connect(self.on_save_project_state)
        self.view.export_annotations_requested.connect(self.on_export_annotations)
        self.view.load_project_state_requested.connect(self.on_load_project_state)
        self.view.restore_autosave_requested.connect(self.on_restore_autosave_requested)

        self.clustering_progress.connect(self.view.update_progress)
        self.clusters_ready.connect(self.on_clusters_ready)

    def cleanup(self):
        logging.info("Cleaning up threads before application exit.")
        # Stop the autosave thread
        self.autosave_thread.quit()
        self.autosave_thread.wait()
        logging.info("Autosave thread terminated.")

    @pyqtSlot()
    def start_clustering(self):
        """
        Initiates the global clustering process.
        """
        logging.info("Clustering process initiated by user.")
        self.clustering_started.emit()  # This signals the start in the GUI
        self.view.reset_progress()  # Ensure the progress bar is reset

        # Initialize and configure the ClusteringWorker
        self.clustering_worker = ClusteringWorker(model=self.model, labels_acquirer=self.region_selector)
        self.clustering_worker.progress_updated.connect(self.clustering_progress.emit)
        self.clustering_worker.clustering_finished.connect(self.on_clustering_finished)
        self.clustering_worker.finished.connect(self.clustering_worker.deleteLater)  # Clean up thread
        self.clustering_worker.start()  # Starts the worker

    @pyqtSlot(dict)
    def on_clusters_ready(self, clusters: Dict[int, List[Annotation]]):
        """
        Handles the completion of the clustering process.
        """
        self.clusters = clusters
        logging.info(f"Clustering finished with {len(clusters)} clusters.")

        # Prepare cluster_info dict
        cluster_info = self.generate_cluster_info()
        # Sort clusters by number of annotations in descending order
        sorted_cluster_info = dict(
            sorted(cluster_info.items(), key=lambda item: item[1]['num_annotations'], reverse=True))
        # Get the first cluster ID to display
        first_cluster_id = list(sorted_cluster_info.keys())[0] if sorted_cluster_info else None
        # Populate the cluster selection ComboBox
        self.view.populate_cluster_selection(sorted_cluster_info, selected_cluster_id=first_cluster_id)
        if first_cluster_id is not None:
            self.display_crops(cluster_id=first_cluster_id)
        self.view.hide_progress_bar()

    @pyqtSlot(list)
    def on_clustering_finished(self, annotations: list):
        """
        Organizes annotations by cluster_id and stores them in self.clusters.
        """
        self.clusters = {}

        # Group annotations by cluster_id
        for annotation in annotations:
            if not isinstance(annotation, Annotation):
                annotation = Annotation.from_dict(annotation)  # Ensure conversion to Annotation instance
            cluster_id = annotation.cluster_id

            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []

            self.clusters[cluster_id].append(annotation)

        logging.info(f"Clustering finished with {len(self.clusters)} clusters.")
        self.clusters_ready.emit(self.clusters)

    def generate_cluster_info(self) -> Dict[int, dict]:
        """
        Generates a dictionary containing cluster information for populating the GUI's cluster selection.
        """
        cluster_info = {}
        for cluster_id, annotations in self.clusters.items():
            image_filenames = set(anno.filename for anno in annotations)
            num_annotations = len(annotations)
            num_images = len(image_filenames)

            # Count labeled annotations (class_id != -1)
            num_labeled = sum(1 for anno in annotations if anno.class_id != -1)
            labeled_percentage = (num_labeled / num_annotations) * 100 if num_annotations > 0 else 0

            cluster_info[cluster_id] = {
                'num_annotations': num_annotations,
                'num_images': num_images,
                'labeled_percentage': labeled_percentage,
                'label': self.cluster_labels.get(cluster_id, '')
            }
        return cluster_info

    @pyqtSlot(int)
    def on_sample_cluster(self, cluster_id: int):
        """
        Handles requests to display crops from a specific cluster.
        """
        logging.info(f"Sampling crops from cluster {cluster_id}.")
        self.display_crops(cluster_id)

    def display_crops(self, cluster_id: int):
        """
        Selects the top 'n' annotations from a single cluster to display as zoomed-in crops.
        """
        cluster_id = int(cluster_id)
        if cluster_id not in self.clusters:
            logging.warning(f"Cluster ID {cluster_id} does not exist.")
            self.view.display_sampled_crops([])
            return

        annotations = self.clusters[cluster_id]
        if not annotations:
            logging.warning(f"No annotations found in cluster {cluster_id}.")
            self.view.display_sampled_crops([])
            return

        # Select the top 'n' annotations within the cluster
        num_samples = min(self.crops_per_cluster, len(annotations))
        selected = annotations[:num_samples]
        logging.debug(f"Selected top {len(selected)} annotations from cluster {cluster_id}.")

        # Check if data is prefetched
        processed_annotations = []
        annotations_to_process = []

        for anno in selected:
            cache_key = (anno.image_index, tuple(anno.coord))
            cached_result = self.image_processor.cache.get(cache_key)
            if cached_result:
                processed_crop, coord_pos = cached_result
                processed_annotations.append({
                    'annotation': anno,
                    'processed_crop': processed_crop,
                    'coord_pos': coord_pos
                })
            else:
                annotations_to_process.append(anno)

        # Process any annotations not already prefetched
        if annotations_to_process:
            self.image_worker = ImageProcessingWorker(
                sampled_annotations=annotations_to_process,
                model=self.model,
                image_processor=self.image_processor
            )
            # Image worker setup
            self.image_thread = QThread()
            self.image_worker.moveToThread(self.image_thread)
            self.image_thread.started.connect(self.image_worker.process_images)
            self.image_worker.processing_finished.connect(self.on_image_processing_finished)
            self.image_worker.processing_finished.connect(self.image_thread.quit)
            self.image_worker.processing_finished.connect(self.image_worker.deleteLater)
            self.image_thread.finished.connect(self.image_thread.deleteLater)
            self.image_thread.start()
        else:
            # If all data is already processed, display it
            self.on_image_processing_finished(processed_annotations)

        # Initiate prefetching for adjacent clusters
        self.prefetch_adjacent_clusters(cluster_id)

    def prefetch_adjacent_clusters(self, current_cluster_id: int):
        """
        Schedules prefetching of data for adjacent clusters after a debounce interval.
        """
        self.current_cluster_id_for_prefetch = current_cluster_id
        self.prefetch_timer.start()

    def execute_prefetch(self):
        """
        Initiates prefetching of data for adjacent clusters.
        """
        current_cluster_id = self.current_cluster_id_for_prefetch
        # Get the list of cluster IDs in the order they appear in the dropdown
        cluster_ids = self.view.get_cluster_id_list()
        if current_cluster_id not in cluster_ids:
            logging.warning(f"Current cluster ID {current_cluster_id} not found in cluster list.")
            return

        current_index = cluster_ids.index(current_cluster_id)

        clusters_to_prefetch = []

        # Prefetch clusters ahead of the current cluster
        for offset in range(1, self.prefetch_buffer_size + 1):
            next_index = current_index + offset
            if next_index < len(cluster_ids):
                clusters_to_prefetch.append(cluster_ids[next_index])

        # Prefetch clusters before the current cluster
        for offset in range(1, self.prefetch_buffer_size + 1):
            prev_index = current_index - offset
            if prev_index >= 0:
                clusters_to_prefetch.append(cluster_ids[prev_index])

        logging.info(f"Prefetching clusters: {clusters_to_prefetch}")

        # Start prefetching in a separate thread
        self.start_prefetching(clusters_to_prefetch)

    def start_prefetching(self, cluster_ids: List[int]):
        """
        Starts the prefetching process for the specified cluster IDs.
        """
        # Cancel any ongoing prefetching
        if hasattr(self, 'prefetch_worker') and self.prefetch_worker.is_running:
            self.prefetch_worker.stop()
            self.prefetch_thread.quit()
            self.prefetch_thread.wait()

            # ... existing code ...

        # Filter out clusters that have already been prefetched
        clusters_to_prefetch = [cid for cid in cluster_ids if cid not in self.prefetched_clusters]

        if not clusters_to_prefetch:
            logging.info("All clusters already prefetched. Skipping prefetching.")
            return

        # Update the set of prefetched clusters
        self.prefetched_clusters.update(clusters_to_prefetch)

        # Initialize PrefetchWorker
        self.prefetch_worker = PrefetchWorker(
            cluster_ids=clusters_to_prefetch,
            clusters=self.clusters,
            model=self.model,
            image_processor=self.image_processor,
            crops_per_cluster=self.crops_per_cluster
        )

        # Set up thread
        self.prefetch_thread = QThread()
        self.prefetch_worker.moveToThread(self.prefetch_thread)
        self.prefetch_thread.started.connect(self.prefetch_worker.run)
        self.prefetch_worker.finished.connect(self.prefetch_thread.quit)
        self.prefetch_worker.finished.connect(self.prefetch_worker.deleteLater)
        self.prefetch_thread.finished.connect(self.prefetch_thread.deleteLater)

        # Start prefetching
        self.prefetch_thread.start()

    @pyqtSlot(list)
    def on_image_processing_finished(self, processed_annotations: List[dict]):
        """
        Handles the processed crops from the ImageProcessingWorker.
        """
        logging.info(f"Received {len(processed_annotations)} processed crops from worker.")
        sampled_crops = []
        for data in processed_annotations:
            annotation = data['annotation']
            processed_crop = data['processed_crop']
            coord_pos = data['coord_pos']
            if processed_crop is None or coord_pos is None:
                logging.warning(f"Processed crop or coord_pos is None for annotation: {annotation}")
                continue
            q_image = self.image_processor.numpy_to_qimage(processed_crop)
            q_image = self.draw_annotation_on_image(q_image, coord_pos)
            q_pixmap = QPixmap.fromImage(q_image)
            sampled_crops.append({
                'annotation': annotation,
                'processed_crop': q_pixmap,
                'coord_pos': coord_pos
            })
        self.view.display_sampled_crops(sampled_crops)
        logging.info(f"Displayed {len(sampled_crops)} sampled crops.")

    def draw_annotation_on_image(self, q_image: QImage, coord_pos: tuple) -> QImage:
        """
        Draws an annotation (circle) on the provided QImage at the specified position.
        """
        zoomed_qimage = q_image.copy()  # Make a copy to ensure data integrity
        painter = QPainter(zoomed_qimage)
        pen = QPen(QColor(0, 255, 0))  # Green color for the circle
        pen.setWidth(5)  # Thickness of the circle outline
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing)

        pos_x_zoomed, pos_y_zoomed = coord_pos

        # Ensure the position is within the image bounds
        pos_x_zoomed = min(max(pos_x_zoomed, 0), zoomed_qimage.width() - 1)
        pos_y_zoomed = min(max(pos_y_zoomed, 0), zoomed_qimage.height() - 1)

        # Define a radius for the circle
        radius = max(8, min(zoomed_qimage.width(), zoomed_qimage.height()) // 40)

        # Draw the unfilled circle
        painter.drawEllipse(QPoint(int(pos_x_zoomed), int(pos_y_zoomed)), radius, radius)

        painter.end()

        return zoomed_qimage

    @pyqtSlot(object)
    def on_class_selected_for_all(self, class_id: Optional[int]):
        """
        Labels all visible crops with the selected class. If class_id is None, unlabel all visible crops.
        """
        if class_id is None:
            logging.info("Unlabeling all visible crops.")
        else:
            logging.info(f"Class {class_id} selected for all visible crops.")

        # Update the class_id for each visible crop
        for crop in self.view.selected_crops:
            cluster_id = int(crop['annotation'].cluster_id)
            image_index = int(crop['annotation'].image_index)
            coord = tuple(crop['annotation'].coord)  # Ensure coord is a tuple

            # Update the class_id for the corresponding Annotation in self.clusters
            for anno in self.clusters.get(cluster_id, []):
                if anno.image_index == image_index and anno.coord == coord:
                    anno.class_id = class_id if class_id is not None else -1
                    logging.debug(f"Annotation for image {image_index}, coord {coord} updated with class_id {class_id}")
                    break

        # Update the view to reflect these changes
        self.view.label_all_visible_crops(class_id)

    @pyqtSlot(int, int)
    def on_class_selected(self, cluster_id: int, class_id: int):
        """
        Handles the event when a class is selected for a cluster.
        Assigns the class label to every sample in the current view for the selected cluster.
        """
        logging.info(f"Class {class_id} selected for all samples in cluster {cluster_id}.")
        annotations = self.clusters.get(cluster_id, [])
        for anno in annotations:
            anno.class_id = class_id
            logging.debug(f"Annotation for image {anno.image_index} updated with class_id {class_id}")

        # Refresh the view to show updated labels
        self.display_crops(cluster_id)

    @pyqtSlot(dict, int)
    def on_crop_label_changed(self, crop_data: dict, class_id: int):
        """
        Handles when a class label is set for an individual crop.
        """
        if not isinstance(class_id, int):
            logging.error(f"class_id is not an integer. Received: {type(class_id)}")
            return

        cluster_id = int(crop_data['cluster_id'])
        image_index = int(crop_data['image_index'])
        coord = tuple(crop_data['coord'])  # Ensure coord is a tuple

        # Update the class_id for the corresponding Annotation instance
        for anno in self.clusters.get(cluster_id, []):
            if anno.image_index == image_index and anno.coord == coord:
                anno.class_id = class_id
                label_action = "unlabeled" if class_id == -1 else "labeled"
                logging.debug(
                    f"Annotation for image {image_index}, coord {coord} {label_action} with class_id {class_id}")
                break
        else:
            logging.warning(f"Annotation for image {image_index}, coord {coord} not found in cluster {cluster_id}")

        # Refresh the cluster info and UI
        cluster_info = self.generate_cluster_info()
        self.view.populate_cluster_selection(cluster_info, selected_cluster_id=cluster_id)

    @pyqtSlot(int, int)
    def on_selection_parameters_changed(self, cluster_id: int, crops_per_cluster: int):
        """
        Handles changes in selection parameters (number of crops per cluster).
        Implements debouncing to prevent excessive sampling.
        """
        logging.info(f"Selection parameters updated: {crops_per_cluster} crops per cluster.")
        self.crops_per_cluster = crops_per_cluster

        # Restart the debounce timer
        if self.debounce_timer.isActive():
            self.debounce_timer.stop()
        self.debounce_timer.start()

    def export_annotations(self):
        """
        Exports the labeled annotations in a final format for downstream use.
        """
        grouped_annotations: Dict[str, List[dict]] = {}
        annotations_without_class = False

        for cluster_id, annotations in self.clusters.items():
            for anno in annotations:
                if anno.class_id is not None and anno.class_id != -1:
                    annotation_data = {
                        'coord': list(anno.coord),
                        'class_id': anno.class_id,
                        'cluster_id': cluster_id
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

    def autosave_project_state(self):
        """
        Autosaves the current project state to a versioned backup file.
        """
        if self.is_saving:
            logging.info("Autosave already in progress, skipping this autosave.")
            return  # Skip if a save is already in progress

        if not self.clusters:
            logging.info("No clusters to save. Skipping autosave.")
            return  # No data to save

        self.is_saving = True

        # Prepare the project state data
        project_state = self.get_current_state()

        # Create a versioned backup path
        versioned_backup_path = self.get_versioned_backup_path(self.temp_file_path, max_backups=5)

        # Send save request to worker
        self.autosave_worker.save_project_state_signal.emit(project_state, versioned_backup_path)

    @pyqtSlot(bool)
    def on_autosave_finished(self, success: bool):
        """
        Handles completion of the autosave operation.
        """
        if success:
            logging.info("Autosave completed successfully.")
        else:
            logging.error("Autosave failed.")

        self.is_saving = False  # Reset the saving flag

    @pyqtSlot(str)
    def on_load_project_state(self, project_file: str):
        """
        Loads the project state from a saved file to resume the session.
        """
        self.load_project_state(project_file)

    @pyqtSlot()
    def on_save_project_state(self):
        """
        Prompts the user to save the current project state to a file.
        """
        options = QFileDialog.Options()
        project_file, _ = QFileDialog.getSaveFileName(
            self.view, "Save Project State", "", "JSON Files (*.json);;All Files (*)", options=options
        )
        if project_file:
            self.save_project_state(project_file)
        else:
            logging.debug("Save project state action was canceled.")

    def save_project_state(self, project_file_path: Optional[str] = None, show_popup: bool = True):
        """
        Saves the current project state, including in-progress annotations and settings.
        The `show_popup` parameter controls whether a success message is displayed.
        """
        project_file = project_file_path or os.path.join(TEMP_DIR, 'labelling_project_state.json')
        project_state = self.get_current_state()

        try:
            with open(project_file, 'w') as f:
                json.dump(project_state, f, indent=4)
            logging.info(f"Project state saved to {project_file}")

            if show_popup:
                QMessageBox.information(self.view, "Project Saved", f"Project state saved to {project_file}")

        except (TypeError, IOError) as e:
            logging.error(f"Failed to save project state: {e}")
            QMessageBox.critical(self.view, "Error", f"Failed to save project state: {e}")

    def get_current_state(self) -> dict:
        """
        Extracts the current project state, organizing clusters and annotations in a dictionary format.
        """
        project_state = {}

        for cluster_id, annotations in self.clusters.items():
            for anno in annotations:
                annotation_data = anno.to_dict()
                project_state.setdefault(anno.filename, []).append(annotation_data)

        return project_state

    def get_versioned_backup_path(self, base_path: str, max_backups: int = 5) -> str:
        """
        Generates a versioned backup path by adding a timestamp.
        Deletes older backups if they exceed max_backups.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.basename(base_path)}_{timestamp}.json"
        backup_path = os.path.join(TEMP_DIR, backup_filename)

        # Remove old backups if exceeding max_backups
        all_backups = sorted([f for f in os.listdir(TEMP_DIR)
                              if f.startswith(os.path.basename(base_path)) and f.endswith(".json")])

        while len(all_backups) > max_backups:
            old_backup = all_backups.pop(0)  # Remove the oldest file
            os.remove(os.path.join(TEMP_DIR, old_backup))
            logging.info(f"Deleted old backup file: {old_backup}")

        return backup_path

    def get_latest_autosave_file(self) -> Optional[str]:
        """
        Finds the most recent autosave file in the TEMP_DIR.
        """
        autosave_files = [f for f in os.listdir(TEMP_DIR)
                          if f.startswith('project_autosave') and f.endswith('.json')]
        if not autosave_files:
            return None  # No autosave files found
        # Sort files by modification time in descending order
        autosave_files.sort(key=lambda f: os.path.getmtime(os.path.join(TEMP_DIR, f)), reverse=True)
        latest_autosave = autosave_files[0]
        return os.path.join(TEMP_DIR, latest_autosave)

    def get_autosave_files(self) -> List[str]:
        """
        Returns a list of available autosave files in TEMP_DIR.
        """
        autosave_files = [f for f in os.listdir(TEMP_DIR)
                          if f.startswith('project_autosave') and f.endswith('.json')]
        if not autosave_files:
            return []
        # Sort files by modification time in descending order
        autosave_files.sort(key=lambda f: os.path.getmtime(os.path.join(TEMP_DIR, f)), reverse=True)
        # Return full paths
        autosave_files_full = [os.path.join(TEMP_DIR, f) for f in autosave_files]
        return autosave_files_full

    def handle_sampling_parameters_changed(self):
        """
        Handles the sampling parameters after debouncing.
        This method is called when the debounce timer times out, indicating that
        the user has finished adjusting sampling parameters such as the number of crops per cluster.
        It refreshes the displayed crops based on the updated parameters.
        """
        logging.info("Handling debounced sampling parameter changes.")

        # Retrieve the currently selected cluster ID from the view
        selected_cluster_id = self.view.get_selected_cluster_id()

        if selected_cluster_id is not None:
            # If a cluster is selected, update the displayed crops based on the new sampling parameters
            logging.debug(f"Selected cluster ID: {selected_cluster_id}. Refreshing displayed crops.")
            self.display_crops(selected_cluster_id)
        else:
            # If no cluster is selected, clear the displayed crops
            logging.warning("No cluster is currently selected. Clearing displayed crops.")
            self.view.display_sampled_crops([])

    @pyqtSlot()
    def on_export_annotations(self):
        """
        Handles exporting labeled annotations to a JSON file.
        """
        self.export_annotations()

    @pyqtSlot()
    def on_restore_autosave_requested(self):
        """
        Handles the manual restoration of an autosave file.
        Prompts the user to select an autosave file to restore.
        """
        autosave_files = self.get_autosave_files()
        if not autosave_files:
            QMessageBox.information(self.view, "No Autosave Found", "There are no autosave files to restore.")
            return

        # Use QFileDialog to let the user select an autosave file
        options = QFileDialog.Options()
        # Set the initial directory to TEMP_DIR
        autosave_file, _ = QFileDialog.getOpenFileName(
            self.view,
            "Select Autosave File to Restore",
            TEMP_DIR,
            "Autosave Files (project_autosave*.json);;All Files (*)",
            options=options
        )

        if autosave_file:
            self.load_project_state(autosave_file)
        else:
            logging.info("User canceled the restore autosave action.")

    def load_project_state(self, project_file: str):
        """
        Loads the project state from a saved file to resume the session.
        """
        if not os.path.exists(project_file):
            logging.info(f"No project file found at {project_file} to load.")
            QMessageBox.warning(self.view, "Load Project", f"No project file found at {project_file}.")
            return

        try:
            with open(project_file, 'r') as f:
                project_state = json.load(f)

            self.clusters = {}
            for filename, annotations in project_state.items():
                for annotation_data in annotations:
                    anno = Annotation.from_dict(annotation_data)
                    self.clusters.setdefault(anno.cluster_id, []).append(anno)

            logging.info(f"Project state loaded from {project_file}")

            # Update the GUI
            cluster_info = self.generate_cluster_info()
            self.view.populate_cluster_selection(cluster_info, selected_cluster_id=None)
            # Automatically display the crops of the first cluster if available
            first_cluster_id = next(iter(self.clusters), None)
            if first_cluster_id is not None:
                self.display_crops(cluster_id=first_cluster_id)
            self.view.hide_progress_bar()

            QMessageBox.information(self.view, "Project Loaded", f"Project state loaded from {project_file}")

        except (IOError, json.JSONDecodeError, KeyError, ValueError) as e:
            logging.error(f"Failed to load project state from {project_file}: {e}")
            QMessageBox.critical(self.view, "Error", f"Failed to load project state: {e}")
