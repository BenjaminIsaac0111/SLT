import gzip
import json
import logging
import os
import tempfile
from collections import OrderedDict
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QTimer, QPoint, QCoreApplication, QEventLoop
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.ImageProcessor import ImageProcessor
from GUI.models.UncertaintyRegionSelector import UncertaintyRegionSelector
from GUI.views.ClusteredCropsView import ClusteredCropsView
from GUI.workers.AutosaveWorker import AutosaveWorker
from GUI.workers.ClusteringWorker import ClusteringWorker
from GUI.workers.ImageProcessingWorker import ImageProcessingWorker

TEMP_DIR = os.path.join(tempfile.gettempdir(), 'my_application_temp')
os.makedirs(TEMP_DIR, exist_ok=True)


class ProjectStateController(QObject):
    """
    ProjectStateController manages saving and loading of the project state.
    It handles autosaving, restoring from autosave files, and maintains versioned backups.
    It communicates with GlobalClusterController via signals and slots.
    """

    # Signals to communicate with other controllers or the view
    autosave_finished = pyqtSignal(bool)
    project_loaded = pyqtSignal(dict)
    project_saved = pyqtSignal(str)
    save_failed = pyqtSignal(str)
    load_failed = pyqtSignal(str)

    def __init__(self, model: ImageDataModel):
        """
        Initializes the ProjectStateController.

        :param model: An instance of the ImageDataModel.
        """
        super().__init__()
        self.model = model
        self.is_saving = False
        self.current_save_path: Optional[str] = None

        # Initialize autosave worker and thread
        self.autosave_worker = AutosaveWorker()
        self.autosave_thread = QThread()
        self.autosave_worker.moveToThread(self.autosave_thread)
        self.autosave_worker.save_finished.connect(self.on_autosave_finished)
        self.autosave_thread.start()

        # Autosave file path within the dedicated temp directory
        self.temp_file_path = os.path.join(TEMP_DIR, 'project_autosave.json.gz')
        logging.info(f"Temporary autosave file path: {self.temp_file_path}")

    def set_current_save_path(self, file_path: str):
        """
        Sets the current file path for saving the project.

        :param file_path: The file path where the project will be saved.
        """
        self.current_save_path = file_path
        logging.info(f"Current save path set to: {file_path}")

    def get_current_save_path(self) -> Optional[str]:
        """
        Returns the current file path where the project is saved.

        :return: The current save path.
        """
        return self.current_save_path

    def autosave_project_state(self, project_state: dict):
        """
        Autosaves the current project state to a versioned backup file.

        :param project_state: The current state of the project.
        """
        if self.is_saving:
            logging.info("Autosave already in progress, skipping this autosave.")
            return  # Skip if a save is already in progress

        if not project_state.get('annotations'):
            logging.info("No annotations to save. Skipping autosave.")
            return  # No data to save

        self.is_saving = True

        # Create a versioned backup path
        versioned_backup_path = self.get_versioned_backup_path(self.temp_file_path, max_backups=5)

        # Send save request to worker
        self.autosave_worker.save_project_state_signal.emit(project_state, versioned_backup_path)

    @pyqtSlot(bool)
    def on_autosave_finished(self, success: bool):
        """
        Handles completion of the autosave operation.

        :param success: True if autosave was successful, False otherwise.
        """
        self.is_saving = False  # Reset the saving flag
        self.autosave_finished.emit(success)

    def save_project_state(self, project_state: dict, file_path: str):
        """
        Saves the current project state to the specified file path.

        :param project_state: The current state of the project.
        :param file_path: The file path where the project will be saved.
        """
        try:
            with gzip.open(file_path, 'wt') as f:
                json.dump(project_state, f, indent=4)
            logging.info(f"Project state saved to {file_path}")
            self.project_saved.emit(file_path)
        except (TypeError, IOError) as e:
            logging.error(f"Failed to save project state: {e}")
            self.save_failed.emit(str(e))

    def load_project_state(self, project_file: str):
        """
        Loads the project state from a saved file to resume the session.

        :param project_file: The file path of the project to load.
        """
        if not os.path.exists(project_file):
            logging.error(f"No project file found at {project_file} to load.")
            self.load_failed.emit(f"No project file found at {project_file}.")
            return

        try:
            with gzip.open(project_file, 'rt') as f:
                project_state = json.load(f)
            logging.info(f"Project state loaded from {project_file}")
            self.project_loaded.emit(project_state)
        except (IOError, json.JSONDecodeError, KeyError, ValueError) as e:
            logging.error(f"Failed to load project state from {project_file}: {e}")
            self.load_failed.emit(str(e))

    def get_versioned_backup_path(self, base_path: str, max_backups: int = 10) -> str:
        """
        Generates a versioned backup path by adding a timestamp.
        Deletes older backups if they exceed max_backups.

        :param base_path: The base file path.
        :param max_backups: Maximum number of backup files to keep.
        :return: The versioned backup file path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.basename(base_path)}_{timestamp}.json.gz"
        backup_path = os.path.join(TEMP_DIR, backup_filename)

        # Remove old backups if exceeding max_backups
        all_backups = sorted([f for f in os.listdir(TEMP_DIR)
                              if f.startswith(os.path.basename(base_path)) and f.endswith(".json.gz")])

        while len(all_backups) > max_backups:
            old_backup = all_backups.pop(0)  # Remove the oldest file
            os.remove(os.path.join(TEMP_DIR, old_backup))
            logging.info(f"Deleted old backup file: {old_backup}")

        return backup_path

    def get_latest_autosave_file(self) -> Optional[str]:
        """
        Finds the most recent autosave file in the TEMP_DIR.

        :return: The path to the latest autosave file, or None if none exist.
        """
        autosave_files = [f for f in os.listdir(TEMP_DIR)
                          if f.startswith('project_autosave') and f.endswith('.json.gz')]
        if not autosave_files:
            return None  # No autosave files found

        # Sort files by modification time in descending order
        autosave_files.sort(key=lambda f: os.path.getmtime(os.path.join(TEMP_DIR, f)), reverse=True)
        latest_autosave = autosave_files[0]
        return os.path.join(TEMP_DIR, latest_autosave)

    def get_autosave_files(self) -> List[str]:
        """
        Returns a list of available autosave files in TEMP_DIR.

        :return: A list of paths to autosave files.
        """
        autosave_files = [f for f in os.listdir(TEMP_DIR)
                          if f.startswith('project_autosave') and f.endswith('.json.gz')]
        if not autosave_files:
            return []
        # Sort files by modification time in descending order
        autosave_files.sort(key=lambda f: os.path.getmtime(os.path.join(TEMP_DIR, f)), reverse=True)
        # Return full paths
        autosave_files_full = [os.path.join(TEMP_DIR, f) for f in autosave_files]
        return autosave_files_full

    def cleanup(self):
        """
        Cleans up the autosave worker and thread before application exit.
        """
        logging.info("Cleaning up autosave thread before application exit.")
        if self.is_saving:
            logging.info("Autosave in progress. Waiting for it to finish before quitting.")
            loop = QEventLoop()
            self.autosave_worker.save_finished.connect(loop.quit)
            loop.exec_()  # This will block until `save_finished` is emitted
        # Now, safely quit the autosave thread
        self.autosave_thread.quit()
        self.autosave_thread.wait()
        logging.info("Autosave thread terminated.")


class ImageProcessingController(QObject):
    """
    ImageProcessingController handles image loading and processing operations independently.
    It communicates with GlobalClusterController via signals and slots.
    """

    # Signals to communicate with other controllers or the view
    crops_ready = pyqtSignal(list)
    progress_updated = pyqtSignal(int)
    crop_loading_started = pyqtSignal()
    crop_loading_finished = pyqtSignal()

    def __init__(self, model: ImageDataModel):
        """
        Initializes the ImageProcessingController.

        :param model: An instance of the ImageDataModel.
        """
        super().__init__()
        self.model = model
        self.image_processor = ImageProcessor()
        self.loading_images = False

        self.clusters = {}  # Initialize clusters as an empty dictionary
        self.cluster_ids = []  # Initialize cluster_ids as an empty list

        # Worker and thread for image processing
        self.image_worker: Optional[ImageProcessingWorker] = None
        self.image_thread: Optional[QThread] = None

        # For preloading adjacent clusters
        self.prefetch_workers: Dict[int, Tuple[ImageProcessingWorker, QThread]] = {}
        self.prefetching_clusters: set = set()

        self.crops_per_cluster = 100  # Default value, can be updated as needed

    def set_clusters(self, clusters: Dict[int, List[Annotation]]):
        self.clusters = clusters
        self.cluster_ids = list(clusters.keys())

    def set_crops_per_cluster(self, num_crops: int):
        """
        Sets the number of crops to sample per cluster.

        :param num_crops: The number of crops to sample.
        """
        self.crops_per_cluster = num_crops
        logging.info(f"Crops per cluster set to {num_crops}.")

    def display_crops(self, annotations: List[Annotation], cluster_id: int):
        """
        Selects the top 'n' annotations from a single cluster to display as zoomed-in crops.

        :param annotations: List of Annotation objects in the cluster.
        :param cluster_id: The ID of the cluster.
        """
        if self.loading_images:
            logging.info("Aborting previous image loading process.")
            self.cancel_image_loading()

        self.loading_images = True
        self.crop_loading_started.emit()

        # Select the top 'n' annotations within the cluster
        num_samples = min(self.crops_per_cluster, len(annotations))
        selected_annotations = annotations[:num_samples]
        logging.debug(f"Selected top {len(selected_annotations)} annotations from cluster {cluster_id}.")

        # Check cache and separate cached and uncached annotations
        processed_annotations = []
        annotations_to_process = []

        for anno in selected_annotations:
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

        if annotations_to_process:
            # Start the image processing worker for uncached annotations
            self.start_image_processing_worker(annotations_to_process, processed_annotations)
        else:
            # All data is cached; proceed to display
            self.display_processed_annotations(processed_annotations)

        # Optionally, preload adjacent clusters if needed
        self.preload_adjacent_clusters(cluster_id)

    def preload_adjacent_clusters(self, cluster_id: int):
        """
        Preloads images for adjacent clusters.

        :param cluster_id: The current cluster ID.
        """
        if not self.cluster_ids:
            logging.warning("No cluster IDs available. Ensure clusters are set before preloading.")
            return

        if cluster_id not in self.cluster_ids:
            logging.warning(f"Cluster ID {cluster_id} not found in cluster IDs.")
            return

        current_index = self.cluster_ids.index(cluster_id)

        adjacent_indices = []
        if current_index > 0:
            adjacent_indices.append(current_index - 1)
        if current_index < len(self.cluster_ids) - 1:
            adjacent_indices.append(current_index + 1)

        for index in adjacent_indices:
            adjacent_cluster_id = self.cluster_ids[index]
            if not self.is_cluster_cached(adjacent_cluster_id):
                self.start_background_loading(adjacent_cluster_id)

    def cancel_image_loading(self):
        """
        Cancels any ongoing image loading process.
        """
        if self.image_thread:
            try:
                if self.image_thread.isRunning():
                    self.image_thread.quit()
                    self.image_thread.wait()
                    logging.info("Image loading process canceled.")
            except RuntimeError as e:
                logging.warning(f"Attempted to cancel a thread that was already deleted: {e}")
            finally:
                # Ensure the thread reference is cleared
                self.image_thread = None
        self.loading_images = False

    def start_image_processing_worker(self, annotations_to_process: List[Annotation],
                                      processed_annotations: List[dict]):
        """
        Starts the ImageProcessingWorker in a separate thread to process uncached annotations.

        :param annotations_to_process: List of uncached Annotation objects to process.
        :param processed_annotations: List of already processed annotations.
        """
        self.image_worker = ImageProcessingWorker(
            sampled_annotations=annotations_to_process,
            hdf5_file_path=self.model.hdf5_file_path,
            image_processor=self.image_processor
        )
        self.image_thread = QThread()
        self.image_worker.moveToThread(self.image_thread)
        self.image_thread.started.connect(self.image_worker.process_images)

        # Use a lambda to pass both cached and newly processed annotations
        self.image_worker.processing_finished.connect(
            lambda new_annotations: self.on_image_processing_finished(
                processed_annotations + new_annotations
            )
        )
        self.image_worker.processing_finished.connect(self.image_thread.quit)
        self.image_worker.processing_finished.connect(self.image_worker.deleteLater)
        self.image_worker.progress_updated.connect(self.progress_updated.emit)
        self.image_thread.finished.connect(self.image_thread.deleteLater)
        self.image_thread.start()

    @pyqtSlot(list)
    def on_image_processing_finished(self, total_annotations: List[dict]):
        """
        Handles the completion of image processing and displays the processed crops.

        :param total_annotations: List of dictionaries containing annotations and processed images.
        """
        self.display_processed_annotations(total_annotations)

    def display_processed_annotations(self, annotations: List[dict]):
        """
        Converts processed annotations to QPixmap and emits the crops_ready signal.

        :param annotations: List of dictionaries with processed annotations.
        """
        sampled_crops = []
        for data in annotations:
            annotation = data['annotation']
            processed_crop = data['processed_crop']
            coord_pos = data['coord_pos']
            if processed_crop is None or coord_pos is None:
                logging.warning(f"Processed crop or coord_pos is None for annotation: {annotation}")
                continue
            q_image = self.image_processor.numpy_to_qimage(processed_crop)
            q_image = self.draw_annotation_on_image(q_image, coord_pos)
            q_pixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
            sampled_crops.append({
                'annotation': annotation,
                'processed_crop': q_pixmap,
                'coord_pos': coord_pos
            })
        self.crops_ready.emit(sampled_crops)
        logging.info(f"Displayed {len(sampled_crops)} sampled crops.")
        self.crop_loading_finished.emit()
        self.loading_images = False

    def draw_annotation_on_image(self, q_image, coord_pos: Tuple[int, int]):
        """
        Draws an annotation (circle) on the provided QImage at the specified position.

        :param q_image: The QImage to draw on.
        :param coord_pos: The (x, y) position to draw the annotation.
        :return: The modified QImage.
        """
        zoomed_qimage = q_image.copy()
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

    def is_cluster_cached(self, cluster_id: int) -> bool:
        """
        Checks if all annotations in a cluster are cached.

        :param cluster_id: The cluster ID to check.
        :return: True if all annotations are cached, False otherwise.
        """
        annotations = self.clusters.get(cluster_id, [])
        for anno in annotations[:self.crops_per_cluster]:
            cache_key = (anno.image_index, tuple(anno.coord))
            if not self.image_processor.cache.get(cache_key):
                return False
        return True

    def start_background_loading(self, cluster_id: int):
        """
        Starts preloading images for a cluster in the background.

        :param cluster_id: The cluster ID to preload.
        """
        if cluster_id in self.prefetching_clusters:
            logging.debug(f"Already preloading cluster {cluster_id}.")
            return

        self.prefetching_clusters.add(cluster_id)

        annotations = self.clusters.get(cluster_id, [])
        num_samples = min(self.crops_per_cluster, len(annotations))
        selected_annotations = annotations[:num_samples]

        if not selected_annotations:
            logging.debug(f"No annotations to preload for cluster {cluster_id}.")
            self.prefetching_clusters.discard(cluster_id)
            return

        prefetch_worker = ImageProcessingWorker(
            sampled_annotations=selected_annotations,
            hdf5_file_path=self.model.hdf5_file_path,
            image_processor=self.image_processor
        )
        prefetch_thread = QThread()
        prefetch_worker.moveToThread(prefetch_thread)
        prefetch_thread.started.connect(prefetch_worker.process_images)
        prefetch_worker.processing_finished.connect(
            lambda _: self.on_prefetch_finished(cluster_id)
        )
        prefetch_worker.processing_finished.connect(prefetch_thread.quit)
        prefetch_worker.processing_finished.connect(prefetch_worker.deleteLater)
        prefetch_thread.finished.connect(prefetch_thread.deleteLater)
        prefetch_thread.start()

        self.prefetch_workers[cluster_id] = (prefetch_worker, prefetch_thread)

    @pyqtSlot()
    def on_prefetch_finished(self, cluster_id: int):
        """
        Handles the completion of background preloading for a cluster.

        :param cluster_id: The cluster ID that was preloaded.
        """
        self.prefetching_clusters.discard(cluster_id)
        self.prefetch_workers.pop(cluster_id, None)
        logging.info(f"Prefetching completed for cluster {cluster_id}.")

    def cleanup(self):
        """
        Cleans up the image processing worker and thread.
        """
        if self.image_thread:
            try:
                self.cancel_image_loading()
            except RuntimeError as e:
                logging.warning(f"Thread cleanup encountered an issue: {e}")
            finally:
                self.image_thread = None  # Clear the reference after cleanup

        # Clean up prefetch workers
        for cluster_id, (worker, thread) in self.prefetch_workers.items():
            if thread:
                try:
                    if thread.isRunning():
                        thread.quit()
                        thread.wait()
                        logging.info(f"Prefetch thread for cluster {cluster_id} terminated during cleanup.")
                except RuntimeError as e:
                    logging.warning(f"Prefetch thread for cluster {cluster_id} already deleted: {e}")
                finally:
                    # Clear the worker and thread references
                    self.prefetch_workers[cluster_id] = (None, None)

        self.prefetch_workers.clear()


class ClusteringController(QObject):
    """
    ClusteringController handles clustering operations independently.
    It communicates with GlobalClusterController via signals and slots.
    """

    # Signals to communicate with other controllers or the view
    clustering_started = pyqtSignal()
    clustering_progress = pyqtSignal(int)
    clusters_ready = pyqtSignal(dict)
    labeling_statistics_updated = pyqtSignal(dict)

    def __init__(self, model, region_selector):
        """
        Initializes the ClusteringController.

        :param model: An instance of the ImageDataModel.
        :param region_selector: An instance of UncertaintyRegionSelector.
        """
        super().__init__()
        self.model = model
        self.region_selector = region_selector

        # Initialize attributes
        self.clusters: Dict[int, List[Annotation]] = {}
        self.cluster_labels: Dict[int, str] = {}
        self.crops_per_cluster = 100  # Default value, can be updated as needed

        # Worker and thread for clustering
        self.clustering_worker: Optional[ClusteringWorker] = None
        self.clustering_thread: Optional[QThread] = None

    @pyqtSlot()
    def start_clustering(self):
        """
        Initiates the clustering process by starting the ClusteringWorker in a separate thread.
        """
        logging.info("Clustering process initiated.")
        self.clustering_started.emit()

        # Initialize and configure the ClusteringWorker
        self.clustering_worker = ClusteringWorker(
            hdf5_file_path=self.model.hdf5_file_path,
            labels_acquirer=self.region_selector
        )
        self.clustering_thread = QThread()
        self.clustering_worker.moveToThread(self.clustering_thread)

        # Connect signals and slots
        self.clustering_thread.started.connect(self.clustering_worker.run)
        self.clustering_worker.progress_updated.connect(self.clustering_progress.emit)
        self.clustering_worker.clustering_finished.connect(self.on_clustering_finished)
        self.clustering_worker.clustering_finished.connect(self.clustering_thread.quit)
        self.clustering_worker.finished.connect(self.clustering_worker.deleteLater)
        self.clustering_thread.finished.connect(self.clustering_thread.deleteLater)

        # Start the clustering thread
        self.clustering_thread.start()

    @pyqtSlot(list)
    def on_clustering_finished(self, annotations: List[Annotation]):
        """
        Handles the completion of the clustering process.

        :param annotations: A list of Annotation objects resulting from clustering.
        """
        # Organize annotations by cluster_id
        self.clusters = self.group_annotations_by_cluster(annotations)
        logging.info(f"Clustering finished with {len(self.clusters)} clusters.")

        # Emit the clusters_ready signal with the clusters data
        self.clusters_ready.emit(self.clusters)

        # Compute and emit labeling statistics
        self.compute_labeling_statistics()

    def group_annotations_by_cluster(self, annotations: List[Annotation]) -> Dict[int, List[Annotation]]:
        """
        Groups annotations by their cluster IDs.

        :param annotations: A list of Annotation objects.
        :return: A dictionary mapping cluster IDs to lists of annotations.
        """
        clusters = {}
        for annotation in annotations:
            # Ensure the annotation is an instance of Annotation
            if not isinstance(annotation, Annotation):
                annotation = Annotation.from_dict(annotation)
            cluster_id = annotation.cluster_id

            clusters.setdefault(cluster_id, []).append(annotation)
        return clusters

    def compute_labeling_statistics(self):
        """
        Computes labeling statistics such as total annotations, total labeled,
        and class counts, and emits the labeling_statistics_updated signal.
        """
        total_annotations = 0
        total_labeled = 0
        class_counts = {class_id: 0 for class_id in CLASS_COMPONENTS.keys()}
        class_counts[-1] = 0  # Unlabeled
        class_counts[-2] = 0  # Unsure

        for cluster_id, annotations in self.clusters.items():
            for anno in annotations:
                total_annotations += 1
                class_id = anno.class_id if anno.class_id is not None else -1

                if class_id in class_counts:
                    class_counts[class_id] += 1
                else:
                    # Handle unexpected class IDs
                    class_counts[class_id] = 1

                if class_id != -1:
                    total_labeled += 1

        statistics = {
            'total_annotations': total_annotations,
            'total_labeled': total_labeled,
            'class_counts': class_counts
        }
        logging.info(f"Labeling statistics computed: {statistics}")
        self.labeling_statistics_updated.emit(statistics)

    def generate_cluster_info(self) -> Dict[int, dict]:
        """
        Generates a dictionary containing cluster information for the GUI.

        :return: A dictionary mapping cluster IDs to cluster information.
        """
        cluster_info = OrderedDict()
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
        logging.debug(f"Cluster info generated: {cluster_info}")
        return cluster_info

    def update_cluster_labels(self, cluster_id: int, label: str):
        """
        Updates the label for a specific cluster.

        :param cluster_id: The ID of the cluster to update.
        :param label: The new label for the cluster.
        """
        self.cluster_labels[cluster_id] = label
        logging.info(f"Cluster {cluster_id} label updated to '{label}'.")

    def get_clusters(self) -> Dict[int, List[Annotation]]:
        """
        Returns the clusters dictionary.

        :return: The clusters dictionary.
        """
        return self.clusters

    def set_crops_per_cluster(self, num_crops: int):
        """
        Sets the number of crops to sample per cluster.

        :param num_crops: The number of crops to sample.
        """
        self.crops_per_cluster = num_crops
        logging.info(f"Crops per cluster set to {num_crops}.")

    def cleanup(self):
        """
        Cleans up the clustering worker and thread.
        """
        if self.clustering_thread and self.clustering_thread.isRunning():
            self.clustering_thread.quit()
            self.clustering_thread.wait()
            logging.info("Clustering thread terminated during cleanup.")


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

    @pyqtSlot(int)
    def on_sampling_parameters_changed(self, crops_per_cluster: int):
        """
        Handles changes in sampling parameters.
        Implements debouncing to prevent excessive sampling.

        :param crops_per_cluster: The new number of crops per cluster.
        """
        logging.info(f"Sampling parameters updated: {crops_per_cluster} crops per cluster.")
        self.image_processing_controller.set_crops_per_cluster(crops_per_cluster)

        # Restart the debounce timer
        if self.debounce_timer.isActive():
            self.debounce_timer.stop()
        self.debounce_timer.start()

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
