import logging
from typing import Optional, Dict, Tuple, List

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor

from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.ImageProcessor import ImageProcessor
from GUI.workers.ImageProcessingWorker import ImageProcessingWorker


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

        self.crops_per_cluster = 8  # Default value, can be updated as needed

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
