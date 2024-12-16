import logging
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage

from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.ImageProcessor import ImageProcessor
from GUI.workers.ImageProcessingWorker import ImageProcessingWorker


class ImageAnnotator:
    """
    Handles drawing annotations (circle and crosshair) on images.
    """

    def __init__(self, circle_color: QColor = QColor(0, 255, 0), circle_width: int = 5,
                 crosshair_color: QColor = QColor(0, 0, 0), crosshair_width: int = 2):  # Crosshair is now black
        self.circle_color = circle_color
        self.circle_width = circle_width
        self.crosshair_color = crosshair_color
        self.crosshair_width = crosshair_width

    def draw_annotation(self, image: QImage, coord_pos: Tuple[int, int]) -> QImage:
        """
        Draws an annotation (circle and crosshair) on the provided QImage at the specified position.

        :param image: The QImage to draw on.
        :param coord_pos: The (x, y) position to draw the annotation.
        :return: The modified QImage.
        """
        annotated_image = image.copy()
        painter = QPainter(annotated_image)
        painter.setRenderHint(QPainter.Antialiasing)

        pos_x_zoomed, pos_y_zoomed = coord_pos

        # Ensure the position is within the image bounds
        pos_x_zoomed = min(max(pos_x_zoomed, 0), annotated_image.width() - 1)
        pos_y_zoomed = min(max(pos_y_zoomed, 0), annotated_image.height() - 1)

        center = QPoint(int(pos_x_zoomed), int(pos_y_zoomed))

        # Define a radius for the circle
        radius = max(8, min(annotated_image.width(), annotated_image.height()) // 40)

        # Draw the circle
        self.draw_circle(painter, center, radius)

        # Draw the crosshair with gap
        image_size = (annotated_image.width(), annotated_image.height())
        self.draw_crosshair_with_gap(painter, center, radius + self.circle_width // 2, image_size)

        painter.end()

        return annotated_image

    def draw_circle(self, painter, center: QPoint, radius: int):
        """
        Draws an unfilled circle at the specified center with the given radius.

        :param painter: QPainter object to draw with.
        :param center: Center point of the circle.
        :param radius: Radius of the circle.
        """
        pen = QPen(self.circle_color)
        pen.setWidth(self.circle_width)
        painter.setPen(pen)
        painter.drawEllipse(center, radius, radius)

    def draw_crosshair_with_gap(self, painter, center: QPoint, gap_radius: int, image_size: Tuple[int, int]):
        """
        Draws crosshairs with a gap at the specified center point.

        :param painter: QPainter object to draw with.
        :param center: Center point of the crosshair (and circle).
        :param gap_radius: Radius around the center where the crosshair is not drawn (to create the gap).
        :param image_size: Tuple (width, height) of the image.
        """
        width_img, height_img = image_size
        pen = QPen(self.crosshair_color)  # Use the black color for the crosshair
        pen.setWidth(self.crosshair_width)
        painter.setPen(pen)

        # Vertical line
        # From top to (center_y - gap_radius)
        painter.drawLine(center.x(), 0, center.x(), center.y() - gap_radius)
        # From (center_y + gap_radius) to bottom
        painter.drawLine(center.x(), center.y() + gap_radius, center.x(), height_img)

        # Horizontal line
        # From left to (center_x - gap_radius)
        painter.drawLine(0, center.y(), center.x() - gap_radius, center.y())
        # From (center_x + gap_radius) to right
        painter.drawLine(center.x() + gap_radius, center.y(), width_img, center.y())


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
        self.annotator = ImageAnnotator()  # Create an instance of ImageAnnotator

        self.clusters = {}  # Initialize clusters as an empty dictionary
        self.cluster_ids = []  # Initialize cluster_ids as an empty list

        # Worker and thread for image processing
        self.image_worker: Optional[ImageProcessingWorker] = None
        self.image_thread: Optional[QThread] = None

        # For preloading adjacent clusters
        self.prefetch_workers: Dict[int, Tuple[ImageProcessingWorker, QThread]] = {}
        self.prefetching_clusters: set = set()

        self.crops_per_cluster = 10  # Default value, can be updated as needed

    def set_clusters(self, clusters: Dict[int, List[Annotation]]):
        self.clusters = clusters
        self.cluster_ids = list(clusters.keys())

    def set_crops_per_cluster(self, num_crops: int, current_cluster_id: Optional[int] = None):
        """
        Sets the number of crops to sample per cluster and refreshes the view.

        :param num_crops: The number of crops to sample.
        :param current_cluster_id: The ID of the currently selected cluster, if available.
        """
        self.crops_per_cluster = num_crops
        logging.info(f"Crops per cluster set to {num_crops}.")
        self.refresh_current_cluster(current_cluster_id)

    def display_crops(self, annotations: List[Annotation], cluster_id: int):
        """
        Selects the top 'n' annotations from a single cluster to display as zoomed-in crops.
        The annotations are first sorted by their uncertainty from highest to lowest.
        """
        if self.loading_images:
            logging.info("Aborting previous image loading process.")
            self.cancel_image_loading()

        self.loading_images = True
        self.crop_loading_started.emit()

        # Sort annotations by uncertainty (highest to lowest)
        annotations = sorted(annotations, key=lambda anno: anno.uncertainty, reverse=True)

        # Select the top 'n' annotations within the cluster
        num_samples = min(self.crops_per_cluster, len(annotations))
        selected_annotations = annotations[:num_samples]
        logging.debug(
            f"Selected top {len(selected_annotations)} annotations from cluster {cluster_id} by highest uncertainty."
        )

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

    def refresh_current_cluster(self, current_cluster_id: Optional[int]):
        """
        Refreshes the crops displayed for the currently selected cluster.

        :param current_cluster_id: The ID of the currently selected cluster.
        """
        if current_cluster_id is not None and current_cluster_id in self.clusters:
            annotations = self.clusters[current_cluster_id]
            self.display_crops(annotations, current_cluster_id)
        else:
            logging.warning(f"Invalid or no current cluster ID provided: {current_cluster_id}")

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

    def start_image_processing_worker(
            self,
            annotations_to_process: List[Annotation],
            processed_annotations: List[dict]
    ):
        """
        Starts the ImageProcessingWorker in a separate thread to process uncached annotations.

        :param annotations_to_process: List of uncached Annotation objects to process.
        :param processed_annotations: List of already processed annotations.
        """
        self.image_worker = ImageProcessingWorker(
            sampled_annotations=annotations_to_process,
            image_data_model=self.model,  # Pass the existing ImageDataModel instance
            image_processor=self.image_processor
        )
        self.image_thread = QThread(parent=self)
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
            q_image = self.annotator.draw_annotation(q_image, coord_pos)
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
            image_data_model=self.model,  # Pass the existing ImageDataModel instance
            image_processor=self.image_processor
        )
        prefetch_thread = QThread(parent=self)
        prefetch_worker.moveToThread(prefetch_thread)
        prefetch_thread.started.connect(prefetch_worker.process_images)
        prefetch_worker.processing_finished.connect(
            lambda new_annotations: self.on_prefetch_finished(cluster_id)
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
