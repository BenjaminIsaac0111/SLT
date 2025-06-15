import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThreadPool, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import BaseImageDataModel
from GUI.models.ImageProcessor import ImageProcessor
from GUI.workers.ImageProcessingWorker import ImageProcessingWorker


class ImageProcessingController(QObject):
    """
    Handles image loading and processing operations, emitting signals to
    communicate progress and results to the view or other controllers.
    """

    # Signals
    crops_ready = pyqtSignal(list)
    progress_updated = pyqtSignal(int)
    crop_loading_started = pyqtSignal()
    crop_loading_finished = pyqtSignal()

    def __init__(self, model: BaseImageDataModel):
        """
        :param model: An ImageDataModel instance for fetching image data.
        """
        super().__init__()
        self.model = model
        self.image_processor = ImageProcessor()

        # Use the global thread pool instead of manual QThreads
        self.threadpool = QThreadPool.globalInstance()

        self.loading_images = False

        self.clusters: Dict[int, List[Annotation]] = {}
        self.cluster_ids: List[int] = []

        # Keep track of clusters we’re currently prefetching to avoid duplicates
        self.prefetching_clusters = set()

    # -------------------------------------------------------------------------
    #                              CLUSTER SETUP
    # -------------------------------------------------------------------------
    def set_clusters(self, clusters: Dict[int, List[Annotation]]):
        self.clusters = clusters
        self.cluster_ids = list(clusters.keys())

    # -------------------------------------------------------------------------
    #                         DISPLAY & PROCESS CROPS
    # -------------------------------------------------------------------------
    def display_crops(self, annotations: List[Annotation]):
        """
        Displays crops for the top 'n' annotations from a single cluster.
        """
        if self.loading_images:
            logging.info("Aborting previous image loading process.")

        self.loading_images = True
        self.crop_loading_started.emit()

        selected = self._sort_annotations_by_uncertainty(annotations)

        processed_annos, uncached_annos = self._split_cached_and_uncached(selected)

        # If anything is uncached, launch a worker; otherwise, just display
        if uncached_annos:
            self._start_processing_worker(uncached_annos, processed_annos)
        else:
            self._display_processed_annotations(processed_annos)

    def refresh_current_cluster(self, current_cluster_id: Optional[int]):
        """
        Reloads crops for the specified cluster, if valid.
        """
        if current_cluster_id is not None and current_cluster_id in self.clusters:
            self.display_crops(self.clusters[current_cluster_id])
        else:
            logging.warning(f"Invalid cluster ID for refresh: {current_cluster_id}")

    # -------------------------------------------------------------------------
    #                           WORKER CREATION
    # -------------------------------------------------------------------------
    def _start_processing_worker(
            self,
            annotations_to_process: List[Annotation],
            processed_annotations: List[dict]
    ):
        """
        Creates and starts a QRunnable worker to process uncached annotations.
        """
        worker = ImageProcessingWorker(
            sampled_annotations=annotations_to_process,
            image_data_model=self.model,
            image_processor=self.image_processor,
            already_processed=processed_annotations
        )
        # Connect the worker's signals
        worker.signals.processing_finished.connect(self.on_image_processing_finished)
        worker.signals.progress_updated.connect(self.progress_updated)

        # Launch it on the thread pool
        self.threadpool.start(worker)

    @pyqtSlot(list)
    def on_image_processing_finished(self, total_annotations: List[dict]):
        """
        Called when the worker finishes processing. Displays final results.
        """
        self._display_processed_annotations(total_annotations)

    # -------------------------------------------------------------------------
    #                         CROP DISPLAY LOGIC
    # -------------------------------------------------------------------------
    def _display_processed_annotations(self, annotations_data: List[dict]):
        """
        Converts the processed numpy arrays into QPixmaps and emits the
        crops_ready signal.
        """
        sampled_crops = []
        for data in annotations_data:
            anno = data['annotation']
            np_image = data['processed_crop']
            coord_pos = data['coord_pos']
            mask_crop = data.get('mask_crop')

            if np_image is None or coord_pos is None:
                logging.warning(f"Missing image or coords for annotation: {anno}")
                continue

            q_pixmap = self._numpy_to_qpixmap(np_image)
            mask_pix = self._numpy_to_qpixmap(mask_crop) if mask_crop is not None else None

            sampled_crops.append({
                'annotation': anno,
                'processed_crop': q_pixmap,
                'coord_pos': coord_pos,
                'mask_pixmap': mask_pix,
            })

        self.crops_ready.emit(sampled_crops)
        logging.info(f"Displayed {len(sampled_crops)} sampled crops.")
        self.crop_loading_finished.emit()
        self.loading_images = False

    @staticmethod
    def _numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
        h, w, _ = arr.shape
        buf = arr.tobytes()  # deep copy, owns memory
        qimg = QImage(buf, w, h, QImage.Format_RGB888)
        qimg._buf = buf  # pin buffer to qimg lifespan
        return QPixmap.fromImage(qimg)

    # -------------------------------------------------------------------------
    #                          PREFETCH LOGIC
    # -------------------------------------------------------------------------
    def preload_next_clusters(self, cluster_id: int):
        """
        Triggers background loading for adjacent clusters.
        """
        if not self.cluster_ids:
            logging.warning("No cluster IDs set; skipping prefetch.")
            return

        if cluster_id not in self.cluster_ids:
            logging.warning(f"Cluster ID {cluster_id} not found in cluster list.")
            return

        index = self.cluster_ids.index(cluster_id)
        neighbors = []
        if index > 0:
            neighbors.append(self.cluster_ids[index - 1])
        if index < len(self.cluster_ids) - 1:
            neighbors.append(self.cluster_ids[index + 1])

        for neighbor_id in neighbors:
            if not self._is_cluster_cached(neighbor_id):
                self.start_background_prefetch(neighbor_id)

    def start_background_prefetch(self, cluster_id: int):
        if cluster_id in self.prefetching_clusters:
            logging.debug(f"Already prefetching cluster {cluster_id}.")
            return

        annos = self.clusters.get(cluster_id, [])
        selected = annos[:min(self.crops_per_cluster, len(annos))]
        if not selected:
            logging.debug(f"No annotations to preload for cluster {cluster_id}.")
            return

        self.prefetching_clusters.add(cluster_id)

        worker = ImageProcessingWorker(
            sampled_annotations=selected,
            image_data_model=self.model,
            image_processor=self.image_processor
        )
        worker.signals.processing_finished.connect(
            lambda _: self._on_prefetch_finished(cluster_id)
        )
        self.threadpool.start(worker)

    @pyqtSlot()
    def _on_prefetch_finished(self, cluster_id: int):
        """
        Marks the cluster as done prefetching.
        """
        self.prefetching_clusters.discard(cluster_id)
        logging.info(f"Prefetch completed for cluster {cluster_id}.")

    # -------------------------------------------------------------------------
    #                           PRIVATE HELPERS
    # -------------------------------------------------------------------------
    @staticmethod
    def _sort_annotations_by_uncertainty(annotations: List[Annotation]) -> List[Annotation]:
        """
        Sorts annotations by their uncertainty in descending order.
        """
        return sorted(annotations, key=lambda a: a.uncertainty, reverse=True)

    def _is_cluster_cached(self, cluster_id: int) -> bool:
        """
        Checks if all required annotations in a cluster are cached.
        """
        annotations = self.clusters.get(cluster_id, [])
        for anno in annotations[:self.crops_per_cluster]:
            if not self.image_processor.cache.get((anno.image_index, tuple(anno.coord))):
                return False
        return True

    def _split_cached_and_uncached(self, annotations: List[Annotation]) -> Tuple[List[dict], List[Annotation]]:
        """
        Splits annotations into already cached (processed) and those that need processing.
        Returns (processed_list, uncached_list).
        """
        processed_list = []
        uncached_list = []
        for anno in annotations:
            cache_key = (anno.image_index, tuple(anno.coord))
            result = self.image_processor.cache.get(cache_key)
            if result:
                np_image, coord_pos = result
                processed_list.append({
                    'annotation': anno,
                    'processed_crop': np_image,
                    'coord_pos': coord_pos
                })
            else:
                uncached_list.append(anno)
        return processed_list, uncached_list

    # -------------------------------------------------------------------------
    #                           CLEANUP (Optional)
    # -------------------------------------------------------------------------
    def cleanup(self):
        """
        If you don't need to forcibly stop tasks in progress, this can be no-op.
        Otherwise, implement logic such as a cooperative stop for the workers.
        """
        self.loading_images = False
        logging.info("ImageProcessingController cleanup done.")
