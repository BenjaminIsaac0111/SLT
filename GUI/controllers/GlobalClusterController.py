# controllers/GlobalClusterController.py

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QTimer
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from sklearn.cluster import AgglomerativeClustering

from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.ImageProcessor import ImageProcessor
from GUI.models.UncertaintyRegionSelector import UncertaintyRegionSelector
from GUI.views.ClusteredCropsView import ClusteredCropsView


class ClusteringWorker(QThread):
    """
    Worker thread for performing global clustering to keep the UI responsive.
    """
    clustering_finished = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)

    def __init__(self, model: ImageDataModel, labels_acquirer: UncertaintyRegionSelector, parent=None):
        super().__init__(parent)
        self.model = model
        self.labels_acquirer = labels_acquirer

    def run(self):
        """
        Executes the clustering process.
        """
        all_annotations = []  # List of dicts with image_index, coord, logit_features
        total_images = self.model.get_number_of_images() // 16
        logging.info(f"Starting clustering on {total_images} images.")

        for idx in range(total_images):
            data = self.model.get_image_data(idx)
            uncertainty_map = data.get('uncertainty', None)
            logits = data.get('logits', None)

            if uncertainty_map is None or logits is None:
                logging.warning(f"Missing data for image index {idx}. Skipping.")
                continue

            logit_features, dbscan_coords = self.labels_acquirer.generate_point_labels(
                uncertainty_map=uncertainty_map,
                logits=logits
            )

            for coord, logit_feature in zip(dbscan_coords, logit_features):
                annotation = {
                    'image_index': idx,
                    'coord': coord,
                    'logit_features': logit_feature
                }
                all_annotations.append(annotation)

            # Emit progress update
            progress = int(((idx + 1) / total_images) * 100)
            self.progress_updated.emit(progress)
            logging.debug(f"Processed image {idx + 1}/{total_images}. Progress: {progress}%")

        logging.info(f"Total annotations collected: {len(all_annotations)}")

        if not all_annotations:
            logging.warning("No annotations found across all images.")
            self.clustering_finished.emit({})
            return

        # Extract logit features for clustering
        try:
            logit_matrix = np.array([anno['logit_features'] for anno in all_annotations])
            logging.debug(f"Logit matrix shape: {logit_matrix.shape}")
        except Exception as e:
            logging.error(f"Error creating logit matrix: {e}")
            self.clustering_finished.emit({})
            return

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

        # Map clusters to annotations
        clusters = {}
        for label, annotation in zip(labels, all_annotations):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(annotation)

        logging.info(f"Global clustering complete. Number of clusters: {len(clusters)}")
        self.clustering_finished.emit(clusters)


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

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.process_single_annotation, anno): anno for anno in self.sampled_annotations}

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    sampled_crops.append(result)

        self.processing_finished.emit(sampled_crops)

    def process_single_annotation(self, anno):
        image_data = self.model.get_image_data(anno['image_index'])
        image_array = image_data.get('image', None)
        coord = anno['coord']
        if image_array is None:
            return None

        q_crop, coord_pos = self.image_processor.extract_crop(
            image_array, coord, crop_size=256, zoom_factor=2
        )
        if q_crop is None:
            return None

        return {
            'cluster_id': anno['cluster_id'],
            'image_index': anno['image_index'],
            'coord': coord,
            'crop': q_crop,
            'coord_pos': coord_pos
        }


class GlobalClusterController(QObject):
    """
    GlobalClusterController handles global clustering across all images and manages the presentation of sampled crops.
    """

    clustering_started = pyqtSignal()
    clustering_progress = pyqtSignal(int)
    clusters_ready = pyqtSignal(dict)  # Signal to notify when clusters are ready

    def __init__(self, model: ImageDataModel, view: ClusteredCropsView):
        super().__init__()
        self.model = model
        self.view = view
        self.image_processor = ImageProcessor()
        self.region_selector = UncertaintyRegionSelector()

        # Cluster data structure: {cluster_id: [annotations]}
        self.clusters = {}

        # Sampling parameters
        self.crops_per_cluster = 10  # Number of crops per selected cluster

        # Debounce timer for sampling parameter changes
        self.debounce_timer = QTimer()
        self.debounce_timer.setInterval(300)  # 300 milliseconds debounce interval
        self.debounce_timer.setSingleShot(True)  # Timer will fire only once per activation
        self.debounce_timer.timeout.connect(self.handle_sampling_parameters_changed)

        # Connect view signals
        self.connect_signals()

    def connect_signals(self):
        """
        Connect signals from the view to controller methods.
        """
        self.view.request_clustering.connect(self.start_clustering)
        self.view.sample_cluster.connect(self.on_sample_cluster)
        self.view.sampling_parameters_changed.connect(self.on_sampling_parameters_changed)
        self.clustering_progress.connect(self.view.update_progress)
        self.clusters_ready.connect(self.on_clusters_ready)

    @pyqtSlot()
    def start_clustering(self):
        """
        Initiates the global clustering process.
        """
        logging.info("Clustering process initiated by user.")
        self.clustering_started.emit()
        self.view.reset_progress()

        # Initialize and start the clustering worker thread
        self.worker = ClusteringWorker(model=self.model, labels_acquirer=self.region_selector)
        self.worker.progress_updated.connect(self.clustering_progress.emit)
        self.worker.clustering_finished.connect(self.clusters_ready.emit)
        self.worker.start()

    @pyqtSlot(dict)
    def on_clusters_ready(self, clusters: Dict[int, List[Dict]]):
        """
        Handles the completion of the clustering process.
        """
        self.clusters = clusters
        logging.info(f"Clustering finished with {len(clusters)} clusters.")
        # Populate the cluster selection ComboBox in the view
        cluster_ids = list(self.clusters.keys())
        self.view.populate_cluster_selection(cluster_ids)
        self.view.hide_progress_bar()  # Hide the progress bar after clustering is done

    @pyqtSlot(int, int)
    def on_sampling_parameters_changed(self, _, crops_per_cluster: int):
        """
        Handles changes in sampling parameters (number of crops per cluster).
        Implements debouncing to prevent excessive sampling.

        :param _: Placeholder for cluster_id which is no longer used.
        :param crops_per_cluster: Number of crops to sample per cluster.
        """
        logging.info(f"Sampling parameters updated: {crops_per_cluster} crops per cluster.")
        self.crops_per_cluster = crops_per_cluster

        # Restart the debounce timer
        if self.debounce_timer.isActive():
            self.debounce_timer.stop()
        self.debounce_timer.start()

    @pyqtSlot()
    def handle_sampling_parameters_changed(self):
        """
        Handles the sampling parameters after debouncing.
        """
        logging.info("Handling debounced sampling parameter changes.")
        # Assuming there's a currently selected cluster
        selected_cluster_id = self.view.get_selected_cluster_id()
        if selected_cluster_id is not None:
            self.sample_and_display_crops(selected_cluster_id)
        else:
            logging.warning("No cluster is currently selected.")
            self.view.display_sampled_crops([])

    def sample_and_display_crops(self, cluster_id: int):
        """
        Samples a subset of annotations from a single cluster to display as zoomed-in crops.

        :param cluster_id: The ID of the cluster to sample from.
        """
        if cluster_id not in self.clusters:
            logging.warning(f"Cluster ID {cluster_id} does not exist.")
            self.view.display_sampled_crops([])
            return

        annotations = self.clusters[cluster_id]
        if not annotations:
            logging.warning(f"No annotations found in cluster {cluster_id}.")
            self.view.display_sampled_crops([])
            return

        # Sample annotations within the cluster
        num_samples = min(self.crops_per_cluster, len(annotations))
        try:
            sampled_annotations = np.random.choice(annotations, size=num_samples, replace=False)
            logging.debug(f"Sampled {len(sampled_annotations)} annotations from cluster {cluster_id}.")
        except ValueError as e:
            logging.error(f"Error sampling annotations: {e}")
            self.view.display_sampled_crops([])
            return

        # Prepare annotations for processing
        processed_annotations = []
        for anno in sampled_annotations:
            processed_annotations.append({
                'cluster_id': cluster_id,
                'image_index': anno['image_index'],
                'coord': anno['coord']
            })

        # Initialize Image Processing Worker
        self.image_worker = ImageProcessingWorker(
            sampled_annotations=processed_annotations,
            model=self.model,
            image_processor=self.image_processor
        )
        self.image_thread = QThread()
        self.image_worker.moveToThread(self.image_thread)
        self.image_thread.started.connect(self.image_worker.process_images)
        self.image_worker.processing_finished.connect(self.on_image_processing_finished)
        self.image_worker.processing_finished.connect(self.image_thread.quit)
        self.image_worker.processing_finished.connect(self.image_worker.deleteLater)
        self.image_thread.finished.connect(self.image_thread.deleteLater)
        self.image_thread.start()

    @pyqtSlot(list)
    def on_image_processing_finished(self, sampled_crops):
        """
        Receives processed crops and updates the view.
        """
        logging.info(f"Received {len(sampled_crops)} sampled crops from processing worker.")
        self.view.display_sampled_crops(sampled_crops)
        logging.info(f"Displayed {len(sampled_crops)} sampled crops.")

    @pyqtSlot(int)
    def on_sample_cluster(self, cluster_id: int):
        """
        Handles requests to display crops from a specific cluster.

        :param cluster_id: The ID of the cluster to sample.
        """
        logging.info(f"Sampling crops from cluster {cluster_id}.")
        self.sample_and_display_crops(cluster_id)
