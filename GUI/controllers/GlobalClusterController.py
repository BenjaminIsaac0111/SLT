# controllers/GlobalClusterController.py

import logging
from typing import List, Tuple, Dict

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
import numpy as np
from PyQt5.QtGui import QPixmap
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

    def __init__(self, model: ImageDataModel, selector: UncertaintyRegionSelector, parent=None):
        super().__init__(parent)
        self.model = model
        self.selector = selector

    def run(self):
        """
        Executes the clustering process.
        """
        all_annotations = []  # List of dicts with image_index, coord, logit_features
        total_images = self.model.get_number_of_images()
        logging.info(f"Starting clustering on {total_images} images.")

        for idx in range(total_images):
            data = self.model.get_image_data(idx)
            uncertainty_map = data['uncertainty']
            logits = data['logits']

            # Use the updated select_regions method which returns (logit_features, dbscan_coords)
            logit_features, dbscan_coords = self.selector.select_regions(
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
        logit_matrix = np.array([anno['logit_features'] for anno in all_annotations])

        # Perform Agglomerative Clustering globally
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.selector.distance_threshold,
            linkage=self.selector.linkage
        )
        clustering.fit(logit_matrix)
        labels = clustering.labels_

        # Map clusters to annotations
        clusters = {}
        for label, annotation in zip(labels, all_annotations):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(annotation)

        logging.info(f"Global clustering complete. Number of clusters: {len(clusters)}")
        self.clustering_finished.emit(clusters)


class GlobalClusterController(QObject):
    """
    GlobalClusterController handles global clustering across all images and manages the presentation of sampled crops.
    """

    clustering_started = pyqtSignal()
    clustering_progress = pyqtSignal(int)
    clusters_ready = pyqtSignal(dict)  # New signal to notify when clusters are ready

    def __init__(self, model: ImageDataModel, view: ClusteredCropsView):
        super().__init__()
        self.model = model
        self.view = view
        self.image_processor = ImageProcessor()
        self.region_selector = UncertaintyRegionSelector()

        # Cluster data structure: {cluster_id: [annotations]}
        self.clusters = {}

        # Connect view signals
        self.connect_signals()

    def connect_signals(self):
        """
        Connect signals from the view to controller methods.
        """
        self.view.request_clustering.connect(self.start_clustering)
        self.view.sample_cluster.connect(self.on_sample_cluster)
        self.clustering_progress.connect(self.view.update_progress)
        self.clusters_ready.connect(self.on_clusters_ready)  # Connect to new slot

    @pyqtSlot()
    def start_clustering(self):
        """
        Initiates the global clustering process.
        """
        logging.info("Clustering process initiated by user.")
        self.clustering_started.emit()
        self.view.reset_progress()

        # Initialize and start the clustering worker thread
        self.worker = ClusteringWorker(model=self.model, selector=self.region_selector)
        self.worker.progress_updated.connect(self.clustering_progress.emit)
        self.worker.clustering_finished.connect(
            self.clusters_ready.emit)  # Emit clusters_ready instead of clustering_finished
        self.worker.start()

    @pyqtSlot(dict)
    def on_clusters_ready(self, clusters: Dict[int, List[Dict]]):
        """
        Handles the completion of the clustering process.
        """
        self.clusters = clusters
        logging.info(f"Clustering finished with {len(clusters)} clusters.")
        # No need to emit clustering_finished here

        # Update the view with sampled crops
        self.sample_and_display_crops()

    def sample_and_display_crops(self, num_clusters: int = 10, crops_per_cluster: int = 5):
        """
        Samples a subset of clusters and their annotations to display as zoomed-in crops.

        :param num_clusters: Number of clusters to sample.
        :param crops_per_cluster: Number of crops to sample per cluster.
        """
        if not self.clusters:
            logging.warning("No clusters available to sample.")
            return

        # Ensure num_clusters does not exceed available clusters
        available_clusters = len(self.clusters)
        num_clusters = min(num_clusters, available_clusters)
        logging.debug(f"Sampling {num_clusters} clusters out of {available_clusters} available.")

        # Sample clusters
        sampled_cluster_ids = np.random.choice(list(self.clusters.keys()),
                                               size=num_clusters,
                                               replace=False)
        sampled_crops = []
        for cluster_id in sampled_cluster_ids:
            annotations = self.clusters[cluster_id]
            if len(annotations) == 0:
                continue
            # Sample annotations within the cluster
            num_samples = min(crops_per_cluster, len(annotations))
            sampled_annotations = np.random.choice(annotations,
                                                   size=num_samples,
                                                   replace=False)
            for anno in sampled_annotations:
                image = self.model.get_image_data(anno['image_index'])['image']
                coord = anno['coord']
                crop_qimage = self.image_processor.extract_crop(
                    image, coord, crop_size=256, zoom_factor=2
                )
                # Convert QImage to QPixmap if necessary
                crop_pixmap = QPixmap.fromImage(crop_qimage)
                sampled_crops.append({
                    'cluster_id': cluster_id,
                    'image_index': anno['image_index'],
                    'coord': coord,
                    'crop': crop_pixmap
                })

        # Pass the sampled crops to the view
        self.view.display_sampled_crops(sampled_crops)
        logging.info(f"Displayed {len(sampled_crops)} sampled crops from {len(sampled_cluster_ids)} clusters.")

    @pyqtSlot(int)
    def on_sample_cluster(self, cluster_id: int):
        """
        Handles requests to display crops from a specific cluster.

        :param cluster_id: The ID of the cluster to sample.
        """
        if cluster_id not in self.clusters:
            logging.warning(f"Cluster ID {cluster_id} does not exist.")
            return

        annotations = self.clusters[cluster_id]
        sampled_crops = []
        num_samples = min(5, len(annotations))
        sampled_annotations = np.random.choice(annotations, size=num_samples, replace=False)
        for anno in sampled_annotations:
            image = self.model.get_image_data(anno['image_index'])['image']
            coord = anno['coord']
            crop_qimage, x_start, y_start = self.image_processor.extract_crop(
                image, coord, crop_size=256, zoom_factor=2
            )
            # Convert QImage to QPixmap if necessary
            crop_pixmap = QPixmap.fromImage(crop_qimage)
            sampled_crops.append({
                'cluster_id': cluster_id,
                'image_index': anno['image_index'],
                'coord': coord,
                'crop': crop_pixmap
            })

        self.view.display_sampled_crops(sampled_crops)
        logging.info(f"Displayed {len(sampled_crops)} sampled crops from cluster {cluster_id}.")
