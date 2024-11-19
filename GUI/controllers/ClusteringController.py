import logging
from collections import OrderedDict
from typing import Dict, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.Annotation import Annotation
from GUI.workers.ClusteringWorker import ClusteringWorker


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
        if self.clustering_thread and self.clustering_thread.isRunning():
            logging.warning("Clustering thread is already running. Skipping new start.")
            return

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

    def on_worker_finished(self):
        self.clustering_worker.deleteLater()
        self.clustering_worker = None

    def on_thread_finished(self):
        self.clustering_thread.deleteLater()
        self.clustering_thread = None

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
        if self.clustering_worker:
            try:
                self.clustering_worker.progress_updated.disconnect()
                self.clustering_worker.clustering_finished.disconnect()
            except TypeError:
                pass
            self.clustering_worker.deleteLater()
            self.clustering_worker = None

        if self.clustering_thread:
            if self.clustering_thread.isRunning():
                self.clustering_thread.quit()
                self.clustering_thread.wait()
            self.clustering_thread.deleteLater()
            self.clustering_thread = None

        logging.info("Clustering worker and thread cleaned up.")
