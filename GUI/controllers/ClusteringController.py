import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Any

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.UncertaintyRegionSelector import UncertaintyRegionSelector
from GUI.workers.ClusteringWorker import ClusteringWorker


class AnnotationExtractionWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)

    def __init__(self, model, region_selector, parent=None):
        super().__init__(parent)
        self.model = model
        self.region_selector = region_selector

    def run(self):
        annotations = []
        total_images = self.model.get_number_of_images()
        for idx, image_index in enumerate(range(total_images)):
            # Extract annotations from each image
            image_data = self.model.get_image_data(image_index)
            annos = self.extract_annotations_from_image(image_data, image_index)
            annotations.extend(annos)

            # Emit progress
            progress = int((idx + 1) / total_images * 100)
            self.progress.emit(progress)

        self.finished.emit(annotations)

    def extract_annotations_from_image(self, image_data, image_index):
        # This is similar to your current extract_annotations_from_image code
        annotations = []
        uncertainty_map = image_data.get('uncertainty', None)
        logits = image_data.get('logits', None)
        filename = image_data.get('filename', None)
        if uncertainty_map is None or logits is None or filename is None:
            return annotations

        logit_features, coords = self.region_selector.generate_point_labels(
            uncertainty_map=uncertainty_map,
            logits=logits
        )

        for coord, logit_feature in zip(coords, logit_features):
            anno = Annotation(
                filename=filename,
                coord=coord,
                logit_features=logit_feature,
                class_id=-1,
                image_index=image_index,
                uncertainty=uncertainty_map[tuple(coord)],
                cluster_id=None,
                model_prediction=CLASS_COMPONENTS.get(np.argmax(logit_feature), "None")
            )
            annotations.append(anno)

        return annotations


class ClusteringController(QObject):
    """
    ClusteringController handles clustering operations independently.
    It communicates with GlobalClusterController via signals and slots.
    """
    clustering_started = pyqtSignal()
    clustering_progress = pyqtSignal(int)
    clusters_ready = pyqtSignal(dict)
    labeling_statistics_updated = pyqtSignal(dict)
    annotation_progress = pyqtSignal(int)
    annotation_progress_finished = pyqtSignal()
    clustering_progress_bar_visible = pyqtSignal(bool)
    show_clustering_progress_bar = pyqtSignal()
    hide_clustering_progress_bar = pyqtSignal()

    def __init__(self, model: ImageDataModel, region_selector: UncertaintyRegionSelector):
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
        if self.clustering_thread and self.clustering_thread.isRunning():
            logging.warning("Clustering thread is already running.")
            return

        logging.info("Clustering initiated.")
        self.clustering_started.emit()

        # Start annotation extraction in the background
        self.annotation_extraction_worker = AnnotationExtractionWorker(self.model, self.region_selector)
        self.annotation_extraction_worker.progress.connect(self.annotation_progress.emit)
        self.annotation_extraction_worker.finished.connect(self.on_annotation_extraction_finished)

        # Begin extraction
        self.annotation_progress.emit(0)
        self.annotation_extraction_worker.start()

    @pyqtSlot(list)
    def on_annotation_extraction_finished(self, all_annotations):
        # Once done:
        self.annotation_progress.emit(100)
        self.annotation_progress_finished.emit()

        # Instead of self.view.show_clustering_progress_bar(), emit a signal:
        self.show_clustering_progress_bar.emit()

        # Instead of self.view.update_clustering_progress_bar(0), just emit clustering_progress(0):
        self.clustering_progress.emit(0)

        # Now proceed with clustering
        self.start_clustering_with_annotations(all_annotations)

    def start_clustering_with_annotations(self, all_annotations):
        self.clustering_worker = ClusteringWorker(
            annotations=all_annotations,
            subsample_ratio=1.0,
            cluster_method="minibatchkmeans"
        )
        self.clustering_worker.clustering_finished.connect(self.on_clustering_finished)
        self.clustering_worker.progress_updated.connect(self.clustering_progress.emit)

        # Start the QThread worker directly
        self.clustering_worker.start()

    def collect_all_annotations(self) -> List[Annotation]:
        total_images = self.model.get_number_of_images()
        annotations = []
        for idx, image_index in enumerate(range(total_images)):
            try:
                image_data = self.model.get_image_data(image_index)
                annotations.extend(self.extract_annotations_from_image(image_data, image_index))
            except Exception as e:
                logging.error(f"Error collecting annotations from image {image_index}: {e}")
                continue

            # Emit progress for annotation extraction
            progress = int((idx + 1) / total_images * 100)
            self.annotation_progress.emit(progress)

        logging.info(f"Total annotations collected: {len(annotations)}")
        return annotations

    def extract_annotations_from_image(self, image_data: Dict[str, Any], image_index: int) -> List[Annotation]:
        """
        Extracts annotations from image data.

        :param image_data: Dictionary containing image data.
        :param image_index: Index of the image.
        :return: A list of Annotation objects.
        """
        annotations = []
        uncertainty_map = image_data.get('uncertainty', None)
        logits = image_data.get('logits', None)
        filename = image_data.get('filename', None)

        if uncertainty_map is None or logits is None or filename is None:
            logging.warning(f"Missing data for image index {image_index}. Skipping.")
            return annotations

        logit_features, coords = self.region_selector.generate_point_labels(
            uncertainty_map=uncertainty_map,
            logits=logits
        )

        for coord, logit_feature in zip(coords, logit_features):
            annotation = Annotation(
                filename=filename,
                coord=coord,
                logit_features=logit_feature,
                class_id=-1,  # Default to -1 (unlabeled)
                image_index=image_index,
                uncertainty=uncertainty_map[tuple(coord)],
                cluster_id=None,
                model_prediction=self._get_model_prediction(logit_feature)
            )
            annotations.append(annotation)

        return annotations

    def _get_model_prediction(self, logit_feature: np.ndarray) -> str:
        """
        Determines the model's prediction based on logit features.

        :param logit_feature: Numpy array of logit features.
        :return: Predicted class name.
        """
        class_index = np.argmax(logit_feature)
        return CLASS_COMPONENTS.get(class_index, "None")

    def on_clustering_finished(self, clusters: Dict[int, List[Annotation]]):
        self.clusters = clusters
        self.clusters_ready.emit(self.clusters)

        # Instead of self.view.hide_clustering_progress_bar():
        self.hide_clustering_progress_bar.emit()

        # Compute and emit labeling statistics as before
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
        class counts, global disagreement count, and agreement percentage.
        Emits labeling_statistics_updated signal with the statistics dictionary.
        """
        clusters = self.get_clusters()  # {cluster_id: [Annotation, ...]}
        total_annotations = 0
        total_labeled = 0
        class_counts = {cid: 0 for cid in CLASS_COMPONENTS.keys()}
        class_counts[-1] = 0  # Unlabeled
        class_counts[-2] = 0  # Unsure

        disagreement_count = 0

        for cluster_id, annotations in clusters.items():
            for anno in annotations:
                total_annotations += 1
                assigned_class = anno.class_id if anno.class_id is not None else -1

                # Update class counts
                if assigned_class not in class_counts:
                    class_counts[assigned_class] = 0
                class_counts[assigned_class] += 1

                # Count labeled (excluding -1 and -2)
                if assigned_class not in [-1, -2]:
                    total_labeled += 1

                # Compute disagreement if annotation is labeled and model_prediction is available
                if anno.model_prediction is not None:
                    model_class_id = self.get_class_id_from_prediction(anno.model_prediction)
                    # Consider it a disagreement only if annotation is labeled (not -1/-2) and model_class_id found
                    if model_class_id is not None and assigned_class not in [-1, -2]:
                        if assigned_class != model_class_id:
                            disagreement_count += 1

        # Compute agreement percentage
        agreement_percentage = 0.0
        if total_labeled > 0:
            agreement_percentage = ((total_labeled - disagreement_count) / total_labeled) * 100

        # Prepare statistics dictionary
        statistics = {
            'total_annotations': total_annotations,
            'total_labeled': total_labeled,
            'class_counts': class_counts,
            'disagreement_count': disagreement_count,
            'agreement_percentage': agreement_percentage
        }

        logging.info(f"Labeling statistics computed: {statistics}")
        self.labeling_statistics_updated.emit(statistics)

    def get_class_id_from_prediction(self, prediction_name: str) -> Optional[int]:
        """
        Maps the model's predicted class name to a class_id.
        Adjust this method if needed based on how you map predictions to classes.

        :param prediction_name: The predicted class name.
        :return: Corresponding class_id or None.
        """
        for c_id, c_name in CLASS_COMPONENTS.items():
            if c_name == prediction_name:
                return c_id
        return None

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
        pass
