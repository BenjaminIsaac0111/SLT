import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Any

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread, pyqtSlot, QThreadPool
from sklearn.utils.class_weight import compute_class_weight

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import ImageDataModel
from GUI.workers.AnnotationClusteringWorker import AnnotationClusteringWorker


class AnnotationExtractionWorker(QThread):
    """
    Extracts annotations from all images in the model, emitting progress signals.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)

    def __init__(self, model: ImageDataModel, annotation_generator: None, parent=None):
        super().__init__(parent)
        self.model = model
        self.annotation_generator = annotation_generator

    def run(self):
        annotations = []
        total_images = self.model.get_number_of_images()
        for idx in range(total_images):
            image_data = self.model.get_image_data(idx)
            annos = self.extract_annotations_from_image(image_data, idx)
            annotations.extend(annos)

            # Emit progress
            progress_pct = int((idx + 1) / total_images * 100)
            self.progress.emit(progress_pct)

        self.finished.emit(annotations)

    def extract_annotations_from_image(self, image_data: Dict[str, Any], image_index: int) -> List[Annotation]:
        """
        Extracts annotations from a single image.
        """
        annotations = []
        uncertainty_map = image_data.get('uncertainty')
        logits = image_data.get('logits')
        filename = image_data.get('filename')

        if uncertainty_map is None or logits is None or filename is None:
            return annotations

        logit_features, coords = self.annotation_generator.generate_annotations(
            uncertainty_map=uncertainty_map,
            logits=logits
        )

        for coord, logit_feature in zip(coords, logit_features):
            anno = Annotation(
                filename=filename,
                coord=coord,
                logit_features=logit_feature,
                class_id=-1,  # Default to unlabeled
                image_index=image_index,
                uncertainty=uncertainty_map[tuple(coord)],
                cluster_id=None,
                # Map the max logit index to a class name
                model_prediction=CLASS_COMPONENTS.get(np.argmax(logit_feature), "None")
            )
            annotations.append(anno)

        return annotations


class AnnotationClusteringController(QObject):
    """
    ClusteringController handles the end-to-end flow:
      1) Annotation extraction
      2) Clustering
      3) Labeling statistics updates
    """

    # ----- Signals -----
    clustering_started = pyqtSignal()
    clustering_progress = pyqtSignal(int)
    clusters_ready = pyqtSignal(dict, object, object)
    labeling_statistics_updated = pyqtSignal(dict)
    annotation_progress = pyqtSignal(int)
    annotation_progress_finished = pyqtSignal()
    clustering_progress_bar_visible = pyqtSignal(bool)

    # Additional signals for show/hide in the UI
    show_clustering_progress_bar = pyqtSignal()
    hide_clustering_progress_bar = pyqtSignal()

    def __init__(self, model: ImageDataModel, annotation_generator):
        """
        Initializes the ClusteringController.

        :param model: An instance of ImageDataModel.
        :param annotation_generator: An instance of some annotation generator.
        """
        super().__init__()
        self.model = model
        self.annotation_generator = annotation_generator

        self.clusters: Dict[int, List[Annotation]] = {}
        self.cluster_labels: Dict[int, str] = {}
        self.crops_per_cluster = 100

        self.clustering_in_progress = False

        self.annotation_extraction_worker: Optional[AnnotationExtractionWorker] = None

        self.threadpool = QThreadPool.globalInstance()

    # -------------------------------------------------------------------------
    #                         PUBLIC SLOTS & METHODS
    # -------------------------------------------------------------------------

    @pyqtSlot()
    def start_clustering(self):
        """
        Begins the clustering process, starting with annotation extraction.
        """
        if self.clustering_in_progress:
            logging.warning("Clustering is already running.")
            return

        logging.info("Clustering initiated.")
        self.clustering_in_progress = True
        self.clustering_started.emit()
        self._init_annotation_extraction()

    @pyqtSlot(list)
    def on_annotation_extraction_finished(self, all_annotations: List[Annotation]):
        """
        Called when the annotation extraction worker is done.
        Then we launch the clustering step.
        """
        self.annotation_progress.emit(100)
        self.annotation_progress_finished.emit()

        self.show_clustering_progress_bar.emit()
        self.clustering_progress.emit(-1)

        self._start_clustering_with_annotations(all_annotations)

    def _start_clustering_with_annotations(self, all_annotations: List[Annotation]):
        """
        Creates and starts the QRunnable-based ClusteringWorker.
        """
        clustering_worker = AnnotationClusteringWorker(
            annotations=all_annotations,
            subsample_ratio=1.0,
            cluster_method="gaussianmixture"
        )

        # Connect signals from the worker's .signals object
        clustering_worker.signals.clustering_finished.connect(self.on_clustering_finished)
        clustering_worker.signals.progress_updated.connect(self.clustering_progress.emit)

        # Submit the worker to the thread pool
        self.threadpool.start(clustering_worker)

    @pyqtSlot(dict, object, object)
    def on_clustering_finished(self, clusters: Dict[int, List[Annotation]], clustering_model, feature_matrix):
        """
        Called when the clustering worker finishes.
        """
        self.clustering_in_progress = False

        self.clusters = clusters
        self.clusters_ready.emit(clusters, clustering_model, feature_matrix)
        self.hide_clustering_progress_bar.emit()
        self.compute_labeling_statistics()

    def collect_all_annotations(self) -> List[Annotation]:
        """
        (Optional) Collects annotations synchronously without threading.
        """
        total_images = self.model.get_number_of_images()
        annotations = []
        for idx in range(total_images):
            try:
                image_data = self.model.get_image_data(idx)
                annotations.extend(self._extract_annotations_from_image(image_data, idx))
            except Exception as e:
                logging.error(f"Error collecting annotations from image {idx}: {e}")
                continue

            progress = int((idx + 1) / total_images * 100)
            self.annotation_progress.emit(progress)

        logging.info(f"Total annotations collected: {len(annotations)}")
        return annotations

    def compute_labeling_statistics(self):
        """
        Computes and emits updated labeling statistics (counts, disagreements, etc.),
        and computes class weights using sklearn's compute_class_weight, ignoring
        special classes (-1, -2, -3).
        """
        clusters = self.get_clusters()
        total_annotations = 0
        total_labeled = 0

        # Initialize class counts, including special IDs
        class_counts = {cid: 0 for cid in CLASS_COMPONENTS.keys()}
        class_counts[-1] = 0  # Unlabeled
        class_counts[-2] = 0  # Unsure
        class_counts[-3] = 0  # Artifact

        disagreement_count = 0
        all_labels = []  # For computing class weights

        for cluster_id, annotations in clusters.items():
            for anno in annotations:
                total_annotations += 1
                # Use -1 if class_id is None
                assigned_class = anno.class_id if anno.class_id is not None else -1
                all_labels.append(assigned_class)

                if assigned_class not in class_counts:
                    class_counts[assigned_class] = 0
                class_counts[assigned_class] += 1

                # Count labeled (excluding special IDs)
                if assigned_class not in (-1, -2, -3):
                    total_labeled += 1

                # Count disagreements if labeled
                if anno.model_prediction:
                    model_class_id = self._get_class_id_from_prediction(anno.model_prediction)
                    if model_class_id is not None and assigned_class not in (-1, -2, -3):
                        if assigned_class != model_class_id:
                            disagreement_count += 1

        agreement_percentage = 0.0
        if total_labeled > 0:
            agreement_percentage = ((total_labeled - disagreement_count) / total_labeled) * 100

        # Filter out special labels (-1, -2, -3) before computing weights
        filtered_labels = [label for label in all_labels if label not in (-1, -2, -3)]
        class_weights = {}
        if filtered_labels:
            # Compute weights for the unique classes in filtered_labels
            classes = np.unique(filtered_labels)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=filtered_labels)
            class_weights = dict(zip(classes, weights))

        statistics = {
            'total_annotations': total_annotations,
            'total_labeled': total_labeled,
            'class_counts': class_counts,
            'disagreement_count': disagreement_count,
            'agreement_percentage': agreement_percentage,
            'class_weights': class_weights,  # Added class weights (excluding special classes)
        }
        logging.info(f"Labeling statistics computed: {statistics}")
        self.labeling_statistics_updated.emit(statistics)

    def generate_cluster_info(self) -> Dict[int, dict]:
        """
        Returns cluster metadata for each cluster ID.
        """
        cluster_info = OrderedDict()
        for cluster_id, annotations in self.clusters.items():
            num_annotations = len(annotations)
            num_labeled = sum(1 for a in annotations if a.class_id != -1)
            labeled_percentage = (num_labeled / num_annotations) * 100 if num_annotations > 0 else 0

            cluster_info[cluster_id] = {
                'num_annotations': num_annotations,
                'num_images': len({a.filename for a in annotations}),
                'labeled_percentage': labeled_percentage,
                'label': self.cluster_labels.get(cluster_id, ''),
            }
        return cluster_info

    def update_cluster_labels(self, cluster_id: int, label: str):
        """
        Updates the label for a specific cluster (user-defined).
        """
        self.cluster_labels[cluster_id] = label
        logging.info(f"Cluster {cluster_id} label updated to '{label}'.")

    def get_clusters(self) -> Dict[int, List[Annotation]]:
        return self.clusters

    def set_crops_per_cluster(self, num_crops: int):
        self.crops_per_cluster = num_crops
        logging.info(f"Crops per cluster set to {num_crops}.")

    def cleanup(self):
        """
        Cleans up resources, if necessary.
        """
        pass

    # -------------------------------------------------------------------------
    #                            PRIVATE HELPERS
    # -------------------------------------------------------------------------

    def _init_annotation_extraction(self):
        """
        Sets up and starts the AnnotationExtractionWorker (QThread-based or also refactor to QRunnable if you want).
        """
        self.annotation_extraction_worker = AnnotationExtractionWorker(self.model, self.annotation_generator)
        self.annotation_extraction_worker.progress.connect(self.annotation_progress.emit)
        self.annotation_extraction_worker.finished.connect(self.on_annotation_extraction_finished)

        self.annotation_progress.emit(0)
        self.annotation_extraction_worker.start()

    def _clustering_thread_is_running(self) -> bool:
        """
        Checks if a clustering thread is currently running.
        """
        return (self.clustering_thread and self.clustering_thread.isRunning()) if self.clustering_thread else False

    def _extract_annotations_from_image(self, image_data: Dict[str, Any], image_index: int) -> List[Annotation]:
        """
        Extracts annotations from a single image (used in the synchronous path).
        """
        annotations = []
        uncertainty_map = image_data.get('uncertainty')
        logits = image_data.get('logits')
        filename = image_data.get('filename')

        if uncertainty_map is None or logits is None or filename is None:
            logging.warning(f"Missing data for image index {image_index}. Skipping.")
            return annotations

        logit_features, coords = self.annotation_generator.generate_annotations(
            uncertainty_map=uncertainty_map,
            logits=logits
        )

        for coord, logit_feature in zip(coords, logit_features):
            annotation = Annotation(
                filename=filename,
                coord=coord,
                logit_features=logit_feature,
                class_id=-1,
                image_index=image_index,
                uncertainty=uncertainty_map[tuple(coord)],
                cluster_id=None,
                model_prediction=self._get_model_prediction(logit_feature)
            )
            annotations.append(annotation)

        return annotations

    def _get_model_prediction(self, logit_feature: np.ndarray) -> str:
        """
        Maps logit feature (highest index) to a class name.
        """
        class_index = np.argmax(logit_feature)
        return CLASS_COMPONENTS.get(class_index, "None")

    def _get_class_id_from_prediction(self, prediction_name: str) -> Optional[int]:
        """
        Looks up the class ID from a predicted class name.
        """
        for cid, cname in CLASS_COMPONENTS.items():
            if cname == prediction_name:
                return cid
        return None
