import logging
from collections import defaultdict
from typing import List, Dict

import numba
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

from GUI.models.Annotation import Annotation


@numba.njit
def compute_initial_distances(X, center_index):
    # Compute the distance from every point in X to the center X[center_index]
    diff = X - X[center_index]
    dist = np.sqrt((diff * diff).sum(axis=1))
    return dist


@numba.njit
def update_distances(X, dist_to_closest_center, new_center_index):
    # Update the distance array using the newly added center
    diff = X - X[new_center_index]
    new_dist = np.sqrt((diff * diff).sum(axis=1))
    for i in range(len(dist_to_closest_center)):
        if new_dist[i] < dist_to_closest_center[i]:
            dist_to_closest_center[i] = new_dist[i]


@numba.njit
def k_center_greedy_numba(X, k, random_state=42):
    """
    Approximate k-center greedy selection with Numba optimization.
    Picks one random center, then repeatedly picks the farthest point.
    """
    np.random.seed(random_state)
    n = X.shape[0]
    if k >= n:
        return np.arange(n)

    # Start by picking a random center
    center_indices = np.empty(k, dtype=np.int64)
    initial_center = np.random.randint(n)
    center_indices[0] = initial_center

    dist_to_closest_center = compute_initial_distances(X, initial_center)

    for idx in range(1, k):
        # Pick the farthest point from the current set of centers
        next_center = np.argmax(dist_to_closest_center)
        center_indices[idx] = next_center
        update_distances(X, dist_to_closest_center, next_center)

    return center_indices


class ClusteringWorker(QThread):
    """
    This worker assumes that all annotations are fully prepared (i.e., have logit_features, uncertainty, etc.).
    It does not acquire or process image data.
    """
    clustering_finished = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)

    def __init__(
            self,
            annotations: List[Annotation],
            subsample_ratio: float = 1.0,
            cluster_method: str = "minibatchkmeans",
            parent=None
    ):
        super().__init__(parent)
        self.annotations = annotations
        self.subsample_ratio = subsample_ratio
        self.cluster_method = cluster_method.lower()  # 'agglomerative', 'minibatchkmeans', or 'hdbscan'

    def run(self):
        logging.info("ClusteringWorker started.")
        total_annotations = len(self.annotations)

        if total_annotations == 0:
            logging.warning("No annotations provided for clustering.")
            self.clustering_finished.emit({})
            return

        try:
            feature_matrix = np.array([
                np.concatenate([anno.logit_features, [anno.uncertainty]])
                for anno in self.annotations
            ], dtype=np.float32)
        except Exception as e:
            logging.error(f"Error creating feature matrix: {e}")
            self.clustering_finished.emit({})
            return

        logging.debug(f"Feature matrix shape: {feature_matrix.shape}")
        self.progress_updated.emit(-1)  # Emit initial progress (5%)

        # Core-Set Selection
        core_set_size = min(5000, len(feature_matrix))
        logging.info(f"Core-set size: {core_set_size}")
        subsample_size = max(int(len(feature_matrix) * self.subsample_ratio), core_set_size)

        subsample_indices = np.random.choice(len(feature_matrix), subsample_size, replace=False)
        subsample_X = feature_matrix[subsample_indices]

        logging.info(f"Selecting core-set using k-center greedy on {len(subsample_X)} points.")
        center_indices_subsample = k_center_greedy_numba(subsample_X, core_set_size)
        core_set_indices = subsample_indices[center_indices_subsample]

        core_set_features = feature_matrix[core_set_indices]
        core_set_annotations = [self.annotations[i] for i in core_set_indices]
        self.progress_updated.emit(-1)  # Core-set selection done

        # Clustering Step
        try:
            if self.cluster_method == 'minibatchkmeans':
                num_clusters = max(1, int(len(core_set_features) * 0.1))
                clustering = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=4096)
                core_set_labels = clustering.fit_predict(core_set_features)
            elif self.cluster_method == 'agglomerative':
                clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=5.0, linkage='ward')
                core_set_labels = clustering.fit(core_set_features)
            else:
                raise ValueError(f"Unknown clustering method: {self.cluster_method}")

            logging.info(f"Clustering complete. Found {len(set(core_set_labels))} clusters.")
            self.progress_updated.emit(-1)  # Clustering done
        except Exception as e:
            logging.error(f"Clustering failed: {e}")
            self.clustering_finished.emit({})
            return

        for label, anno in zip(core_set_labels, core_set_annotations):
            anno.cluster_id = int(label)

        # Finalizing
        clusters_dict = self.group_annotations_by_cluster(core_set_annotations)
        self.progress_updated.emit(100)  # Final progress
        self.clustering_finished.emit(clusters_dict)

    @staticmethod
    def group_annotations_by_cluster(annotations: List[Annotation]) -> Dict[int, List[Annotation]]:
        """
        Groups annotations by their cluster IDs.

        :param annotations: A list of Annotation objects.
        :return: A dictionary mapping cluster IDs to lists of annotations.
        """
        clusters = defaultdict(list)
        for annotation in annotations:
            clusters[annotation.cluster_id].append(annotation)
        return dict(clusters)
