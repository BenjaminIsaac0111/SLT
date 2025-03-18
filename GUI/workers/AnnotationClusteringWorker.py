import logging
from collections import defaultdict
from typing import List, Dict

import numba
import numpy as np
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

from GUI.models.Annotation import Annotation


@numba.njit
def compute_initial_distances(X, center_index):
    """
    Compute the distance from every point in X to X[center_index].
    """
    diff = X - X[center_index]
    dist = np.sqrt((diff * diff).sum(axis=1))
    return dist

@numba.njit
def update_distances(X, dist_to_closest_center, new_center_index):
    """
    Update distances to the new center if it's closer than existing centers.
    """
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


class ClusteringWorkerSignals(QObject):
    """
    Signals container for ClusteringWorker.
    """
    clustering_finished = pyqtSignal(dict)  # Emitted when clustering is complete
    progress_updated = pyqtSignal(int)  # Emitted to update progress (e.g., -1 means stage done)


class AnnotationClusteringWorker(QRunnable):
    """
    Performs clustering on a potentially large subset (core-set),
    then down-samples each cluster to exactly 'cluster_size' members.

    Submit this to a QThreadPool to run in the background:

        worker = ClusteringWorker(annotations, ...)
        worker.signals.clustering_finished.connect(...)
        threadpool.start(worker)
    """
    def __init__(
            self,
            annotations: List[Annotation],
            subsample_ratio: float = 1.0,
            cluster_method: str = "minibatchkmeans",
            cluster_size: int = 6
    ):
        super().__init__()
        self.signals = ClusteringWorkerSignals()

        self.annotations = annotations
        self.subsample_ratio = subsample_ratio
        self.cluster_method = cluster_method.lower()  # 'agglomerative' or 'minibatchkmeans'
        self.cluster_size = cluster_size

    def run(self):
        """
        Main entry point for the QRunnable. This is executed in a worker thread.
        """
        logging.info("ClusteringWorker started.")
        total_annotations = len(self.annotations)

        if total_annotations == 0:
            logging.warning("No annotations provided for clustering.")
            self.signals.clustering_finished.emit({})
            return

        # Build the feature matrix (logit_features + [uncertainty])
        try:
            feature_matrix = np.array([
                np.concatenate([anno.logit_features, [anno.uncertainty]])
                for anno in self.annotations
            ], dtype=np.float32)
        except Exception as e:
            logging.error(f"Error creating feature matrix: {e}")
            self.signals.clustering_finished.emit({})
            return

        logging.debug(f"Feature matrix shape: {feature_matrix.shape}")
        self.signals.progress_updated.emit(-1)  # Indicate start of core-set selection

        # ------------------------ Core-Set Selection -------------------------
        core_set_size = min(10000, len(feature_matrix))
        logging.info(f"Core-set size: {core_set_size}")
        subsample_size = max(int(len(feature_matrix) * self.subsample_ratio), core_set_size)

        # Subsample if needed
        subsample_indices = np.random.choice(len(feature_matrix), subsample_size, replace=False)
        subsample_X = feature_matrix[subsample_indices]

        logging.info(f"Selecting core-set using k-center greedy on {len(subsample_X)} points.")
        center_indices_subsample = k_center_greedy_numba(subsample_X, core_set_size)
        core_set_indices = subsample_indices[center_indices_subsample]

        core_set_features = feature_matrix[core_set_indices]
        core_set_annotations = [self.annotations[i] for i in core_set_indices]
        self.signals.progress_updated.emit(-1)  # Core-set selection done

        # -------------------------- Clustering Step --------------------------
        try:
            if self.cluster_method == 'minibatchkmeans':
                num_clusters = max(1, int(len(core_set_features) * 0.1))
                clustering = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=4096)
                core_set_labels = clustering.fit_predict(core_set_features)
                cluster_centers = None

            elif self.cluster_method == 'agglomerative':
                clustering = AgglomerativeClustering(n_clusters=None,
                                                     distance_threshold=5.0,
                                                     linkage='ward')
                core_set_labels = clustering.fit_predict(core_set_features)
                cluster_centers = None
            else:
                raise ValueError(f"Unknown clustering method: {self.cluster_method}")

            logging.info(f"Clustering complete. Found {len(set(core_set_labels))} clusters.")
            self.signals.progress_updated.emit(-1)  # Clustering done

        except Exception as e:
            logging.error(f"Clustering failed: {e}")
            self.signals.clustering_finished.emit({})
            return

        # Assign cluster IDs to the subset annotations
        for label, anno in zip(core_set_labels, core_set_annotations):
            anno.cluster_id = int(label)

        # Group into initial clusters
        clusters_dict = self.group_annotations_by_cluster(core_set_annotations)

        # Down-sample each cluster
        final_clusters_dict = self.downsample_clusters(clusters_dict, cluster_centers)

        self.signals.progress_updated.emit(-1)  # Final progress
        self.signals.clustering_finished.emit(final_clusters_dict)

    @staticmethod
    def group_annotations_by_cluster(annotations: List[Annotation]) -> Dict[int, List[Annotation]]:
        """
        Groups annotations by their cluster IDs.
        """
        clusters_map = defaultdict(list)
        for annotation in annotations:
            clusters_map[annotation.cluster_id].append(annotation)
        return dict(clusters_map)

    def downsample_clusters(
            self,
            clusters_dict: Dict[int, List[Annotation]],
            cluster_centers: np.ndarray
    ) -> Dict[int, List[Annotation]]:
        """
        Down-sample each cluster to exactly 'self.cluster_size' members using either:
        1) Center-based selection if cluster_centers are available.
        2) Diversity-based selection (k-center greedy) if no centers are available.
        """
        final_clusters = {}
        for cluster_id, cluster_annos in clusters_dict.items():
            if len(cluster_annos) <= self.cluster_size:
                # If cluster has <= cluster_size members, take them all
                final_clusters[cluster_id] = cluster_annos
                continue

            cluster_features = np.array([
                np.concatenate([anno.logit_features, [anno.uncertainty]])
                for anno in cluster_annos
            ], dtype=np.float32)

            if cluster_centers is not None:
                # Pick members closest to the cluster center
                center = cluster_centers[cluster_id]
                dists = np.linalg.norm(cluster_features - center, axis=1)
                closest_indices = np.argsort(dists)[:self.cluster_size]
                selected_annotations = [cluster_annos[idx] for idx in closest_indices]
            else:
                # Pick members via k-center greedy
                selected_indices = k_center_greedy_numba(cluster_features,
                                                         self.cluster_size,
                                                         random_state=42)
                selected_annotations = [cluster_annos[idx] for idx in selected_indices]

            final_clusters[cluster_id] = selected_annotations

        return final_clusters
