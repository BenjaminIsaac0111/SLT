import logging

import numba
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import ImageDataModel
from GUI.models.UncertaintyRegionSelector import UncertaintyRegionSelector


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
    Worker thread for performing global clustering to keep the UI responsive.
    """
    clustering_finished = pyqtSignal(list)
    progress_updated = pyqtSignal(int)

    def __init__(self,
                 hdf5_file_path: str,
                 labels_acquirer: UncertaintyRegionSelector,
                 use_kcenters_greedy: bool = True,
                 subsample_ratio: float = 1.0,
                 parent=None):
        super().__init__(parent)
        self.hdf5_file_path = hdf5_file_path  # Path to HDF5 file
        self.labels_acquirer = labels_acquirer
        self.model = None  # Will be initialized in the thread
        self.use_kcenters_greedy = use_kcenters_greedy
        # subsample_ratio: fraction of points to consider for k-center selection to speed things up
        self.subsample_ratio = subsample_ratio

    def process_image(self, idx):
        """
        Processes a single image and assigns a cluster_id to each Annotation.
        """
        data = self.model.get_image_data(idx)
        uncertainty_map = data.get('uncertainty', None)
        logits = data.get('logits', None)
        filename = data.get('filename', None)

        if uncertainty_map is None or logits is None or filename is None:
            logging.warning(f"Missing data for image index {idx}. Skipping.")
            return [], idx

        logit_features, dbscan_coords = self.labels_acquirer.generate_point_labels(
            uncertainty_map=uncertainty_map,
            logits=logits
        )

        annotations = [
            Annotation(
                filename=filename,
                coord=coord,
                logit_features=logit_feature,
                class_id=-1,  # Default to -1 (unlabeled)
                image_index=idx,
                uncertainty=uncertainty_map[coord[0], coord[1]],
                cluster_id=None,
                model_prediction=self._get_model_prediction(logit_feature)
            )
            for coord, logit_feature in zip(dbscan_coords, logit_features)
        ]

        return annotations, idx

    def _get_model_prediction(self, logit_feature: np.ndarray) -> str:
        """
        Determines the model's prediction based on logit features.
        """
        class_index = np.argmax(logit_feature)
        return CLASS_COMPONENTS.get(class_index, "None")

    def run(self):
        """
        Executes the clustering process using a core-set selection (either k-means++ or approximate k-center greedy).
        """
        # Initialize the ImageDataModel in this thread
        self.model = ImageDataModel(self.hdf5_file_path)

        all_annotations = []
        total_images = self.model.get_number_of_images()
        logging.info(f"Starting clustering on {total_images} images.")

        # Process each image sequentially
        for i in range(total_images):
            try:
                annotations, idx = self.process_image(i)
                all_annotations.extend(annotations)
            except Exception as e:
                logging.error(f"Error processing image {i}: {e}")
                continue

            # Emit progress update
            progress = int(((i + 1) / total_images) * 100)
            self.progress_updated.emit(progress)
            logging.debug(f"Processed image {i + 1}/{total_images}. Progress: {progress}%")

        logging.info(f"Total annotations collected: {len(all_annotations)}")

        if not all_annotations:
            logging.warning("No annotations found across all images.")
            self.clustering_finished.emit({})
            return

        # Extract logit features for clustering
        try:
            logit_matrix = np.array(
                [np.concatenate([anno.logit_features, [anno.uncertainty]]) for anno in all_annotations]
            )
            logging.debug(f"Logit matrix shape: {logit_matrix.shape}")
        except Exception as e:
            logging.error(f"Error creating logit matrix: {e}")
            self.clustering_finished.emit({})
            return

        # Determine core set size and subsample
        core_set_size = min(10000, len(logit_matrix))
        logging.info(f"Core-set size determined: {core_set_size}")

        # If using k-centers greedy, we approximate by subsampling before selection
        if self.use_kcenters_greedy:
            subsample_size = int(len(logit_matrix) * self.subsample_ratio)
            subsample_size = max(subsample_size, core_set_size)  # Ensure we have at least k points
            if subsample_size < len(logit_matrix):
                # Subsample indices
                subsample_indices = np.random.choice(len(logit_matrix), subsample_size, replace=False)
                subsample_X = logit_matrix[subsample_indices]
            else:
                subsample_indices = np.arange(len(logit_matrix))
                subsample_X = logit_matrix

            logging.info(f"Selecting core-set using approximate k-centers greedy from {len(subsample_X)} points.")
            # Use JIT-optimized k-center greedy on subsample
            center_indices_subsample = k_center_greedy_numba(subsample_X, core_set_size)
            # Map back to original indices
            core_set_indices = subsample_indices[center_indices_subsample]

        else:
            # Use k-means++ initialization via MiniBatchKMeans
            logging.info("Selecting core-set using k-means++.")
            try:
                kmeans = MiniBatchKMeans(
                    n_clusters=core_set_size,
                    init='k-means++',
                    random_state=42,
                    batch_size=3072
                )
                kmeans.fit(logit_matrix)
                core_set_indices = np.unique(kmeans.labels_, return_index=True)[1]
            except Exception as e:
                logging.error(f"Error in core-set selection with k-means: {e}")
                self.clustering_finished.emit({})
                return

        core_set_features = logit_matrix[core_set_indices]
        core_set_annotations = [all_annotations[i] for i in core_set_indices]
        logging.info(f"Core-set selected with {len(core_set_features)} features.")

        # Step 2: Perform clustering on the core-set using Agglomerative Clustering
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1.50,
                linkage='ward'
            )
            clustering.fit(core_set_features)
            core_set_labels = clustering.labels_
            logging.info(f"Core-set clustering complete. Number of clusters: {len(set(core_set_labels))}")
        except Exception as e:
            logging.error(f"Clustering on core-set failed: {e}")
            self.clustering_finished.emit({})
            return

        # Assign cluster IDs to the core-set annotations
        for label, annotation in zip(core_set_labels, core_set_annotations):
            annotation.cluster_id = int(label)

        # Close the model's HDF5 file
        if self.model is not None:
            self.model.close()

        logging.info("Clustering complete for the core-set.")
        self.clustering_finished.emit(core_set_annotations)
