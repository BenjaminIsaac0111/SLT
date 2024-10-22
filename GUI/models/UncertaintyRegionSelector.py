import logging
from typing import List, Tuple, Dict

import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.cluster import DBSCAN, AgglomerativeClustering


class UncertaintyRegionSelector:
    """
    The UncertaintyRegionSelector class identifies uncertain regions from an uncertainty map.
    It uses DBSCAN for spatial clustering of significant points and Agglomerative Clustering
    for clustering the logit features to ensure selected regions are semantically diverse.
    """

    def __init__(
            self,
            filter_size: int = 32,
            aggregation_method: str = 'mean',
            gaussian_sigma: float = 3.0,
            edge_buffer: int = 64,
            eps: float = 1.0,  # DBSCAN parameter for spatial clustering
            min_samples: int = 1,  # DBSCAN parameter for spatial clustering
            distance_threshold: float = 2.5,  # Agglomerative Clustering parameter for logit features
            linkage: str = 'ward'  # Linkage criteria for Agglomerative Clustering
    ):
        """
        Initializes the UncertaintyRegionSelector.

        :param filter_size: Size for the maximum filter to identify local maxima.
        :param aggregation_method: Method to aggregate 3D uncertainty map into 2D ('mean', 'max', 'std').
        :param gaussian_sigma: Sigma parameter for the Gaussian filter.
        :param edge_buffer: Buffer from image edges to exclude certain coordinates.
        :param eps: DBSCAN epsilon parameter for spatial clustering.
        :param min_samples: DBSCAN minimum samples parameter for spatial clustering.
        :param distance_threshold: The linkage distance threshold for Agglomerative Clustering.
        :param linkage: The linkage criterion for Agglomerative Clustering ('ward', 'complete', 'average', 'single').
        """
        self.filter_size = filter_size
        self.aggregation_method = aggregation_method
        self.gaussian_sigma = gaussian_sigma
        self.edge_buffer = edge_buffer
        self.eps = eps
        self.min_samples = min_samples
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self._validate_params()
        logging.info(
            "UncertaintyRegionSelector initialized with filter_size=%d, aggregation_method='%s', "
            "gaussian_sigma=%.2f, edge_buffer=%d, eps=%.2f, min_samples=%d, distance_threshold=%.2f, linkage='%s'.",
            filter_size, aggregation_method, gaussian_sigma, edge_buffer, eps, min_samples, distance_threshold, linkage
        )

    def _validate_params(self):
        """
        Validates input parameters.
        """
        if self.filter_size <= 0:
            raise ValueError("filter_size must be a positive integer.")
        if self.aggregation_method not in ('mean', 'max', 'std'):
            raise ValueError("aggregation_method must be 'mean', 'max', or 'std'.")
        if self.eps <= 0:
            raise ValueError("eps must be a positive float.")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be a positive integer.")
        if self.distance_threshold is None or self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be a positive float.")
        if self.linkage not in ('ward', 'complete', 'average', 'single'):
            raise ValueError("linkage must be one of 'ward', 'complete', 'average', or 'single'.")

    def generate_point_labels(
            self,
            uncertainty_map: np.ndarray,
            logits: np.ndarray,
    ) -> Tuple[ndarray, List[Tuple[int, int]]]:
        """
        Selects the uncertain regions from the uncertainty map, clusters the coordinates using DBSCAN,
        and clusters the logit features using Agglomerative Clustering.

        :param uncertainty_map: A 3D numpy array representing uncertainty values.
        :param logits: A 3D numpy array of logit values per pixel.
        :return: A dictionary where keys are cluster labels and values are lists of (row, column) tuples.
        """
        # Step 1: Aggregate the 3D uncertainty map into a 2D map
        uncertainty_map_2d = self._aggregate_uncertainty(uncertainty_map)

        # Step 2: Apply Gaussian smoothing. (Not sure how mu)
        uncertainty_map_smoothed = gaussian_filter(uncertainty_map_2d, sigma=self.gaussian_sigma)
        logging.debug("Applied Gaussian filter with sigma=%.2f.", self.gaussian_sigma)

        # Step 3: Normalize the uncertainty map
        uncertainty_normalized = self._normalize_uncertainty(uncertainty_map_smoothed)

        # Step 4: Identify significant coordinates
        initial_coords = self._identify_significant_coords(uncertainty_normalized)
        logging.info("Identified %d significant coordinates.", len(initial_coords))

        if not initial_coords:
            logging.warning("No significant coordinates found.")
            return {}

        # Step 5: Perform DBSCAN on spatial coordinates to group them
        dbscan_coords = self._dbscan_cluster_coordinates(initial_coords)
        logging.info("DBSCAN identified %d points in spatial domain.", len(dbscan_coords))

        # Step 6: Extract logit features at the DBSCAN-clustered coordinates
        logit_features = self._extract_logit_features(
            np.concatenate([logits, uncertainty_map_2d[..., np.newaxis]], axis=-1), dbscan_coords)
        logging.debug("Extracted logit features.")

        return logit_features, dbscan_coords

    def _aggregate_uncertainty(self, uncertainty_map: np.ndarray) -> np.ndarray:
        """
        Aggregates a 3D uncertainty map into a 2D map using the specified method.

        :param uncertainty_map: A 3D numpy array.
        :return: A 2D numpy array.
        """
        aggregation_methods = {
            'mean': np.mean,
            'max': np.max,
            'std': np.std
        }
        aggregated = aggregation_methods[self.aggregation_method](uncertainty_map, axis=-1)
        logging.debug("Aggregated uncertainty map using method '%s'.", self.aggregation_method)
        return aggregated.astype(np.float32)

    @staticmethod
    def _normalize_uncertainty(uncertainty_map: np.ndarray) -> np.ndarray:
        """
        Normalizes the uncertainty map to the range [0, 1].

        :param uncertainty_map: A 2D numpy array.
        :return: A normalized 2D numpy array.
        """
        min_val = uncertainty_map.min()
        max_val = uncertainty_map.max()
        if np.isclose(max_val, min_val):
            logging.warning("Uncertainty map has zero variance.")
            return np.zeros_like(uncertainty_map, dtype=np.float32)
        normalized = (uncertainty_map - min_val) / (max_val - min_val)
        logging.debug("Uncertainty map normalized to range [0, 1].")
        return normalized.astype(np.float32)

    def _identify_significant_coords(self, uncertainty_map: np.ndarray) -> List[Tuple[int, int]]:
        """
        Identifies significant coordinates based on local maxima in the uncertainty map.

        :param uncertainty_map: A 2D numpy array of normalized uncertainty values.
        :return: A list of (row, column) tuples representing coordinates.
        """
        # Apply maximum filter to find local maxima
        local_max = maximum_filter(uncertainty_map, size=self.filter_size, mode='constant') == uncertainty_map
        coords = np.argwhere(local_max)

        # Filter out points near the edges
        rows, cols = uncertainty_map.shape
        valid_mask = (
                (coords[:, 0] >= self.edge_buffer) &
                (coords[:, 0] < rows - self.edge_buffer) &
                (coords[:, 1] >= self.edge_buffer) &
                (coords[:, 1] < cols - self.edge_buffer)
        )
        valid_coords = coords[valid_mask]

        # Convert to list of tuples
        valid_coords = [tuple(coord) for coord in valid_coords]
        logging.debug("Filtered out coordinates near edges, %d remaining.", len(valid_coords))
        return valid_coords

    def _dbscan_cluster_coordinates(self, coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Clusters the spatial coordinates using DBSCAN to ensure spatial diversity.

        :param coords: A list of (row, column) tuples representing significant coordinates.
        :return: A list of (row, column) tuples after spatial clustering.
        """
        if not coords:
            return []

        # Convert list of tuples to NumPy array for DBSCAN
        spatial_coords = np.array(coords)  # Shape: (n_samples, 2)

        # Initialize DBSCAN with the specified parameters
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean', n_jobs=-1)

        # Fit DBSCAN on spatial coordinates
        dbscan.fit(spatial_coords)

        labels = dbscan.labels_
        unique_labels = set(labels)

        clustered_coords = []

        # Add coordinates from each cluster
        for label in unique_labels:
            if label == -1:
                # Skip noise points (DBSCAN label -1)
                continue
            cluster_indices = np.where(labels == label)[0]
            clustered_coords.extend([coords[idx] for idx in cluster_indices])

        logging.debug("DBSCAN clustering complete. Number of spatial clusters: %d.", len(unique_labels) - 1)
        return clustered_coords

    def _extract_logit_features(self, logits: np.ndarray, coords: List[Tuple[int, int]]) -> np.ndarray:
        """
        Extracts logit features at the specified coordinates.

        :param logits: A 3D numpy array of logit values per pixel.
        :param coords: A list of (row, column) tuples.
        :return: A 2D numpy array of logit features.
        """
        rows, cols = zip(*coords)
        logit_features = logits[rows, cols, :]  # Shape: (n_coords, logit_channels)
        return logit_features.astype(np.float32)  # Ensure float32 for consistency

    def cluster_logit_features(
            self,
            logit_features: np.ndarray,
            coords: List[Tuple[int, int]]
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Clusters the logit features using Agglomerative Clustering and maps them back to coordinates.

        :param logit_features: A 2D numpy array of logit features.
        :param coords: A list of (row, column) tuples.
        :return: A dictionary where keys are cluster labels and values are lists of (row, column) tuples.
        """
        if logit_features.size == 0:
            logging.warning("No logit features to cluster.")
            return {}

        # Initialize Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,  # Do not specify number of clusters
            distance_threshold=self.distance_threshold,  # Define the linkage distance threshold
            linkage=self.linkage
        )

        # Fit Agglomerative Clustering on logit features
        clustering.fit(logit_features)

        labels = clustering.labels_
        unique_labels = set(labels)

        clusters: Dict[int, List[Tuple[int, int]]] = {}
        for label in unique_labels:
            if label not in clusters:
                clusters[label] = []
            # Retrieve all coordinates belonging to this cluster
            cluster_indices = np.where(labels == label)[0]
            for idx in cluster_indices:
                clusters[label].append(coords[idx])

        logging.debug("Agglomerative clustering complete. Number of clusters: %d.", len(clusters))
        return clusters
