# models/UncertaintyRegionSelector.py

import logging
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.cluster import DBSCAN


class UncertaintyRegionSelector:
    """
    The UncertaintyRegionSelector class identifies uncertain regions from an uncertainty map.
    It uses image processing techniques to find significant points and then delegates clustering
    to a density-based clustering algorithm to ensure selected regions are diverse.
    """

    def __init__(
            self,
            filter_size: int = 64,
            aggregation_method: str = 'max',
            gaussian_sigma: float = 2.0,
            edge_buffer: int = 8,
            eps: float = 50.0,  # DBSCAN parameter: maximum distance between two samples
            min_samples: int = 1,  # DBSCAN parameter: minimum number of samples in a neighborhood
    ):
        """
        Initializes the UncertaintyRegionSelector.

        :param filter_size: Minimum distance between selected uncertain regions.
        :param aggregation_method: Method to aggregate 3D uncertainty map into 2D ('mean', 'max', 'std').
        :param gaussian_sigma: Sigma parameter for the Gaussian filter.
        :param edge_buffer: Buffer from image edges to exclude certain coordinates.
        :param eps: DBSCAN parameter for neighborhood size.
        :param min_samples: DBSCAN parameter for minimum samples in a neighborhood.
        """
        self.filter_size = filter_size
        self.aggregation_method = aggregation_method
        self.gaussian_sigma = gaussian_sigma
        self.edge_buffer = edge_buffer
        self.eps = eps
        self.min_samples = min_samples
        self._validate_params()
        logging.info(
            "UncertaintyRegionSelector initialized with filter_size=%d, aggregation_method='%s', "
            "gaussian_sigma=%.2f, eps=%.2f, min_samples=%d.",
            filter_size, aggregation_method, gaussian_sigma, eps, min_samples
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

    def select_regions(
            self,
            uncertainty_map: np.ndarray,
            logits: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Selects the uncertain regions from the uncertainty map.

        :param uncertainty_map: A 3D numpy array representing uncertainty values.
        :param logits: A 3D numpy array of logit values per pixel.
        :return: A list of (row, column) tuples representing the coordinates of selected regions.
        """
        # Step 1: Aggregate the 3D uncertainty map into a 2D map
        uncertainty_map_2d = self._aggregate_uncertainty(uncertainty_map)

        # Step 2: Apply Gaussian smoothing
        uncertainty_map_smoothed = gaussian_filter(uncertainty_map_2d, sigma=self.gaussian_sigma)
        logging.debug("Applied Gaussian filter with sigma=%.2f.", self.gaussian_sigma)

        # Step 3: Normalize the uncertainty map
        uncertainty_normalized = self._normalize_uncertainty(uncertainty_map_smoothed)

        # Step 4: Identify significant coordinates
        initial_coords = self._identify_significant_coords(uncertainty_normalized)
        logging.info("Identified %d significant coordinates.", len(initial_coords))

        if not initial_coords:
            logging.warning("No significant coordinates found.")
            return []

        # Step 5: Extract features for clustering (e.g., logits at the coordinates)
        features = self._extract_features(logits, initial_coords)

        # Optional: Reduce feature dimensionality for faster clustering
        # Uncomment if applicable
        # features = self._reduce_feature_dimensionality(features, n_components=3)

        # Step 6: Cluster the coordinates to ensure diversity using DBSCAN
        clustered_coords = self._cluster_regions(initial_coords, features)
        logging.info("Selected %d clustered coordinates.", len(clustered_coords))

        return clustered_coords

    def _aggregate_uncertainty(self, uncertainty_map: np.ndarray) -> np.ndarray:
        """
        Aggregates a 3D uncertainty map into a 2D map using the specified method.

        :param uncertainty_map: A 3D numpy array.
        :return: A 2D numpy array.
        """
        if self.aggregation_method == 'mean':
            aggregated = np.mean(uncertainty_map, axis=-1)
        elif self.aggregation_method == 'max':
            aggregated = np.max(uncertainty_map, axis=-1)
        elif self.aggregation_method == 'std':
            aggregated = np.std(uncertainty_map, axis=-1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        logging.debug("Aggregated uncertainty map using method '%s'.", self.aggregation_method)
        return aggregated

    @staticmethod
    def _normalize_uncertainty(uncertainty_map: np.ndarray) -> np.ndarray:
        """
        Normalizes the uncertainty map to the range [0, 1].

        :param uncertainty_map: A 2D numpy array.
        :return: A normalized 2D numpy array.
        """
        min_val = np.min(uncertainty_map)
        max_val = np.max(uncertainty_map)
        if max_val - min_val < 1e-8:
            logging.warning("Uncertainty map has zero variance.")
            return np.zeros_like(uncertainty_map)
        normalized = (uncertainty_map - min_val) / (max_val - min_val)
        logging.debug("Uncertainty map normalized to range [0, 1].")
        return normalized

    def _identify_significant_coords(self, uncertainty_map: np.ndarray) -> List[Tuple[int, int]]:
        """
        Identifies significant coordinates based on local maxima in the uncertainty map.

        :param uncertainty_map: A 2D numpy array of normalized uncertainty values.
        :return: A list of (row, column) tuples representing coordinates.
        """
        # Apply maximum filter to find local maxima
        footprint = np.ones((self.filter_size, self.filter_size))
        local_max = maximum_filter(uncertainty_map, footprint=footprint, mode='constant') == uncertainty_map
        coords = np.column_stack(np.nonzero(local_max))

        # Filter out points near the edges
        rows, cols = uncertainty_map.shape
        mask = (
                (coords[:, 0] >= self.edge_buffer) &
                (coords[:, 0] < rows - self.edge_buffer) &
                (coords[:, 1] >= self.edge_buffer) &
                (coords[:, 1] < cols - self.edge_buffer)
        )
        valid_coords = coords[mask]

        # Convert to list of tuples
        valid_coords = [tuple(coord) for coord in valid_coords]
        logging.debug("Filtered out coordinates near edges, %d remaining.", len(valid_coords))
        return valid_coords

    def _extract_features(self, logits: np.ndarray, coords: List[Tuple[int, int]]) -> np.ndarray:
        """
        Extracts features from logits at the specified coordinates.

        :param logits: A 3D numpy array of logit values per pixel.
        :param coords: A list of (row, column) tuples.
        :return: A 2D numpy array of features.
        """
        # Assuming logits shape is (height, width, channels)
        # Flatten the features for clustering
        features = np.array([logits[row, col, :] for row, col in coords])
        logging.debug("Extracted features for clustering.")
        return features

    def _cluster_regions(
            self,
            coords: List[Tuple[int, int]],
            features: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Clusters the coordinates to ensure diversity using DBSCAN.

        :param coords: A list of (row, column) tuples.
        :param features: A 2D numpy array of features.
        :return: A list of (row, column) tuples representing clustered coordinates.
        """
        if len(coords) == 0:
            return []

        # Initialize DBSCAN with the specified parameters
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean', n_jobs=-1)

        # Fit DBSCAN on spatial coordinates (not on features)
        spatial_features = np.array(coords)  # Shape: (n_samples, 2)
        dbscan.fit(spatial_features)

        labels = dbscan.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label if present

        clustered_coords = []
        for label in unique_labels:
            class_member_mask = (labels == label)
            cluster_coords = spatial_features[class_member_mask]

            # Select the point with the highest uncertainty within the cluster
            # Assuming that higher uncertainty corresponds to higher values in the aggregated uncertainty map
            # To implement this, you need to pass the aggregated uncertainty map or original uncertainty values
            # For simplicity, we'll select the first point
            selected_coord = tuple(cluster_coords[0])
            clustered_coords.append(selected_coord)

        # Handle noise points as individual regions
        noise_mask = (labels == -1)
        noise_coords = spatial_features[noise_mask]
        for coord in noise_coords:
            clustered_coords.append(tuple(coord))

        return clustered_coords
