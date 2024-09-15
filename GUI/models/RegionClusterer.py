# models/RegionClusterer.py

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import logging


class RegionClusterer:
    """
    RegionClusterer clusters uncertain regions based on features to ensure diversity among selected regions.
    """

    def __init__(self, distance_threshold: float = 1.0):
        """
        Initializes the RegionClusterer.

        :param distance_threshold: The linkage distance threshold for clustering.
        """
        self.distance_threshold = distance_threshold
        logging.info("RegionClusterer initialized with distance_threshold=%.2f.", distance_threshold)

    def cluster_regions(
        self,
        coords: List[Tuple[int, int]],
        features: np.ndarray,
        num_clusters: int
    ) -> List[Tuple[int, int]]:
        """
        Clusters the regions based on features and selects one coordinate per cluster.

        :param coords: A list of (row, column) tuples representing coordinates.
        :param features: A 2D numpy array of features corresponding to the coords.
        :param num_clusters: Desired number of clusters.
        :return: A list of (row, column) tuples representing clustered coordinates.
        """
        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        logging.debug("Features standardized for clustering.")

        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=num_clusters,
            linkage='ward'
        ).fit(features_scaled)
        labels = clustering.labels_
        logging.info("Clustering completed with %d clusters.", len(np.unique(labels)))

        # Select one coordinate per cluster (e.g., the one with the highest uncertainty)
        clustered_coords = []
        for label in np.unique(labels):
            indices = np.where(labels == label)[0]
            selected_index = indices[0]  # You can apply a selection criterion here
            clustered_coords.append(coords[selected_index])
            logging.debug("Cluster %d: selected coordinate %s.", label, coords[selected_index])

        return clustered_coords


