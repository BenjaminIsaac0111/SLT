import logging
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


class UncertaintyRegionSelector:
    """
    The UncertaintyRegionSelector class identifies uncertain regions from an uncertainty map
    by selecting local maxima after optional Gaussian smoothing.
    """

    def __init__(
            self,
            filter_size: int = 48,
            gaussian_sigma: float = 3.0,
            edge_buffer: int = 64,
            use_gaussian: bool = False,
    ):
        """
        Initializes the UncertaintyRegionSelector.

        :param filter_size: Size for the maximum filter to identify local maxima.
        :param gaussian_sigma: Sigma parameter for the Gaussian filter.
        :param edge_buffer: Buffer from image edges to exclude certain coordinates.
        :param use_gaussian: Whether to apply Gaussian smoothing before finding local maxima.
        """
        self.filter_size = filter_size
        self.gaussian_sigma = gaussian_sigma
        self.edge_buffer = edge_buffer
        self.use_gaussian = use_gaussian
        self._validate_params()
        logging.info(
            "UncertaintyRegionSelector initialized with filter_size=%d, "
            "gaussian_sigma=%.2f, edge_buffer=%d, use_gaussian=%s.",
            filter_size, gaussian_sigma, edge_buffer, use_gaussian
        )

    def _validate_params(self):
        """
        Validates input parameters.
        """
        if not isinstance(self.filter_size, int) or self.filter_size <= 0:
            raise ValueError("filter_size must be a positive integer.")
        if not isinstance(self.gaussian_sigma, (int, float)) or self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be a positive float.")
        if not isinstance(self.edge_buffer, int) or self.edge_buffer < 0:
            raise ValueError("edge_buffer must be a non-negative integer.")
        if not isinstance(self.use_gaussian, bool):
            raise ValueError("use_gaussian must be a boolean.")

    def generate_point_labels(
            self,
            uncertainty_map: np.ndarray,
            logits: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Selects uncertain regions from the uncertainty map by identifying local maxima.
        Optionally applies Gaussian smoothing before selecting maxima.

        :param uncertainty_map: A 2D numpy array representing uncertainty values (already normalized).
                                If it is 3D, it will be averaged along the last axis to form 2D.
        :param logits: A 3D numpy array of logit values per pixel (H, W, num_classes).
        :return: A tuple containing logit features (numpy array) and a list of (row, column) tuples.
        """
        # Prepare uncertainty map (aggregate if 3D)
        uncertainty_map_2d = self._prepare_uncertainty_map(uncertainty_map)

        if self.use_gaussian:
            # Apply Gaussian smoothing
            uncertainty_map_processed = gaussian_filter(uncertainty_map_2d, sigma=self.gaussian_sigma)
            logging.debug("Applied Gaussian filter with sigma=%.2f.", self.gaussian_sigma)
        else:
            uncertainty_map_processed = uncertainty_map_2d
            logging.debug("Skipping Gaussian filter as per configuration.")

        # Identify local maxima as significant coordinates
        coords = self._identify_significant_coords(uncertainty_map_processed)
        logging.info("Identified %d significant coordinates.", len(coords))

        if not coords:
            logging.warning("No significant coordinates found.")
            return np.array([]), []

        # Extract logit features at these coordinates
        logit_features = self._extract_logit_features(logits, coords)
        logging.debug("Extracted logit features with shape %s.", logit_features.shape)

        return logit_features, coords

    def _prepare_uncertainty_map(self, uncertainty_map: np.ndarray) -> np.ndarray:
        """
        Prepares the uncertainty map for processing. If the uncertainty map is 3D, it aggregates it into 2D.

        :param uncertainty_map: A 2D or 3D numpy array.
        :return: A 2D numpy array.
        """
        if uncertainty_map.ndim == 3:
            # Aggregate the 3D uncertainty map into a 2D map using the mean
            uncertainty_map_2d = np.mean(uncertainty_map, axis=-1)
            logging.debug("Aggregated 3D uncertainty map into 2D using mean.")
        elif uncertainty_map.ndim == 2:
            uncertainty_map_2d = uncertainty_map
            logging.debug("Uncertainty map is already 2D.")
        else:
            raise ValueError("Uncertainty map must be a 2D or 3D numpy array.")
        return uncertainty_map_2d.astype(np.float32)

    def _identify_significant_coords(self, uncertainty_map: np.ndarray) -> List[tuple]:
        """
        Identifies significant coordinates based on local maxima in the uncertainty map.

        :param uncertainty_map: A 2D numpy array of uncertainty values.
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

    def _extract_logit_features(self, logits: np.ndarray, coords: List[Tuple[int, int]]) -> np.ndarray:
        """
        Extracts logit features at the specified coordinates.

        :param logits: A 3D numpy array of logit values per pixel (H, W, num_classes).
        :param coords: A list of (row, column) tuples.
        :return: A 2D numpy array of logit features.
        """
        if not coords:
            logging.warning("No coordinates provided for logit feature extraction.")
            return np.array([])

        rows, cols = zip(*coords)
        logit_features = logits[rows, cols, :]  # Shape: (n_coords, num_classes)
        return logit_features.astype(np.float32)  # Ensure float32 for consistency
