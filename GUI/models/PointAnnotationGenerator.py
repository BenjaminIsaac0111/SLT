import logging
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter, maximum_filter


class BasePointAnnotationGenerator:
    """
    Abstract base class for generating point annotations.
    It provides utility methods to prepare the uncertainty map and extract logit features.
    Subclasses must implement the `_generate_coords` method.
    """

    def __init__(self, edge_buffer: int = 64):
        self.edge_buffer = edge_buffer

    def _prepare_uncertainty_map(self, uncertainty_map: np.ndarray) -> np.ndarray:
        if uncertainty_map.ndim == 3:
            uncertainty_map_2d = np.mean(uncertainty_map, axis=-1)
            logging.debug("Aggregated 3D uncertainty map into 2D using mean.")
        elif uncertainty_map.ndim == 2:
            uncertainty_map_2d = uncertainty_map
            logging.debug("Uncertainty map is already 2D.")
        else:
            raise ValueError("Uncertainty map must be a 2D or 3D numpy array.")
        return uncertainty_map_2d.astype(np.float32)

    def _extract_logit_features(self, logits: np.ndarray, coords: List[Tuple[int, int]]) -> np.ndarray:
        if not coords:
            logging.warning("No coordinates provided for logit feature extraction.")
            return np.array([])
        rows, cols = zip(*coords)
        logit_features = logits[rows, cols, :]  # Shape: (n_coords, num_classes)
        return logit_features.astype(np.float32)

    def generate_annotations(self, uncertainty_map: np.ndarray, logits: np.ndarray) -> Tuple[ndarray, List[tuple]]:
        """
        Prepares the uncertainty map, generates coordinates using the subclass's method,
        and extracts logit features at these coordinates.
        """
        uncertainty_map_2d = self._prepare_uncertainty_map(uncertainty_map)
        coords = self._generate_coords(uncertainty_map_2d)
        if not coords:
            logging.warning("No coordinates found.")
            return np.array([]), []
        logit_features = self._extract_logit_features(logits, coords)
        return logit_features, coords

    def _generate_coords(self, uncertainty_map: np.ndarray) -> List[tuple]:
        raise NotImplementedError("Subclasses must implement _generate_coords.")


class LocalMaximaPointAnnotationGenerator(BasePointAnnotationGenerator):
    """
    Generates annotations by identifying local maxima in the uncertainty map.
    Optionally applies Gaussian smoothing prior to detection.
    """

    def __init__(self, filter_size: int = 48, gaussian_sigma: float = 4.0,
                 edge_buffer: int = 64, use_gaussian: bool = False):
        super().__init__(edge_buffer=edge_buffer)
        self.filter_size = filter_size
        self.gaussian_sigma = gaussian_sigma
        self.use_gaussian = use_gaussian

        # Parameter validation
        if not isinstance(filter_size, int) or filter_size <= 0:
            raise ValueError("filter_size must be a positive integer.")
        if not isinstance(gaussian_sigma, (int, float)) or gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be a positive float.")
        if not isinstance(use_gaussian, bool):
            raise ValueError("use_gaussian must be a boolean.")

        logging.info("LocalMaximaPointAnnotationGenerator initialized with filter_size=%d, "
                     "gaussian_sigma=%.2f, use_gaussian=%s, edge_buffer=%d.",
                     filter_size, gaussian_sigma, use_gaussian, edge_buffer)

    def _generate_coords(self, uncertainty_map: np.ndarray) -> List[tuple]:
        # Optionally apply Gaussian smoothing
        if self.use_gaussian:
            processed_map = gaussian_filter(uncertainty_map, sigma=self.gaussian_sigma)
            logging.debug("Applied Gaussian filter with sigma=%.2f.", self.gaussian_sigma)
        else:
            processed_map = uncertainty_map
            logging.debug("Skipping Gaussian filter as per configuration.")

        # Identify local maxima using a maximum filter
        local_max = maximum_filter(processed_map, size=self.filter_size, mode='constant') == processed_map
        coords = np.argwhere(local_max)

        rows, cols = uncertainty_map.shape
        valid_mask = (
                (coords[:, 0] >= self.edge_buffer) &
                (coords[:, 0] < rows - self.edge_buffer) &
                (coords[:, 1] >= self.edge_buffer) &
                (coords[:, 1] < cols - self.edge_buffer)
        )
        valid_coords = coords[valid_mask]
        valid_coords = [tuple(coord) for coord in valid_coords]
        logging.debug("Identified %d significant coordinates using local maxima.", len(valid_coords))
        return valid_coords


class EquidistantPointAnnotationGenerator(BasePointAnnotationGenerator):
    """
    Generates annotations by creating a uniform grid (equidistant spots)
    over the uncertainty map.
    """

    def __init__(self, grid_spacing: int = 64, edge_buffer: int = 64):
        super().__init__(edge_buffer=edge_buffer)
        self.grid_spacing = grid_spacing

        if not isinstance(grid_spacing, int) or grid_spacing <= 0:
            raise ValueError("grid_spacing must be a positive integer.")
        logging.info("EquidistantPointAnnotationGenerator initialized with grid_spacing=%d, edge_buffer=%d.",
                     grid_spacing, edge_buffer)

    def _generate_coords(self, uncertainty_map: np.ndarray) -> List[tuple]:
        rows, cols = uncertainty_map.shape
        coords = []
        for row in range(self.edge_buffer, rows - self.edge_buffer, self.grid_spacing):
            for col in range(self.edge_buffer, cols - self.edge_buffer, self.grid_spacing):
                coords.append((row, col))
        logging.debug("Generated equidistant grid with %d coordinates.", len(coords))
        return coords
