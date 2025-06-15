import logging
from typing import List, Tuple

from .annotations import AnnotationBase, PointAnnotation

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

logger = logging.getLogger(__name__)  # module-level logger


# ----------------------------------------------------------------------------- #
#                               BASE CLASS
# ----------------------------------------------------------------------------- #
class BasePointAnnotationGenerator:
    """Abstract base class for generating point annotations."""

    def __init__(self, edge_buffer: int = 64):
        self.edge_buffer = edge_buffer
        logger.info(
            "%s initialised (edge_buffer=%d)",
            self.__class__.__name__, edge_buffer,
        )

    # ---------------------------- utilities --------------------------------- #
    @staticmethod
    def _prepare_uncertainty_map(uncertainty_map: np.ndarray) -> np.ndarray:
        """
        Reduce a 3-D uncertainty volume to 2-D (mean over last axis) or pass
        through a 2-D map.  Raises ``ValueError`` otherwise.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Preparing uncertainty map: dtype=%s, shape=%s",
                uncertainty_map.dtype, uncertainty_map.shape,
            )

        if uncertainty_map.ndim == 3:
            uncertainty_map_2d = np.mean(uncertainty_map, axis=-1)
            logger.debug("Aggregated 3-D map across axis −1 → shape %s",
                         uncertainty_map_2d.shape)
        elif uncertainty_map.ndim == 2:
            uncertainty_map_2d = uncertainty_map
            logger.debug("Uncertainty map already 2-D.")
        else:
            raise ValueError("Uncertainty map must be 2-D or 3-D.")

        return uncertainty_map_2d.astype(np.float32, copy=False)

    @staticmethod
    def _extract_logit_features(
            logits: np.ndarray, coords: List[tuple]
    ) -> np.ndarray:
        if not coords:
            logger.warning("No coordinates for logit extraction.")
            return np.array([])
        rows, cols = zip(*coords)
        feats = logits[rows, cols, :]  # (n_coords, n_classes)

        logger.debug(
            "Extracted logits @ %d coords → shape %s", len(coords), feats.shape
        )
        return feats.astype(np.float32, copy=False)

    # --------------------- public dispatcher -------------------------------- #
    def generate_annotations(
            self, uncertainty_map: np.ndarray, logits: np.ndarray
    ) -> List[AnnotationBase]:
        """Generate :class:`AnnotationBase` objects for the given inputs."""
        map2d = self._prepare_uncertainty_map(uncertainty_map)
        coords = self._generate_coords(map2d)

        if not coords:
            logger.warning("%s: no coordinates produced.", self.__class__.__name__)
            return []

        feats = self._extract_logit_features(logits, coords)
        annos: List[AnnotationBase] = []
        for c, f in zip(coords, feats):
            annos.append(
                PointAnnotation(
                    image_index=-1,
                    filename="",
                    coord=tuple(c),
                    logit_features=f,
                    uncertainty=float(map2d[tuple(c)]),
                )
            )

        logger.info("%s produced %d annotations.", self.__class__.__name__, len(annos))
        return annos

    # enforced in subclasses
    def _generate_coords(self, uncertainty_map: np.ndarray) -> List[tuple]:
        raise NotImplementedError


# ----------------------------------------------------------------------------- #
#                    LOCAL-MAXIMA IMPLEMENTATION
# ----------------------------------------------------------------------------- #
class LocalMaximaPointAnnotationGenerator(BasePointAnnotationGenerator):
    """
    Find local maxima in an uncertainty map (with optional Gaussian smoothing).
    """

    def __init__(
            self,
            filter_size: int = 48,
            gaussian_sigma: float = 4.0,
            edge_buffer: int = 64,
            use_gaussian: bool = False,
    ):
        super().__init__(edge_buffer=edge_buffer)

        # validate
        if filter_size <= 0:
            raise ValueError("filter_size must be positive.")
        if gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive.")
        if not isinstance(use_gaussian, bool):
            raise ValueError("use_gaussian must be a bool.")

        self.filter_size = filter_size
        self.gaussian_sigma = gaussian_sigma
        self.use_gaussian = use_gaussian

        logger.info(
            "LocalMaximaPointAnnotationGenerator(filter=%d, sigma=%.1f, "
            "use_gaussian=%s)",
            filter_size, gaussian_sigma, use_gaussian,
        )

    # --------------------------------------------------------------------- #
    def _generate_coords(self, uncertainty_map: np.ndarray) -> List[tuple]:
        if self.use_gaussian:
            processed = gaussian_filter(uncertainty_map, sigma=self.gaussian_sigma)
            logger.debug("Applied Gaussian σ=%.1f", self.gaussian_sigma)
        else:
            processed = uncertainty_map
            logger.debug("Gaussian smoothing disabled.")

        # local-max test
        local_max = maximum_filter(
            processed, size=self.filter_size, mode="constant"
        ) == processed
        coords = np.argwhere(local_max)

        # border mask
        r, c = uncertainty_map.shape
        mask = (
                (coords[:, 0] >= self.edge_buffer)
                & (coords[:, 0] < r - self.edge_buffer)
                & (coords[:, 1] >= self.edge_buffer)
                & (coords[:, 1] < c - self.edge_buffer)
        )
        before, after = coords.shape[0], mask.sum()
        coords = [tuple(x) for x in coords[mask]]

        logger.debug(
            "Local maxima: %d raw, %d after edge buffer (%d px).",
            before, after, self.edge_buffer,
        )
        return coords


# ----------------------------------------------------------------------------- #
#                   EQUIDISTANT GRID IMPLEMENTATION
# ----------------------------------------------------------------------------- #
class EquidistantPointAnnotationGenerator(BasePointAnnotationGenerator):
    """Uniform grid of points."""

    def __init__(self, grid_spacing: int = 64, edge_buffer: int = 64):
        super().__init__(edge_buffer=edge_buffer)
        if grid_spacing <= 0:
            raise ValueError("grid_spacing must be positive.")
        self.grid_spacing = grid_spacing
        logger.info(
            "EquidistantPointAnnotationGenerator(spacing=%d, edge_buffer=%d)",
            grid_spacing, edge_buffer,
        )

    # --------------------------------------------------------------------- #
    def _generate_coords(self, uncertainty_map: np.ndarray) -> List[tuple]:
        rows, cols = uncertainty_map.shape
        coords = [
            (r, c)
            for r in range(self.edge_buffer, rows - self.edge_buffer, self.grid_spacing)
            for c in range(self.edge_buffer, cols - self.edge_buffer, self.grid_spacing)
        ]
        logger.debug(
            "Equidistant grid: produced %d coords (spacing=%d).",
            len(coords), self.grid_spacing,
        )
        return coords


# ----------------------------------------------------------------------------- #
#                    CENTRE-ONLY IMPLEMENTATION
# ----------------------------------------------------------------------------- #
class CenterPointAnnotationGenerator(BasePointAnnotationGenerator):
    """Always return the geometric centre."""

    def __init__(self):
        super().__init__(edge_buffer=0)  # no margin needed

    # --------------------------------------------------------------------- #
    def _generate_coords(self, uncertainty_map: np.ndarray) -> List[Tuple[int, int]]:
        if uncertainty_map.ndim != 2:
            raise ValueError("uncertainty_map must be 2-D.")
        r, c = uncertainty_map.shape
        centre = (r // 2, c // 2)
        logger.debug("Centre point at %s", centre)
        return [centre]
