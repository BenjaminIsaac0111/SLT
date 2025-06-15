import logging
from typing import List, Tuple

import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops

from .PointAnnotationGenerator import BasePointAnnotationGenerator
from .Annotation import Annotation

logger = logging.getLogger(__name__)


class SLICSuperpixelGenerator(BasePointAnnotationGenerator):
    """Generate superpixel regions ranked by uncertainty."""

    def __init__(self, n_segments: int = 250, compactness: float = 0.1, edge_buffer: int = 64):
        super().__init__(edge_buffer=edge_buffer)
        self.n_segments = n_segments
        self.compactness = compactness
        logger.info(
            "SLICSuperpixelGenerator(n_segments=%d, compactness=%.2f)",
            n_segments,
            compactness,
        )

    def generate_annotations(
        self, uncertainty_map: np.ndarray, logits: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], List[list]]:
        map2d = self._prepare_uncertainty_map(uncertainty_map)
        segments = slic(map2d, n_segments=self.n_segments, compactness=self.compactness, start_label=1)
        props = regionprops(segments, intensity_image=map2d)
        props = sorted(props, key=lambda p: p.mean_intensity, reverse=True)

        coords: List[Tuple[int, int]] = []
        feats: List[np.ndarray] = []
        masks: List[list] = []
        for p in props:
            cy, cx = map(int, p.centroid)
            if (
                cy < self.edge_buffer
                or cy >= map2d.shape[0] - self.edge_buffer
                or cx < self.edge_buffer
                or cx >= map2d.shape[1] - self.edge_buffer
            ):
                continue
            mask = segments == p.label
            coords.append((cy, cx))
            feats.append(logits[mask].mean(axis=0))
            masks.append(Annotation.encode_mask(mask))

        if not coords:
            logger.warning("SLICSuperpixelGenerator produced no annotations")
            return np.empty((0, logits.shape[-1]), dtype=np.float32), [], []

        return np.stack(feats).astype(np.float32), coords, masks

