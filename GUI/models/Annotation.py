# models/Annotation.py
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class Annotation:
    image_index: int
    filename: str
    coord: Tuple[int, int]
    logit_features: np.ndarray
    uncertainty: float
    class_id: Optional[int] = -1  # Default to -1 (Unlabelled)
    cluster_id: Optional[int] = None  # Default to None if not assigned

    def to_dict(self) -> dict:
        """
        Converts the Annotation instance to a dictionary for serialization.
        """
        return {
            'image_index': self.image_index,
            'filename': self.filename,
            'coord': list(self.coord),  # Convert tuple to list for JSON compatibility
            'logit_features': self.logit_features.tolist(),  # Convert numpy array to list
            'uncertainty': self.uncertainty,
            'class_id': self.class_id,
            'cluster_id': self.cluster_id
        }

    @staticmethod
    def from_dict(data: dict) -> 'Annotation':
        """
        Creates an Annotation instance from a dictionary.
        """
        return Annotation(
            image_index=int(data.get('image_index', -1)),
            filename=str(data.get('filename', '')),
            coord=tuple(data.get('coord', (0, 0))),
            logit_features=np.array(data.get('logit_features', [])),
            uncertainty=float(data.get('uncertainty', 0.0)),
            class_id=int(data.get('class_id', -1)),
            cluster_id=int(data.get('cluster_id', None)) if 'cluster_id' in data else None
        )
