from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np


@dataclass
class Annotation:
    image_index: int
    filename: str
    coord: Tuple[int, int]
    logit_features: np.ndarray
    uncertainty: Union[float, np.ndarray]  # Allow uncertainty to be either a float or array
    class_id: Optional[int] = -1  # Default to -1 (Unlabelled)
    cluster_id: Optional[int] = None  # Default to None if not assigned

    def to_dict(self) -> dict:
        """
        Converts the Annotation instance to a dictionary for serialization.
        """
        data = {
            'filename': self.filename,
            'coord': [int(c) for c in self.coord],  # Convert to list of ints
            'logit_features': self.logit_features.tolist(),  # Convert NumPy array to list
            'class_id': int(self.class_id) if self.class_id is not None else -1,
            'image_index': int(self.image_index),
            'uncertainty': (
                self.uncertainty.tolist() if isinstance(self.uncertainty, np.ndarray)
                else float(self.uncertainty)  # Convert scalar to float
            ),
            'cluster_id': int(self.cluster_id) if self.cluster_id is not None else None,
        }
        return data

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
            uncertainty=(
                np.array(data['uncertainty']) if isinstance(data['uncertainty'], list)
                else float(data['uncertainty'])
            ),
            class_id=int(data.get('class_id', -1)),
            cluster_id=int(data.get('cluster_id', None)) if 'cluster_id' in data else None
        )
