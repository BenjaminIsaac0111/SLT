from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np


@dataclass
class Annotation:
    """
    Holds metadata for a single annotation on an image, including:
      - Image index and filename
      - Coordinate of the annotation
      - Logits and uncertainty information
      - Class/cluster assignments
      - Optional model prediction
      - Adjusted uncertainty (for propagating uncertainty reduction)
    """
    image_index: int
    filename: str
    coord: Tuple[int, int]
    logit_features: np.ndarray
    uncertainty: Union[float, np.ndarray]
    class_id: Optional[int] = -1  # Default to -1 (unlabeled)
    cluster_id: Optional[int] = None  # No cluster assignment by default
    model_prediction: Optional[str] = None  # Optional field for model's class prediction
    adjusted_uncertainty: Optional[Union[float, np.ndarray]] = None

    def __post_init__(self):
        # If no adjusted uncertainty is provided, initialize it as the original uncertainty.
        if self.adjusted_uncertainty is None:
            self.adjusted_uncertainty = self.uncertainty

    def to_dict(self) -> dict:
        """
        Converts this Annotation instance into a serializable dictionary.
        """
        return {
            'image_index': int(self.image_index),
            'filename': self.filename,
            'coord': [int(c) for c in self.coord],
            'logit_features': self.logit_features.tolist(),
            'uncertainty': (
                self.uncertainty.tolist() if isinstance(self.uncertainty, np.ndarray)
                else float(self.uncertainty)
            ),
            'adjusted_uncertainty': (
                self.adjusted_uncertainty.tolist() if isinstance(self.adjusted_uncertainty, np.ndarray)
                else float(self.adjusted_uncertainty)
            ),
            'class_id': int(self.class_id) if self.class_id is not None else -1,
            'cluster_id': (
                int(self.cluster_id) if self.cluster_id is not None else None
            ),
            'model_prediction': self.model_prediction,
        }

    @staticmethod
    def from_dict(data: dict) -> 'Annotation':
        """
        Creates an Annotation instance from a dictionary produced by to_dict().
        """
        uncertainty_value = data.get('uncertainty', 0.0)
        if isinstance(uncertainty_value, list):
            uncertainty_value = np.array(uncertainty_value)
        else:
            uncertainty_value = float(uncertainty_value)

        adjusted_uncertainty_value = data.get('adjusted_uncertainty', None)
        if adjusted_uncertainty_value is None:
            adjusted_uncertainty_value = uncertainty_value
        elif isinstance(adjusted_uncertainty_value, list):
            adjusted_uncertainty_value = np.array(adjusted_uncertainty_value)
        else:
            adjusted_uncertainty_value = float(adjusted_uncertainty_value)

        cluster_id_raw = data.get('cluster_id', None)
        cluster_id_val = int(cluster_id_raw) if cluster_id_raw is not None else None

        return Annotation(
            image_index=int(data.get('image_index', -1)),
            filename=str(data.get('filename', '')),
            coord=tuple(data.get('coord', (0, 0))),
            logit_features=np.array(data.get('logit_features', [])),
            uncertainty=uncertainty_value,
            adjusted_uncertainty=adjusted_uncertainty_value,
            class_id=int(data.get('class_id', -1)),
            cluster_id=cluster_id_val,
            model_prediction=data.get('model_prediction', None),
        )
