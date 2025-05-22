from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np


@dataclass
class Annotation:
    # --- required ------------------------------------------------------------------
    image_index: int
    filename: str
    coord: Tuple[int, int]
    logit_features: np.ndarray
    uncertainty: Union[float, np.ndarray]
    is_manual: bool = False
    class_id: Optional[int] = -1
    cluster_id: Optional[int] = None
    model_prediction: Optional[str] = None
    adjusted_uncertainty: Optional[Union[float, np.ndarray]] = None

    # ---------- core ---------------------------------------------------------------
    def __setattr__(self, name, value):
        if name == "uncertainty" and hasattr(self, "uncertainty"):
            raise AttributeError("`uncertainty` is read-only once set.")
        super().__setattr__(name, value)

    def __post_init__(self):
        if self.adjusted_uncertainty is None:
            self.adjusted_uncertainty = self.uncertainty

    # ---------- public helpers -----------------------------------------------------
    def reset_uncertainty(self) -> None:
        """Restore posterior to prior."""
        self.adjusted_uncertainty = self.uncertainty

    def to_dict(self) -> dict:
        """Converts this Annotation instance into a JSON-serialisable dict."""
        return {
            "image_index": int(self.image_index),
            "filename": self.filename,
            "coord": [int(c) for c in self.coord],
            "logit_features": self.logit_features.tolist(),
            "uncertainty": (
                self.uncertainty.tolist()
                if isinstance(self.uncertainty, np.ndarray)
                else float(self.uncertainty)
            ),
            "adjusted_uncertainty": (
                self.adjusted_uncertainty.tolist()
                if isinstance(self.adjusted_uncertainty, np.ndarray)
                else float(self.adjusted_uncertainty)
            ),
            "is_manual": bool(self.is_manual),
            "class_id": int(self.class_id) if self.class_id is not None else -1,
            "cluster_id": (
                int(self.cluster_id) if self.cluster_id is not None else None
            ),
            "model_prediction": self.model_prediction,
        }

    @staticmethod
    def from_dict(data: dict) -> "Annotation":
        """Creates an Annotation instance from a dictionary produced by ``to_dict``."""
        # --- uncertainty -----------------------------------------------------------
        uncertainty_value = data.get("uncertainty", 0.0)
        if isinstance(uncertainty_value, list):
            uncertainty_value = np.array(uncertainty_value)
        else:
            uncertainty_value = float(uncertainty_value)

        adjusted_uncertainty_value = data.get("adjusted_uncertainty", None)
        if adjusted_uncertainty_value is None:
            adjusted_uncertainty_value = uncertainty_value
        elif isinstance(adjusted_uncertainty_value, list):
            adjusted_uncertainty_value = np.array(adjusted_uncertainty_value)
        else:
            adjusted_uncertainty_value = float(adjusted_uncertainty_value)

        # --- other optional fields -------------------------------------------------
        cluster_id_raw = data.get("cluster_id", None)
        cluster_id_val = int(cluster_id_raw) if cluster_id_raw is not None else None

        is_manual_val = data.get("is_manual", False)

        # --- construct -------------------------------------------------------------
        return Annotation(
            image_index=int(data.get("image_index", -1)),
            filename=str(data.get("filename", "")),
            coord=tuple(data.get("coord", (0, 0))),
            logit_features=np.array(data.get("logit_features", [])),
            uncertainty=uncertainty_value,
            adjusted_uncertainty=adjusted_uncertainty_value,
            is_manual=is_manual_val,  # NEW FIELD
            class_id=int(data.get("class_id", -1)),
            cluster_id=cluster_id_val,
            model_prediction=data.get("model_prediction", None),
        )
