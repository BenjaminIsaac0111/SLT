from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Protocol

import numpy as np


class AnnotationBase(Protocol):
    """Protocol for annotation objects."""

    image_index: int
    filename: str
    coord: Tuple[int, int]
    logit_features: np.ndarray
    uncertainty: Union[float, np.ndarray]
    is_manual: bool
    class_id: Optional[int]
    cluster_id: Optional[int]
    model_prediction: Optional[str]
    adjusted_uncertainty: Optional[Union[float, np.ndarray]]

    def to_dict(self) -> dict:
        ...


@dataclass
class PointAnnotation:
    """Simple point annotation."""

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

    def __post_init__(self) -> None:
        if self.adjusted_uncertainty is None:
            self.adjusted_uncertainty = self.uncertainty

    # ------------------------------------------------------------------
    def reset_uncertainty(self) -> None:
        """Restore ``adjusted_uncertainty`` to the original value."""
        self.adjusted_uncertainty = self.uncertainty

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "type": "point",
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
            "cluster_id": int(self.cluster_id) if self.cluster_id is not None else None,
            "model_prediction": self.model_prediction,
        }


@dataclass
class MaskAnnotation(PointAnnotation):
    """Annotation represented by a segmentation mask."""

    mask: np.ndarray | None = None

    def to_dict(self) -> dict:  # type: ignore[override]
        d = super().to_dict()
        d["type"] = "mask"
        d["mask"] = self.mask.tolist() if self.mask is not None else []
        return d


# ---------------------------------------------------------------------------
#  Factory helper
# ---------------------------------------------------------------------------

def annotation_from_dict(data: dict) -> AnnotationBase:
    """Create :class:`PointAnnotation` or :class:`MaskAnnotation` from ``data``."""
    ann_type = data.get("type", "point")

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

    common = dict(
        image_index=int(data.get("image_index", -1)),
        filename=str(data.get("filename", "")),
        coord=tuple(data.get("coord", (0, 0))),
        logit_features=np.array(data.get("logit_features", [])),
        uncertainty=uncertainty_value,
        adjusted_uncertainty=adjusted_uncertainty_value,
        is_manual=bool(data.get("is_manual", False)),
        class_id=int(data.get("class_id", -1)),
        cluster_id=data.get("cluster_id", None),
        model_prediction=data.get("model_prediction", None),
    )

    if ann_type == "mask":
        mask_array = np.array(data.get("mask", []))
        return MaskAnnotation(mask=mask_array, **common)

    return PointAnnotation(**common)
