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
    mask_rle: Optional[list] = None
    mask_shape: Optional[Tuple[int, int]] = None

    # ---------- core ---------------------------------------------------------------
    def __setattr__(self, name, value):
        if name == "uncertainty" and hasattr(self, "uncertainty"):
            raise AttributeError("`uncertainty` is read-only once set.")
        super().__setattr__(name, value)

    def __post_init__(self):
        if self.adjusted_uncertainty is None:
            self.adjusted_uncertainty = self.uncertainty

    # ---------- mask helpers -------------------------------------------------
    @staticmethod
    def encode_mask(mask: np.ndarray) -> list:
        """Return run-length encoding for ``mask``."""
        pixels = mask.astype(np.uint8).flatten(order="F")
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return runs.tolist()

    @staticmethod
    def decode_mask(rle: list, shape: Tuple[int, int]) -> np.ndarray:
        """Decode RLE ``rle`` back to a binary mask of ``shape``."""
        rle = np.asarray(rle, dtype=int)
        starts = rle[0::2] - 1
        lengths = rle[1::2]
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for s, e in zip(starts, ends):
            img[s:e] = 1
        return img.reshape(shape, order="F")

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
            "mask_rle": self.mask_rle,
            "mask_shape": list(self.mask_shape) if self.mask_shape else None,
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
        mask_rle_val = data.get("mask_rle")
        mask_shape_val = (
            tuple(data.get("mask_shape")) if data.get("mask_shape") else None
        )

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
            mask_rle=mask_rle_val,
            mask_shape=mask_shape_val,
        )
