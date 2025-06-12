from __future__ import annotations

import logging
from typing import List

import numpy as np
from PyQt5.QtCore import QObject, QRunnable, pyqtSignal

from GUI.models.Annotation import Annotation
from GUI.models.CacheManager import CacheManager
from GUI.models.ImageDataModel import BaseImageDataModel
from GUI.models.ImageProcessor import ImageProcessor


class OverlayPreviewSignals(QObject):
    """Signals for :class:`OverlayPreviewWorker`."""

    finished = pyqtSignal(int, np.ndarray, np.ndarray)


class OverlayPreviewWorker(QRunnable):
    """Background task that renders annotation overlays."""

    cache = CacheManager()

    def __init__(
            self,
            image_index: int,
            annotations: List[Annotation],
            model: BaseImageDataModel,
            processor: ImageProcessor,
    ) -> None:
        super().__init__()
        self.image_index = image_index
        self.annotations = annotations
        self.model = model
        self.processor = processor
        self.signals = OverlayPreviewSignals()

    def run(self) -> None:
        key = (
            self.image_index,
            tuple((tuple(ann.coord), ann.class_id) for ann in self.annotations),
        )
        cached = self.cache.get(key)
        if cached is None:
            try:
                data = self.model.get_image_data(self.image_index)
            except Exception as exc:  # pragma: no cover - defensive
                logging.error("Failed to load image %d: %s", self.image_index, exc)
                return
            image = data.get("image")
            overlay_img = self.processor.create_annotation_overlay(
                image, self.annotations, show_labels=True
            )
            base = np.array(image)
            overlay = np.array(overlay_img.convert("RGB"))
            cached = (base, overlay)
            self.cache.set(key, cached)
        base, overlay = cached
        self.signals.finished.emit(self.image_index, base, overlay)
