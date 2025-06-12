from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import BaseImageDataModel
from GUI.models.ImageProcessor import ImageProcessor
from GUI.workers.OverlayPreviewWorker import OverlayPreviewWorker


class AnnotationPreviewDialog(QDialog):
    """Display full images with annotation overlays."""

    def __init__(self, model: BaseImageDataModel, annotations: List[Annotation], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Annotation Preview")

        self.model = model
        self.processor = ImageProcessor()
        self.threadpool = QThreadPool.globalInstance()

        self._by_image: Dict[int, List[Annotation]] = {}
        for ann in annotations:
            if ann.class_id == -1:
                continue
            self._by_image.setdefault(ann.image_index, []).append(ann)
        self._indices = sorted(self._by_image.keys())
        self._current_idx = 0
        self._image_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        layout.addLayout(self._create_legend())

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._update_opacity)
        layout.addWidget(self.opacity_slider)

        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.show_previous)
        self.next_btn.clicked.connect(self.show_next)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        layout.addLayout(btn_layout)

        self._load_current()

    # ------------------------------------------------------------------
    def _create_legend(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        for cid, name in CLASS_COMPONENTS.items():
            color = self.processor.class_color_map.get(cid, (255, 255, 255))
            swatch = QLabel()
            swatch.setFixedSize(20, 20)
            swatch.setStyleSheet(
                f"background-color: rgb({color[0]}, {color[1]}, {color[2]});"
            )
            layout.addWidget(swatch)
            layout.addWidget(QLabel(name))
        layout.addStretch(1)
        return layout

    # ------------------------------------------------------------------
    def _load_current(self) -> None:
        if not self._indices:
            self.image_label.setText("No labelled annotations")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
        idx = self._indices[self._current_idx]
        if idx in self._image_cache:
            self._display_cached(idx)
        else:
            anns = self._by_image[idx]
            worker = OverlayPreviewWorker(idx, anns, self.model, self.processor)
            worker.signals.finished.connect(self._on_overlay_ready)
            self.threadpool.start(worker)
            self.image_label.setText("Loading…")

    def _on_overlay_ready(self, image_index: int, base: np.ndarray, overlay: np.ndarray):
        self._image_cache[image_index] = (base, overlay)
        if self._indices[self._current_idx] == image_index:
            self._display_cached(image_index)

    def _display_cached(self, index: int) -> None:
        base, overlay = self._image_cache[index]
        pix = self._blend_images(base, overlay)
        self.image_label.setPixmap(pix)

    def _blend_images(self, base: np.ndarray, overlay: np.ndarray) -> QPixmap:
        alpha = self.opacity_slider.value() / 100.0
        arr = (base * (1 - alpha) + overlay * alpha).astype(np.uint8)
        return self.processor.numpy_to_qpixmap(arr)

    def _update_opacity(self, value: int) -> None:  # noqa: ARG002 - slot
        if not self._indices:
            return
        idx = self._indices[self._current_idx]
        if idx in self._image_cache:
            self._display_cached(idx)

    # ------------------------------------------------------------------
    def show_next(self) -> None:
        if not self._indices:
            return
        self._current_idx = (self._current_idx + 1) % len(self._indices)
        self._load_current()

    def show_previous(self) -> None:
        if not self._indices:
            return
        self._current_idx = (self._current_idx - 1) % len(self._indices)
        self._load_current()
