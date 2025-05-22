#!/usr/bin/env python3
"""
Annotation-Clustering Controller
================================
A single self-contained module that

1.  extracts point annotations from every image (cancellable QThread);
2.  clusters those points (cancellable QRunnable on the global QThreadPool);
3.  keeps ``self.clusters`` up-to-date and emits basic statistics.

All UI feedback is provided through the *signals* declared at the top.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import (
    QObject, QThread, QThreadPool, pyqtSignal, pyqtSlot
)
from sklearn.utils.class_weight import compute_class_weight

from GUI.configuration.configuration import CLASS_COMPONENTS
from GUI.models.Annotation import Annotation
from GUI.models.ImageDataModel import BaseImageDataModel
from GUI.models.PointAnnotationGenerator import CenterPointAnnotationGenerator
from GUI.workers.AnnotationClusteringWorker import AnnotationClusteringWorker


# ──────────────────────────────── 1 · extraction worker ──────────────────
class AnnotationExtractionWorker(QThread):
    """Per-image sampling; terminates early when `cancel()` is called."""

    progress = pyqtSignal(int)  # 0–100
    finished = pyqtSignal(list)  # list[Annotation]
    cancelled = pyqtSignal()  # early abort

    def __init__(self, model: BaseImageDataModel, generator, parent=None):
        super().__init__(parent)
        self._model = model
        self._generator = generator
        self._abort = False

    # public ----------------------------------------------------
    def cancel(self):
        self._abort = True

    # worker entry ----------------------------------------------
    def run(self):
        annos: list[Annotation] = []
        n_images = self._model.get_number_of_images()
        for idx in range(n_images):
            if self._abort:
                self.cancelled.emit()
                return

            img = self._model.get_image_data(idx)
            annos.extend(self._extract_from_image(img, idx))
            self.progress.emit(int((idx + 1) / n_images * 100))

        self.finished.emit(annos)

    # helpers ---------------------------------------------------
    def _extract_from_image(self, img: dict, idx: int) -> list[Annotation]:
        out: list[Annotation] = []
        umap, logits, fname = img.get("uncertainty"), img.get("logits"), img.get("filename")
        if umap is None or logits is None or fname is None:
            return out

        feats, coords = self._generator.generate_annotations(uncertainty_map=umap, logits=logits)
        for c, f in zip(coords, feats):
            if not f.any():
                continue
            out.append(
                Annotation(
                    filename=fname,
                    coord=c,
                    logit_features=f,
                    class_id=-1,
                    image_index=idx,
                    uncertainty=float(umap[tuple(c)]),
                    cluster_id=None,
                    model_prediction=CLASS_COMPONENTS.get(int(np.argmax(f)), "None"),
                )
            )
        return out


# ──────────────────────────────── 2 · controller ─────────────────────────
class AnnotationClusteringController(QObject):
    """
    Drives extraction → clustering → statistics and supports cancellation.
    """

    # progress / life-cycle
    clustering_started = pyqtSignal()
    annotation_progress = pyqtSignal(int)
    annotation_progress_finished = pyqtSignal()
    clustering_progress = pyqtSignal(int)  # -1 ⇒ “busy” bar
    clusters_ready = pyqtSignal(dict)  # {cid: [Annotation]}
    cancelled = pyqtSignal()

    # analytics
    labeling_statistics_updated = pyqtSignal(dict)

    # init -------------------------------------------------------
    def __init__(self, model: BaseImageDataModel, annotation_generator):
        super().__init__()
        self.model = model
        self.annotation_generator = annotation_generator

        self.clusters: Dict[int, List[Annotation]] = {}
        self.cluster_labels: Dict[int, str] = {}

        self._extract_worker: Optional[AnnotationExtractionWorker] = None
        self._cluster_worker: Optional[AnnotationClusteringWorker] = None

        self._abort = False
        self._in_progress = False
        self._pool = QThreadPool.globalInstance()

    # ---------------------------------------------------------------- API
    @pyqtSlot()
    def start_clustering(self):
        """Kick off a new job if none is currently running."""
        if self._in_progress:
            logging.warning("Clustering already running.")
            return

        logging.info("Clustering job started.")
        self._abort = False
        self._in_progress = True
        self.clustering_started.emit()

        self._extract_worker = AnnotationExtractionWorker(self.model, self.annotation_generator)
        self._extract_worker.progress.connect(self.annotation_progress)
        self._extract_worker.finished.connect(self._on_annotations_extracted)
        self._extract_worker.cancelled.connect(self._on_cancelled)

        self.annotation_progress.emit(0)
        self._extract_worker.start()

    @pyqtSlot()
    def cancel_clustering(self):
        """User abort from the GUI."""
        if not self._in_progress:
            return
        self._abort = True
        if self._extract_worker and self._extract_worker.isRunning():
            self._extract_worker.cancel()
        if self._cluster_worker:
            self._cluster_worker.cancel()

    # ─────────────────── extraction callbacks ───────────────────
    @pyqtSlot(list)
    def _on_annotations_extracted(self, annos: List[Annotation]):
        self.annotation_progress.emit(100)
        self.annotation_progress_finished.emit()

        if self._abort:
            self._on_cancelled();
            return

        # centre-only generator → bypass clustering
        if isinstance(self.annotation_generator, CenterPointAnnotationGenerator):
            self.clusters = {i: [a] for i, a in enumerate(annos)}
            self._finish_success();
            return

        # kick off clustering worker
        self.clustering_progress.emit(-1)
        self._cluster_worker = AnnotationClusteringWorker(
            annotations=annos,
            subsample_ratio=1.0,
            cluster_method="minibatchkmeans",
        )
        self._cluster_worker.signals.progress_updated.connect(self.clustering_progress.emit)
        self._cluster_worker.signals.clustering_finished.connect(self._on_clustering_finished)
        self._cluster_worker.signals.cancelled.connect(self._on_cancelled)
        self._pool.start(self._cluster_worker)

    # ─────────────────── clustering callbacks ───────────────────
    @pyqtSlot(dict)
    def _on_clustering_finished(self, clusters: Dict[int, List[Annotation]]):
        if self._abort:
            self._on_cancelled();
            return
        self.clusters = clusters
        self._finish_success()

    # ───────────────── unified paths ─────────────────────────────
    def _finish_success(self):
        self._in_progress = False
        self.clusters_ready.emit(self.clusters)
        self._emit_label_stats()

    @pyqtSlot()
    def _on_cancelled(self):
        if not self._in_progress:
            return
        logging.info("Clustering job cancelled.")
        self._in_progress = False
        self.cancelled.emit()

    # ───────────────────────── statistics ───────────────────────
    def _emit_label_stats(self):
        self.labeling_statistics_updated.emit(self.compute_labeling_statistics())

    def compute_labeling_statistics(self) -> dict:
        total = sum(len(v) for v in self.clusters.values())
        class_counts = {cid: 0 for cid in CLASS_COMPONENTS}
        class_counts.update({-1: 0, -2: 0, -3: 0})

        labeled, disagreements = 0, 0
        y_all: list[int] = []

        for annos in self.clusters.values():
            for a in annos:
                cid = a.class_id if a.class_id is not None else -1
                class_counts[cid] = class_counts.get(cid, 0) + 1
                y_all.append(cid)

                if cid not in (-1, -2, -3):
                    labeled += 1
                    pred = self.get_class_id_from_prediction(a.model_prediction)
                    if pred is not None and pred != cid:
                        disagreements += 1

        agree_pct = 0.0 if labeled == 0 else (labeled - disagreements) / labeled * 100
        weights = {}
        filtered = [c for c in y_all if c not in (-1, -2, -3)]
        if filtered:
            cls = np.unique(filtered)
            weights = dict(zip(cls, compute_class_weight("balanced", classes=cls, y=filtered)))

        return {
            "total_annotations": total,
            "total_labeled": labeled,
            "class_counts": class_counts,
            "disagreement_count": disagreements,
            "agreement_percentage": agree_pct,
            "class_weights": weights,
        }

    # ─────────────────────── cluster meta ───────────────────────
    def generate_cluster_info(self) -> Dict[int, dict]:
        """Return light-weight per-cluster metadata for the UI."""
        info = OrderedDict()
        for cid, annos in self.clusters.items():
            num = len(annos)
            labeled = sum(1 for a in annos if a.class_id != -1)
            info[cid] = {
                "num_annotations": num,
                "num_images": len({a.filename for a in annos}),
                "labeled_percentage": (labeled / num * 100) if num else 0.0,
                "label": self.cluster_labels.get(cid, ""),
                "average_adjusted_uncertainty": float(np.mean([a.adjusted_uncertainty for a in annos] or [0])),
                "average_uncertainty": float(np.mean([a.uncertainty for a in annos] or [0])),
            }
        return info

    # ───────────────────────── utilities ────────────────────────
    def update_cluster_labels(self, cluster_id: int, label: str):
        self.cluster_labels[cluster_id] = label

    def get_clusters(self) -> Dict[int, List[Annotation]]:
        return self.clusters

    def cleanup(self):
        """Called by MainController on app exit."""
        self.cancel_clustering()  # best-effort; threads quit on their own

    # ───────────────────────── helpers ──────────────────────────
    @staticmethod
    def get_class_id_from_prediction(pred: str) -> Optional[int]:
        for cid, name in CLASS_COMPONENTS.items():
            if name == pred:
                return cid
        return None
