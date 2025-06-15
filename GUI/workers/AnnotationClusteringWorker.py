#!/usr/bin/env python3
"""
Point-annotation clustering in a background thread.

Pipeline
--------
1.  Build a feature matrix (optionally appending uncertainty).
2.  Core-set selection with farthest-first k-centre (Numba-accelerated).
3.  Run the chosen clustering backend on the core-set.
4.  Down-sample each cluster to *cluster_size* representatives.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Dict, Optional

import numba
import numpy as np
from PyQt5.QtCore import QObject, QRunnable, pyqtSignal
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from GUI.models.annotations import AnnotationBase


# -----------------------------------------------------------------------------
# NUMBA‑ACCELERATED HELPERS
# -----------------------------------------------------------------------------

@numba.njit
def _compute_initial_distances(X: np.ndarray, idx: int):
    diff = X - X[idx]
    return np.sqrt((diff * diff).sum(axis=1))


@numba.njit
def _update_distances(X: np.ndarray, dist: np.ndarray, idx: int):
    diff = X - X[idx]
    new = np.sqrt((diff * diff).sum(axis=1))
    for i in range(len(dist)):
        if new[i] < dist[i]:
            dist[i] = new[i]


@numba.njit
def k_center_greedy_numba(X: np.ndarray, k: int, random_state: Optional[int] = None):
    """Return *k* indices according to the greedy k‑centre heuristic (farthest‑first).

    ‑ *X* : (n,d) float32 array
    ‑ *k* : number of centres requested (k ≤ n expected)
    ‑ *random_state* : if ``None`` the current RNG state is used; otherwise it is
      used to seed *NumPy* inside numba for deterministic behaviour.
    """
    if random_state is not None:
        np.random.seed(random_state)
    n = X.shape[0]
    if k >= n:
        return np.arange(n)

    centres = np.empty(k, np.int64)
    centres[0] = np.random.randint(n)
    dist = _compute_initial_distances(X, centres[0])

    for j in range(1, k):
        centres[j] = np.argmax(dist)
        _update_distances(X, dist, centres[j])
    return centres


# -----------------------------------------------------------------------------
# QT SIGNALLING STRUCTURES
# -----------------------------------------------------------------------------

class ClusteringWorkerSignals(QObject):
    progress_updated = pyqtSignal(int)  # 0-100 or -1 for stage sentinels
    clustering_finished = pyqtSignal(dict)  # {cluster_id: [AnnotationBase]}
    cancelled = pyqtSignal()


# -----------------------------------------------------------------------------
# MAIN WORKER
# -----------------------------------------------------------------------------

class AnnotationClusteringWorker(QRunnable):
    """
    Cluster a large annotation pool using a greedy core-set plus a backend
    clustering algorithm, then down-sample each cluster.

    Parameters
    ----------
    annotations
        Full annotation list.
    subsample_ratio
        Fraction (0,1] of the pool to feed into core-set selection.
    cluster_method
        ``"minibatchkmeans"``, ``"agglomerative"``, or ``"gaussianmixture"``.
    cluster_size
        Return at most this many annotations per final cluster.
    include_uncertainty_in_feature_vector
        If True, append the uncertainty scalar to each feature vector.
    random_state
        Seed for deterministic behaviour (NumPy & scikit).
    """

    def __init__(
            self,
            *,
            annotations: List[AnnotationBase],
            subsample_ratio: float = 1.0,
            cluster_method: str = "minibatchkmeans",
            cluster_size: int = 6,
            include_uncertainty_in_feature_vector: bool = False,
            random_state: Optional[int] = 42,
    ):
        super().__init__()
        self.signals = ClusteringWorkerSignals()

        # user parameters
        self.annotations = annotations
        self.subsample_ratio = float(np.clip(subsample_ratio, 0.0, 1.0))
        self.cluster_method = cluster_method.lower()
        self.cluster_size = int(cluster_size)
        self.include_uncertainty = bool(include_uncertainty_in_feature_vector)
        self.random_state = random_state

        # runtime
        self._abort = False
        self.clustering_model = None  # optional: returned to caller if needed

    def cancel(self):
        """Called from Controller → stop ASAP."""
        self._abort = True

    def run(self):
        log = logging.getLogger(__name__)
        log.info("Worker started – %d annotations", len(self.annotations))

        if not self.annotations or self._abort:
            self.signals.cancelled.emit()
            return

        # 1 · Feature matrix
        try:
            feature_matrix = np.asarray(
                [self._anno_to_vec(a) for a in self.annotations], dtype=np.float32
            )
        except Exception as e:
            log.exception("feature-matrix build failed: %s", e)
            self.signals.cancelled.emit()
            return

        # stage sentinel
        self.signals.progress_updated.emit(-1)  # “core-set selection”

        # 2 · Core-set greedy k-centre
        if self._abort:
            self.signals.cancelled.emit();
            return

        subsz = max(1, int(len(self.annotations) * self.subsample_ratio))
        subsz = min(subsz, len(self.annotations))
        subs_i = np.random.choice(len(self.annotations), subsz, replace=False)
        subs_X = feature_matrix[subsz > 0 and subs_i]

        k_core = min(5_000, subs_X.shape[0])
        core_sub_idx = k_center_greedy_numba(subs_X, k_core, self.random_state)
        core_idx = subs_i[core_sub_idx]
        core_X = feature_matrix[core_idx]
        core_ann = [self.annotations[i] for i in core_idx]

        # stage sentinel
        self.signals.progress_updated.emit(-1)  # “clustering”

        if self._abort:
            self.signals.cancelled.emit();
            return

        # 3 · Clustering backend
        try:
            labels = self._cluster_core(core_X)
        except Exception as e:
            log.exception("clustering backend failed: %s", e)
            self.signals.cancelled.emit()
            return

        for lbl, ann in zip(labels, core_ann):
            ann.cluster_id = int(lbl)

        # 4 · Down-sample
        clusters = self._downsample_cluster(self._group_by_cluster(core_ann))

        if self._abort:
            self.signals.cancelled.emit()
        else:
            self.signals.progress_updated.emit(-1)  # done
            self.signals.clustering_finished.emit(clusters)

    # --------------------------------------------------- helpers
    def _anno_to_vec(self, ann: AnnotationBase) -> np.ndarray:
        if self.include_uncertainty:
            return np.concatenate([ann.logit_features, [ann.uncertainty]], dtype=np.float32)
        return ann.logit_features.astype(np.float32, copy=False)

    @staticmethod
    def _group_by_cluster(annos: List[AnnotationBase]) -> Dict[int, List[AnnotationBase]]:
        clust: Dict[int, List[AnnotationBase]] = defaultdict(list)
        for a in annos:
            clust[a.cluster_id].append(a)
        return clust

    # --------------- backend clustering -----------------
    def _cluster_core(self, X: np.ndarray):
        m = self.cluster_method
        rng = self.random_state

        if m == "minibatchkmeans":
            k = max(1, int(X.shape[0] * 0.10))
            km = MiniBatchKMeans(n_clusters=k, random_state=rng, batch_size=4096)
            lbl = km.fit_predict(X)
            self.clustering_model = km

        elif m == "agglomerative":
            ag = AgglomerativeClustering(n_clusters=None, distance_threshold=5.0, linkage="ward")
            lbl = ag.fit_predict(X)
            self.clustering_model = ag

        elif m == "gaussianmixture":
            k = max(1, int(X.shape[0] * 0.15))
            gm = GaussianMixture(n_components=k, random_state=rng)
            gm.fit(X)
            lbl = gm.predict(X)
            self.clustering_model = gm

        else:
            raise ValueError(f"Unknown clustering method '{m}'")

        logging.info("Backend produced %d clusters", len(set(lbl)))
        return lbl

    # --------------- down-sampling ----------------------
    def _downsample_cluster(self, clusters: Dict[int, List[AnnotationBase]]):
        final: Dict[int, List[AnnotationBase]] = {}
        for cid, annos in clusters.items():
            if len(annos) <= self.cluster_size:
                final[cid] = annos
                continue

            feats = np.vstack([self._anno_to_vec(a) for a in annos]).astype(np.float32)

            chosen = k_center_greedy_numba(
                feats, self.cluster_size, self.random_state
            )

            final[cid] = [annos[i] for i in chosen]
        return final
