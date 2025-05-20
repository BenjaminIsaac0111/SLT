import logging
from collections import defaultdict
from typing import List, Dict, Optional

import numba
import numpy as np
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from GUI.models.Annotation import Annotation


# -----------------------------------------------------------------------------
# NUMBA‑ACCELERATED HELPERS
# -----------------------------------------------------------------------------

@numba.njit
def _compute_initial_distances(X: np.ndarray, center_index: int):
    """Return the Euclidean distance of every row in *X* to *X[center_index]*."""
    diff = X - X[center_index]
    return np.sqrt((diff * diff).sum(axis=1))


@numba.njit
def _update_distances(X: np.ndarray, dist_to_closest_center: np.ndarray, new_center_index: int):
    """In‑place relaxation: keep the smaller of the existing distance and the new one."""
    diff = X - X[new_center_index]
    new_dist = np.sqrt((diff * diff).sum(axis=1))
    # vectorised minimum – numba supports this broadcasting pattern
    for i in range(len(dist_to_closest_center)):
        if new_dist[i] < dist_to_closest_center[i]:
            dist_to_closest_center[i] = new_dist[i]


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

    centres = np.empty(k, dtype=np.int64)

    # 1. random initial centre
    centres[0] = np.random.randint(n)
    dists = _compute_initial_distances(X, centres[0])

    # 2. farthest‑first
    for idx in range(1, k):
        centres[idx] = np.argmax(dists)
        _update_distances(X, dists, centres[idx])

    return centres


# -----------------------------------------------------------------------------
# QT SIGNALLING STRUCTURES
# -----------------------------------------------------------------------------

class ClusteringWorkerSignals(QObject):
    """Signal bundle emitted by :class:`AnnotationClusteringWorker`."""

    clustering_finished = pyqtSignal(dict, object, object)  # clusters, model, core‑set features
    progress_updated = pyqtSignal(int)  # negative values = stage sentinel


# -----------------------------------------------------------------------------
# MAIN WORKER
# -----------------------------------------------------------------------------

class AnnotationClusteringWorker(QRunnable):
    """Cluster a large annotation pool via core‑set sampling then down‑sampling.

    Parameters
    ----------
    annotations : list[Annotation]
        Full pool.
    subsample_ratio : float, default 1.0
        Proportion of the pool to take as an *initial* random sub‑sample before
        the greedy selection.  Values in ``(0,1]``.  Set < 1.0 to cap memory/time.
    cluster_method : {"minibatchkmeans", "agglomerative", "gaussianmixture"}
        Backend algorithm applied on the core‑set.
    cluster_size : int, default 6
        How many elements to return per final cluster.
    include_uncertainty_in_feature_vector : bool, default True
        Whether to append the per‑annotation uncertainty score to the feature
        vector used *throughout* the pipeline.
    random_state : int | None, default 42
        Seed passed to :pyfunc:`k_center_greedy_numba`. ``None`` ⇒ stochastic.
    """

    def __init__(
            self,
            annotations: List[Annotation],
            subsample_ratio: float = 1.0,
            cluster_method: str = "minibatchkmeans",
            cluster_size: int = 8,
            *,
            include_uncertainty_in_feature_vector: bool = False,
            random_state: Optional[int] = 42,
    ):
        super().__init__()
        self.signals = ClusteringWorkerSignals()

        # user‑provided arguments
        self.annotations = annotations
        self.subsample_ratio = float(np.clip(subsample_ratio, 0.0, 1.0))  # guard
        self.cluster_method = cluster_method.lower()
        self.cluster_size = int(cluster_size)
        self.include_uncertainty = include_uncertainty_in_feature_vector
        self.random_state = random_state

        # runtime artefacts
        self.clustering_model = None

    # ------------------------------------------------------------------
    # QRunnable interface
    # ------------------------------------------------------------------

    def run(self):
        """Entry point executed inside the worker thread."""
        logging.info("AnnotationClusteringWorker started with %d annotations.", len(self.annotations))
        if not self.annotations:
            logging.warning("No annotations provided; aborting clustering.")
            self.signals.clustering_finished.emit({})
            return

        # ----------- Feature matrix construction ----------------------------------
        try:
            feature_matrix = np.asarray([
                self._annotation_to_vec(a) for a in self.annotations
            ], dtype=np.float32)
        except Exception as exc:
            logging.error("Failed to build feature matrix: %s", exc)
            self.signals.clustering_finished.emit({})
            return

        self.signals.progress_updated.emit(-1)  # stage: core‑set start

        # ----------- Core‑set greedy selection ------------------------------------
        core_set_cap = min(5_000, feature_matrix.shape[0])
        subsample_size = min(int(feature_matrix.shape[0] * self.subsample_ratio), core_set_cap)
        subsample_idx = np.random.choice(feature_matrix.shape[0], subsample_size, replace=False)
        subsample_X = feature_matrix[subsample_idx]

        actual_k = min(core_set_cap, subsample_size)
        if actual_k < core_set_cap:
            logging.warning("Sub‑sample (%d) smaller than core‑set cap (%d); core‑set will consist of %d items.",
                            subsample_size, core_set_cap, actual_k)

        centre_idx_sub = k_center_greedy_numba(
            subsample_X,
            actual_k,
            random_state=self.random_state,
        )
        core_set_idx = subsample_idx[centre_idx_sub]

        core_features = feature_matrix[core_set_idx]
        core_annotations = [self.annotations[i] for i in core_set_idx]
        self.signals.progress_updated.emit(-1)  # stage: clustering start

        # ----------- Clustering ----------------------------------------------------
        try:
            core_labels, cluster_centres = self._cluster_core_set(core_features)
        except Exception as exc:
            logging.exception("Clustering failed: %s", exc)
            self.signals.clustering_finished.emit({})
            return

        # propagate labels
        for lbl, anno in zip(core_labels, core_annotations):
            anno.cluster_id = int(lbl)

        # ----------- Down‑sampling -------------------------------------------------
        clusters = self._group_by_cluster(core_annotations)
        final_clusters = self._downsample_clusters(clusters, cluster_centres)

        self.signals.progress_updated.emit(-1)  # stage: done
        self.signals.clustering_finished.emit(final_clusters, self.clustering_model, core_features)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _annotation_to_vec(self, anno: Annotation) -> np.ndarray:
        """Convert *Annotation* → ndarray according to *include_uncertainty*."""
        if self.include_uncertainty:
            return np.concatenate([anno.logit_features, np.asarray([anno.uncertainty], dtype=np.float32)])
        return anno.logit_features.astype(np.float32, copy=False)

    # ......................................................

    @staticmethod
    def _group_by_cluster(annotations: List[Annotation]) -> Dict[int, List[Annotation]]:
        clusters: Dict[int, List[Annotation]] = defaultdict(list)
        for anno in annotations:
            clusters[anno.cluster_id].append(anno)
        return clusters

    # ......................................................

    def _cluster_core_set(self, X: np.ndarray):
        """Run the selected clustering backend on *X* and return (labels, centres|None)."""
        method = self.cluster_method
        logging.info("Clustering method: %s", method)

        if method == "minibatchkmeans":
            k = max(1, int(X.shape[0] * 0.10))  # 10% heuristic
            self.clustering_model = MiniBatchKMeans(n_clusters=k, random_state=self.random_state, batch_size=4096)
            labels = self.clustering_model.fit_predict(X)
            centres = self.clustering_model.cluster_centers_.astype(np.float32)

        elif method == "agglomerative":
            self.clustering_model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=5.0, linkage="ward"
            )
            labels = self.clustering_model.fit_predict(X)
            centres = None  # Ward does not expose explicit centroids.

        elif method == "gaussianmixture":
            k = max(1, int(X.shape[0] * 0.15))
            self.clustering_model = GaussianMixture(n_components=k, random_state=self.random_state)
            self.clustering_model.fit(X)
            labels = self.clustering_model.predict(X)
            centres = self.clustering_model.means_.astype(np.float32)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        logging.info("Clustering produced %d clusters.", len(set(labels)))
        return labels, centres

    # ......................................................

    def _downsample_clusters(self, clusters: Dict[int, List[Annotation]], centres: Optional[np.ndarray]):
        """Return a dict with at most *cluster_size* members in each cluster."""
        final: Dict[int, List[Annotation]] = {}

        for cid, annos in clusters.items():
            # trivial cases -------------------------------------------------
            if len(annos) <= self.cluster_size:
                final[cid] = annos
                continue

            # build feature block for selection
            feats = np.vstack([self._annotation_to_vec(a) for a in annos]).astype(np.float32)

            if centres is not None:
                centre = centres[cid]
                dists = np.linalg.norm(feats - centre, axis=1)
                chosen = np.argsort(dists)[: self.cluster_size]
            else:
                chosen = k_center_greedy_numba(
                    feats,
                    self.cluster_size,
                    random_state=self.random_state,
                )

            final[cid] = [annos[i] for i in chosen]

        return final
