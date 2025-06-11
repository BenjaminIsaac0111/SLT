"""cluster_selectors.py
=====================

A small *strategy* framework for deciding **which cluster of image crops a
human annotator should label next**.

The module abstracts the decision policy away from the GUI–controller layer so
that different heuristics can be plugged‑in, unit‑tested, or even swapped at
runtime *without touching the user‑interface code*.

Key points
~~~~~~~~~~
* **Stateless (or minimally stateful) strategies** – each class implements a
  single public method :py:meth:`BaseClusterSelector.select_next`.
* **Factory pattern** – use :pyfunc:`make_selector` to obtain a strategy by
  name; no caller needs to import concrete classes.
* **Type‑safe** – the strategies depend only on a *protocol* comprising the two
  methods they actually need from the ``AnnotationClusteringController``.
  This makes static analysis and unit testing easier.

The default bundle includes three policies:

+---------------------------+----------------------------------------------+
| Name                      | Purpose                                      |
+===========================+==============================================+
| ``"greedy"`` (default)    | Exploits *uncertainty*, *class rarity*, and   |
|                           | *coverage* to maximise expected information. |
+---------------------------+----------------------------------------------+
| ``"sequential"``          | Deterministic walk‑through (ascending IDs).   |
+---------------------------+----------------------------------------------+
| ``"random"``              | Uniform baseline; useful for ablation tests. |
+---------------------------+----------------------------------------------+

Example
-------
from cluster_selectors import make_selector
selector = make_selector("greedy", clustering_controller)
next_id = selector.select_next(current_cluster_id)

Glossary
~~~~~~~~
``cluster``
    A set of spatial crop proposals (annotations) that the model believes are
    mutually similar. Each cluster is identified by an integer ID.
``class_id``
    Integer label assigned by a human annotator; ``-1`` denotes *unlabelled*.
``adjusted_uncertainty``
    Per‑annotation float updated by the active‑learning loop.

"
"""
from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
import scipy.spatial as spatial

__all__ = [
    "BaseClusterSelector",
    "GreedyClusterSelector",
    "SequentialClusterSelector",
    "RandomClusterSelector",
    "make_selector",
]


# -----------------------------------------------------------------------------
#  Controller protocol – narrows the API a selector relies on
# -----------------------------------------------------------------------------

class _ClusteringControllerProto(Protocol):
    """A minimal subset of the AnnotationClusteringController interface.

    Only two methods are required by all selector implementations.  By using a
    *protocol* instead of the concrete class, we keep the selectors independent
    of PyQt and facilitate unit testing with light‑weight stubs.
    """

    # pylint: disable=too-few-public-methods

    def get_clusters(self) -> Dict[int, Sequence]:
        """Return a mapping *cluster_id → sequence of Annotation objects*."""

    def get_class_id_from_prediction(self, model_prediction) -> int:
        """Map a raw model prediction to the corresponding *class_id*."""


# -----------------------------------------------------------------------------
#  Base class
# -----------------------------------------------------------------------------

class BaseClusterSelector(ABC):
    """Abstract base for *cluster‑selection* strategies.

    Sub‑classes must implement :py:meth:`select_next`.  Construction requires a
    reference to an object that satisfies
    :class:`~cluster_selectors._ClusteringControllerProto`.
    """

    def __init__(self, clustering_controller: _ClusteringControllerProto):
        if clustering_controller is None:
            raise ValueError("Cluster selector received a *None* controller.")
        self.ctrl: _ClusteringControllerProto = clustering_controller

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def select_next(self, current_cluster_id: Optional[int] = None) -> Optional[int]:
        """Return the ID of the **next** cluster to label.

        Parameters
        ----------
        current_cluster_id
            The cluster presently displayed in the UI.  When given, a selector
            should *exclude* that ID from its candidate pool so that the user
            is not told to label the same cluster twice in succession.

        Returns
        -------
        Optional[int]
            * ``int`` – a candidate cluster ID.
            * *None* – no suitable cluster remains (all labelled or none exist).
        """


# -----------------------------------------------------------------------------
#  Helper dataclass – specific to the Greedy strategy
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class _ClusterMetrics:
    """Container for the three raw features used by the Greedy strategy."""

    cid: int
    uncertainty: float
    rarity: float
    coverage: float

    @property
    def vec(self) -> np.ndarray:
        """Return the metric triple as a NumPy vector ``shape == (3,)``."""
        return np.asarray([self.uncertainty, self.rarity, self.coverage], dtype=np.float32)


# -----------------------------------------------------------------------------
#  Concrete strategies
# -----------------------------------------------------------------------------

class GreedyClusterSelector(BaseClusterSelector):
    """Heuristic that combines **uncertainty**, **rarity**, and **coverage**.

    Composite *z‑score* per cluster :math:`c`::

        z(U_c) + z(R_c) + z(C_c)

    where

    * ``U_c`` – mean *adjusted uncertainty* of **unlabelled** items within the
      cluster.
    * ``R_c`` – rarity = ``1 / (label_counts[majority_class] + 1)``.
    * ``C_c`` – Euclidean distance from the cluster centre to the *closest*
      **labelled** sample in feature space, encouraging exploration of poorly
      covered regions.
    """

    # ------------------------------------------------------------------
    #  Implementation of abstract method
    # ------------------------------------------------------------------

    def select_next(self, current_cluster_id: Optional[int] = None) -> Optional[int]:  # noqa: D401
        clusters = self.ctrl.get_clusters()
        if not clusters:
            return None

        candidates = self._filter_candidates(clusters, current_cluster_id)
        if not candidates:
            logging.info("[Greedy] no candidate clusters left.")
            return None

        label_counts, kd_tree = self._global_stats(clusters)
        metrics = [self._compute_metrics(cid, anns, label_counts, kd_tree)
                   for cid, anns in candidates.items()]
        return self._argmax(metrics)

    # ---------------- private helpers ---------------------------------

    @staticmethod
    def _filter_candidates(clusters: Dict[int, Sequence], current: Optional[int]):
        """Return clusters that (i) differ from *current* and (ii) contain ≥1 *unlabelled* item."""

        def has_unlab(annos):
            return any(a.class_id == -1 for a in annos)

        return {cid: a for cid, a in clusters.items() if cid != current and has_unlab(a)}

    def _global_stats(self, clusters):
        """Compute label frequency table & KD‑tree over *labelled* embeddings."""
        labelled = [a for annos in clusters.values() for a in annos
                    if a.class_id not in (-1, -2, -3)]
        label_counts = Counter(a.class_id for a in labelled) or {0: 1}
        kd_tree = (spatial.cKDTree(np.vstack([a.logit_features for a in labelled]))
                   if labelled else None)
        return label_counts, kd_tree

    # ..................................................................

    def _compute_metrics(self, cid, annos, label_counts, kd_tree):
        """Return the raw (U, R, C) triple for a single cluster."""
        unlab = [a for a in annos if a.class_id == -1]
        lab = [a for a in annos if a.class_id not in (-1, -2, -3)]

        # Uncertainty ---------------------------------------------------
        U = float(np.mean([a.adjusted_uncertainty for a in unlab]))

        # Rarity --------------------------------------------------------
        majority_cls = (
            Counter(a.class_id for a in lab).most_common(1)[0][0]
            if lab else
            Counter(self.ctrl.get_class_id_from_prediction(a.model_prediction)
                    for a in annos).most_common(1)[0][0]
        )
        R = 1.0 / (label_counts.get(majority_cls, 0) + 1)

        # Coverage ------------------------------------------------------
        centre = np.mean([a.logit_features for a in annos], axis=0)
        C = kd_tree.query(centre)[0] if kd_tree is not None else 0.0

        return _ClusterMetrics(cid, U, R, C)

    # ..................................................................

    @staticmethod
    def _argmax(metrics: List[_ClusterMetrics]):
        """Return the *cid* with the highest composite z‑score."""
        mat = np.vstack([m.vec for m in metrics])
        mean, std = mat.mean(axis=0), mat.std(axis=0, ddof=0)
        std[std == 0] = 1.0  # guard against zero variance

        best_cid, best_score = None, -np.inf
        for m in metrics:
            score = ((m.vec - mean) / std).sum()
            if score > best_score:
                best_cid, best_score = m.cid, score
        logging.debug("[Greedy] chosen cid=%s (%.3f)", best_cid, best_score)
        return best_cid


# -----------------------------------------------------------------------------

class SequentialClusterSelector(BaseClusterSelector):
    """Walk through clusters *in ascending numeric order* (wrap‑around at end)."""

    def __init__(self, clustering_controller: _ClusteringControllerProto):
        super().__init__(clustering_controller)
        self._cursor: Optional[int] = None  # remember last returned ID

    # ------------------------------------------------------------------
    #  Implementation of abstract method
    # ------------------------------------------------------------------

    def select_next(self, current_cluster_id: Optional[int] = None) -> Optional[int]:  # noqa: D401
        ids = sorted(self.ctrl.get_clusters())
        if not ids:
            return None

        # initialise or advance cursor ---------------------------------
        if self._cursor is None or self._cursor not in ids:
            self._cursor = ids[0]
        else:
            self._cursor = ids[(ids.index(self._cursor) + 1) % len(ids)]

        # avoid returning *current* twice in a row ----------------------
        if self._cursor == current_cluster_id and len(ids) > 1:
            self._cursor = ids[(ids.index(self._cursor) + 1) % len(ids)]
        return self._cursor


# -----------------------------------------------------------------------------

class RandomClusterSelector(BaseClusterSelector):
    """Pick an unlabelled cluster **uniformly at random**.

    This baseline is useful for A/B comparisons or as a pessimistic fallback
    when more sophisticated selectors fail.
    """

    def select_next(self, current_cluster_id: Optional[int] = None) -> Optional[int]:  # noqa: D401
        clusters = self.ctrl.get_clusters()
        candidates = [cid for cid, annos in clusters.items()
                      if cid != current_cluster_id and any(a.class_id == -1 for a in annos)]
        return random.choice(candidates) if candidates else None


# -----------------------------------------------------------------------------
#  Registry & factory
# -----------------------------------------------------------------------------

_REGISTRY = {
    "greedy": GreedyClusterSelector,
    "sequential": SequentialClusterSelector,
    "random": RandomClusterSelector,
}


def make_selector(name: str, ctrl: _ClusteringControllerProto) -> BaseClusterSelector:  # noqa: D401
    """Instantiate a selector by *registry name*.

    Parameters
    ----------
    name
        Identifier of the strategy – case insensitive.  Built‑in options are
        ``'greedy'``, ``'sequential'``, and ``'random'``.
    ctrl
        Object providing cluster/annotation access.  Any object that conforms
        to :class:`~cluster_selectors._ClusteringControllerProto` will work.

    Raises
    ------
    ValueError
        If *name* is not registered.
    """
    try:
        cls = _REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown selector '{name}'. Available: {', '.join(_REGISTRY)}"
        ) from exc
    return cls(ctrl)
