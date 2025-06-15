"""uncertainty_propagation
=================================
Distance–based uncertainty propagation for interactive labelling
----------------------------------------------------------------
This module implements a light‑weight but expressive strategy for updating the
*uncertainty* values that drive active‑learning user interfaces.  Starting from
an immutable *prior* uncertainty supplied by the upstream model, every
annotation receives a *posterior* value that depends on its proximity to any
currently labelled example.

Algorithmic overview
~~~~~~~~~~~~~~~~~~~~
Given a collection of feature vectors :math:`\{x_i\}_{i=1}^{n}` (one per point)
and a subset of *labelled* vectors :math:`L \subseteq \{1,\ldots,n\}`, we define

.. math::

    u_i^{\text{post}} \;=\; u_i^{\text{prior}} \;\bigl(1 - e^{-\lambda \; d_i^2 }\bigr),

where

* :math:`u_i^{\text{prior}} \in [0,1]` is the immutable baseline uncertainty that
  came with the model prediction;
* :math:`\lambda` is a scale parameter (``lambda_param`` in the code) controlling
  how quickly evidence decays with distance;
* :math:`d_i^2 = \min_{j \in L} \lVert x_i - x_j \rVert_2^2` is the squared
  Euclidean distance to the nearest *labelled* neighbour and is truncated by a
  user‑supplied ``threshold`` to avoid numerical underflow for very distant
  points.

This simple Gaussian kernel enjoys three convenient properties that are often
expected from annotation tools:

* **Monotonic decrease** – posterior uncertainty never exceeds the prior
  (:math:`u_i^{\text{post}} \le u_i^{\text{prior}}`).
* **Reversibility** – removing all labels restores the original uncertainty
  field.
* **Idempotence** – repeated propagation with an unchanged label set leaves the
  field unchanged.

The heavy‑lifting inner loop is compiled just‑in‑time with *Numba* and is fast
enough for real‑time GUI use on tens of thousands of points.

Notes for developers
~~~~~~~~~~~~~~~~~~~~
*   The *prior* uncertainty must already be present on the :class:`~GUI.models.Annotation.Annotation`
    objects (attribute ``uncertainty``).  This module never overwrites it.
*   The *posterior* is written back to ``Annotation.adjusted_uncertainty``.
    Down‑stream widgets are expected to read that field.
*   All feature arrays are required to be ``np.float32``; silent copies are made
    when necessary, but paying attention to the dtype can save memory
    allocations during tight interaction loops.
*   A convenience helper :func:`propagate_for_annotations` wraps everything
    together for typical GUI usage.  Advanced users can directly instantiate
    the propagator classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from GUI.models.annotations import AnnotationBase

__all__ = [
    "auto_lambda",
    "BaseUncertaintyPropagator",
    "DistanceBasedPropagator",
    "propagate_for_annotations",
]


# -----------------------------------------------------------------------------
# Helper: data‑driven kernel scale
# -----------------------------------------------------------------------------


def auto_lambda(
        feats: np.ndarray,
        k: Optional[int] = None,
) -> float:
    """Estimate a sensible :math:`\lambda` from the data.

    The rule‑of‑thumb uses the *median* distance to the *k‑th* nearest
    neighbour across the whole feature cloud and sets

    .. math:: \lambda = 1 / \operatorname{median}(d_k)^2 .

    On isotropic data this heuristic leads to a kernel whose half‑width is on
    the order of the local density of points; on highly anisotropic manifolds
    you may want to tune :pydata:`k` manually.

    Parameters
    ----------
    feats
        An ``(n, d)`` matrix of *float32* feature vectors.
    k
        Index of the neighbour to consider.  If *None*, defaults to
        ``ceil(log2(n))`` which grows slowly and works well in practice.

    Returns
    -------
    float
        ``lambda_param`` suitable for the Gaussian kernel.
    """
    # choose k adaptively once
    n = feats.shape[0]
    if k is None:
        k = max(1, int(np.ceil(np.log2(n))))

    # fast exact k‑NN search
    tree = cKDTree(feats)
    # query returns the distance to each of the first k+1 neighbours
    dists, _ = tree.query(feats, k=k + 1)  # include self at column 0
    sigma = np.median(dists[:, -1])  # k‑th neighbour

    return 1.0 / sigma ** 2  # λ = 1/σ²


# -----------------------------------------------------------------------------
# NUMBA kernel
# -----------------------------------------------------------------------------

@njit(cache=True)
def _numba_distance_kernel(
        unlabeled_features: np.ndarray,
        labeled_features: np.ndarray,
        baseline_u: np.ndarray,
        lambda_param: float,
        threshold: float,
) -> np.ndarray:  # pragma: no cover – compiled
    """Numba‑accelerated Gaussian‑kernel posterior.

    Parameters
    ----------
    unlabeled_features
        Feature matrix of *all* points (labelled or not).
    labeled_features
        Feature matrix containing only the *currently labelled* points.
    baseline_u
        Prior uncertainties, same length as ``unlabeled_features``.
    lambda_param
        Pre‑computed kernel scale ``λ``.
    threshold
        Maximum radius to consider when searching the nearest label.  Anything
        beyond contributes zero evidence.

    Returns
    -------
    np.ndarray
        Posterior uncertainties for **all** points (labelled points will be
        overwritten by the caller).
    """
    n_unlabeled, d = unlabeled_features.shape
    n_labeled = labeled_features.shape[0]

    # No labels: return the prior unchanged
    if n_labeled == 0:
        return baseline_u.copy()

    out = np.empty(n_unlabeled, dtype=baseline_u.dtype)
    thr2 = threshold * threshold

    for i in range(n_unlabeled):
        min_d2 = thr2
        for j in range(n_labeled):
            s = 0.0
            for k in range(d):
                diff = unlabeled_features[i, k] - labeled_features[j, k]
                s += diff * diff
            if s < min_d2:
                min_d2 = s

        similarity = np.exp(-lambda_param * min_d2)
        out[i] = baseline_u[i] * (1.0 - similarity)
    return out


# -----------------------------------------------------------------------------
# Abstract propagator base class
# -----------------------------------------------------------------------------


class BaseUncertaintyPropagator(ABC):
    """Strategy interface for uncertainty propagation.

    Concrete subclasses are expected to implement
    :meth:`~BaseUncertaintyPropagator.propagate`.
    """

    @abstractmethod
    def propagate(self, feature_matrix: np.ndarray, prior_u: np.ndarray) -> np.ndarray:  # noqa: D401,E501
        """Compute posterior uncertainties.

        The method must be *pure*: given the same inputs (including the internal
        state of ``self``) it must always produce the identical output.

        Parameters
        ----------
        feature_matrix
            ``(n, d)`` array of ``np.float32`` feature vectors for **all**
            points.
        prior_u
            ``(n,)`` array of prior uncertainties.

        Returns
        -------
        np.ndarray
            Posterior uncertainties of shape ``(n,)``.
        """


# -----------------------------------------------------------------------------
# Concrete distance‑based propagator
# -----------------------------------------------------------------------------


class DistanceBasedPropagator(BaseUncertaintyPropagator):
    """Gaussian distance‑kernel implementation.

    Parameters
    ----------
    labeled_features
        ``(m, d)`` matrix of labelled points (float32).
    lambda_param
        Kernel scale :math:`\lambda`.  Values too small make the kernel almost
        flat; values too large will squash uncertainties aggressively.
    threshold
        Maximum radius to search for a label.  Distances beyond this value are
        treated as infinite (i.e. contribute no evidence).
    """

    def __init__(
            self,
            labeled_features: np.ndarray,
            *,
            lambda_param: float = 1.0,
            threshold: float = np.inf,
    ) -> None:
        self.labeled_features = labeled_features.astype(np.float32, copy=False)
        self.lambda_param = float(lambda_param)
        self.threshold = float(threshold)

    # ------------------------------------------------------------------
    # BaseUncertaintyPropagator interface
    # ------------------------------------------------------------------

    def propagate(self, feature_matrix: np.ndarray, prior_u: np.ndarray) -> np.ndarray:  # noqa: D401,E501
        """See base class for full description."""
        if feature_matrix.dtype != np.float32:
            feature_matrix = feature_matrix.astype(np.float32)
        return _numba_distance_kernel(
            feature_matrix,
            self.labeled_features,
            prior_u.astype(np.float32),
            self.lambda_param,
            self.threshold,
        )


# -----------------------------------------------------------------------------
# High‑level convenience wrapper
# -----------------------------------------------------------------------------


def propagate_for_annotations(
        annos: List[AnnotationBase],
        *,
        lambda_param: Union[float, str] = "auto",
        threshold: float = np.inf,
        k_auto: Optional[int] = None,
) -> None:
    """Propagate uncertainties *in‑place* on a list of annotations.

    Each :class:`~GUI.models.Annotation.Annotation` must expose the following
    attributes:

    * ``logit_features`` – ``(d,)`` array of *float32* that embeds the point
      into a metric space where Euclidean distances make semantic sense.
    * ``uncertainty`` – scalar prior uncertainty.
    * ``class_id`` – integer label.  Values *-1* and *-2* are interpreted as
      *unlabelled* by convention.

    After calling the function, the *posterior* uncertainty is written to
    ``adjusted_uncertainty`` on every annotation.  Labelled annotations are
    forced to zero, as they are considered fully certain.

    Parameters
    ----------
    annos
        List of annotation objects to update.  If empty, the function is a
        no‑op.
    lambda_param
        Either a numeric value to use directly, or the string ``"auto"`` to
        invoke the :func:`auto_lambda` heuristic.
    threshold
        Distance above which labels are ignored.  Useful to avoid accidental
        influence in extremely sparse regions.
    k_auto
        Custom *k* for the automatic λ heuristic.  Ignored unless
        ``lambda_param == 'auto'``.
    """

    if not annos:
        return

    feats = np.stack([a.logit_features for a in annos]).astype(np.float32)
    prior_u = np.array([a.uncertainty for a in annos], dtype=np.float32)
    mask_lbl = np.array([a.class_id not in (-1, -2) for a in annos])
    lbl_feats = feats[mask_lbl]

    # ---------- automatic λ -----------------------------------------------
    if isinstance(lambda_param, str) and lambda_param.lower() == "auto":
        lambda_param = auto_lambda(feats, k=k_auto)
    # ----------------------------------------------------------------------

    post_u = DistanceBasedPropagator(
        lbl_feats,
        lambda_param=lambda_param,
        threshold=threshold,
    ).propagate(feats, prior_u)

    post_u[mask_lbl] = 0.0  # labels are fully certain
    for a, u in zip(annos, post_u):
        a.adjusted_uncertainty = float(u) if np.isscalar(u) else u
