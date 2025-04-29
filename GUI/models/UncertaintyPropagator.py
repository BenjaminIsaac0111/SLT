"""
This implementation allows uncertainties to both increase and decrease when
labels are added or removed.  Every annotation keeps an immutable *prior*
(`Annotation.uncertainty`).  The propagator always starts from that prior and
computes a new posterior (`Annotation.adjusted_uncertainty`) given the current
set of labelled points.

Key properties
--------------
* Posterior is always in the interval [0, prior].
* Removing the last bit of evidence (un‑labelling) restores the field to the
  original prior.
* Idempotent: repeated propagation without changing labels is a no‑op.
"""

from abc import ABC, abstractmethod
from typing import Optional
from typing import Union, List

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from GUI.models.Annotation import Annotation


def auto_lambda(
        feats: np.ndarray,
        k: Optional[int] = None
) -> float:
    """
    Data-driven λ based on the median k-NN distance.

    Parameters
    ----------
    feats : (n, d) float32
        Feature matrix (all points, labelled or not).
    k : int, optional
        Order of the neighbour to use.  Default log2(n).

    Returns
    -------
    lambda_param : float
        Scale parameter for the Gaussian kernel.
    """
    # choose k adaptively once
    n = feats.shape[0]
    if k is None:
        k = max(1, int(np.ceil(np.log2(n))))

    # fast exact k-NN search
    tree = cKDTree(feats)
    # query returns the distance to each of the first k+1 neighbours
    dists, _ = tree.query(feats, k=k + 1)  # include self at column 0
    sigma = np.median(dists[:, -1])  # k-th neighbour

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
) -> np.ndarray:
    """Gaussian‑kernel posterior uncertainty.

    For every point *x* we compute the squared distance to its nearest labelled
    neighbour, capped by ``threshold``.  The similarity is

        s = exp(‑lambda_param * d²)

    and the posterior uncertainty is

        u_post = u_prior * (1 − s)

    which lies in ``[0, u_prior]``.
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
    """Strategy interface for uncertainty propagation."""

    @abstractmethod
    def propagate(self, feature_matrix: np.ndarray, prior_u: np.ndarray) -> np.ndarray:
        """Return the posterior uncertainties given the current labels."""


# -----------------------------------------------------------------------------
# Concrete distance‑based propagator
# -----------------------------------------------------------------------------

class DistanceBasedPropagator(BaseUncertaintyPropagator):
    """Gaussian‑distance kernel propagator (vectorised via Numba)."""

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

    def propagate(self, feature_matrix: np.ndarray, prior_u: np.ndarray) -> np.ndarray:
        if feature_matrix.dtype != np.float32:
            feature_matrix = feature_matrix.astype(np.float32)
        return _numba_distance_kernel(
            feature_matrix,
            self.labeled_features,
            prior_u.astype(np.float32),
            self.lambda_param,
            self.threshold,
        )


def propagate_for_annotations(
        annos: List[Annotation],
        *,
        lambda_param: Union[float, str] = "auto",
        threshold: float = np.inf,
        k_auto: Optional[int] = None,
) -> None:
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

    post_u[mask_lbl] = 0.0
    for a, u in zip(annos, post_u):
        a.adjusted_uncertainty = float(u) if np.isscalar(u) else u
