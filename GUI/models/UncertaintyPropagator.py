from abc import ABC, abstractmethod

import numpy as np
from numba import njit


@njit
def _numba_distance_based_propagation(unlabeled_features: np.ndarray,
                                      labeled_features: np.ndarray,
                                      uncertainties: np.ndarray,
                                      lambda_param: float,
                                      threshold: float) -> np.ndarray:
    """
    Propagates uncertainty via a Gaussian kernel similarity measure computed from distances.

    For each unlabeled feature vector, the minimum Euclidean distance (capped by threshold)
    to any labeled feature is computed. This distance is converted into a similarity via:
        similarity = exp(-lambda_param * d^2)
    The original uncertainty is then updated:
        u_new = u * (1 - similarity)
    The result is clipped to ensure it stays in [0, u].

    Parameters
    ----------
    unlabeled_features : np.ndarray
        Array of shape (n_unlabeled, n_features) for unlabeled data.
    labeled_features : np.ndarray
        Array of shape (n_labeled, n_features) for labeled data.
    uncertainties : np.ndarray
        1D array of original uncertainty values for each unlabeled data point.
    lambda_param : float
        Parameter controlling the decay in similarity as a function of distance.
    threshold : float
        Maximum distance to consider; distances above this value are capped.

    Returns
    -------
    np.ndarray
        1D array of updated uncertainty values.
    """
    n_unlabeled = unlabeled_features.shape[0]
    n_labeled = labeled_features.shape[0]
    updated_uncertainties = np.empty(n_unlabeled)

    for i in range(n_unlabeled):
        min_distance = threshold
        for j in range(n_labeled):
            sum_sq = 0.0
            for k in range(unlabeled_features.shape[1]):
                diff = unlabeled_features[i, k] - labeled_features[j, k]
                sum_sq += diff * diff
            distance = sum_sq ** 0.5
            if distance < min_distance:
                min_distance = distance

        similarity = np.exp(-lambda_param * (min_distance ** 2))
        new_uncertainty = uncertainties[i] * (1.0 - similarity)
        if new_uncertainty < 0:
            new_uncertainty = 0.0
        elif new_uncertainty > uncertainties[i]:
            new_uncertainty = uncertainties[i]

        updated_uncertainties[i] = new_uncertainty

    return updated_uncertainties


class BaseUncertaintyPropagator(ABC):
    """
    Abstract Base Class for Uncertainty Propagation.

    Subclasses must implement the `propagate` method for updating uncertainties.
    """

    @abstractmethod
    def propagate(self,
                  feature_matrix: np.ndarray,
                  uncertainties: np.ndarray) -> np.ndarray:
        """
        Propagate uncertainties based on the chosen strategy.

        Parameters
        ----------
        feature_matrix : np.ndarray
            Matrix where each row is a feature vector corresponding to an annotation.
        uncertainties : np.ndarray
            Original uncertainty values.

        Returns
        -------
        np.ndarray
            Updated uncertainty values (clipped to valid bounds).
        """
        pass


class DistanceBasedPropagator(BaseUncertaintyPropagator):
    """
    Uncertainty Propagator based on Distance between Features.

    For each unlabeled annotation, the minimum Euclidean distance to any labeled
    annotation is computed. A similarity measure is then derived using a Gaussian kernel,
    and the uncertainty is updated as:
        u_new = u * (1 - similarity)
    """

    def __init__(self,
                 labeled_features: np.ndarray,
                 lambda_param: float = 1.0,
                 threshold: float = np.inf) -> None:
        """
        Initialize the distance-based propagator with the necessary components.

        Parameters
        ----------
        labeled_features : np.ndarray
            Matrix of features corresponding to labeled annotations.
        lambda_param : float, default=1.0
            Parameter for decay in the Gaussian kernel.
        threshold : float, default=np.inf
            Upper limit on the distance used in the kernel.
        """
        self.labeled_features = labeled_features
        self.lambda_param = lambda_param
        self.threshold = threshold

    def propagate(self,
                  feature_matrix: np.ndarray,
                  uncertainties: np.ndarray) -> np.ndarray:
        """
        Propagate uncertainty using a distance-based strategy.

        Parameters
        ----------
        feature_matrix : np.ndarray
            Feature matrix for unlabeled annotations.
        uncertainties : np.ndarray
            Original uncertainty values.

        Returns
        -------
        np.ndarray
            Updated uncertainty values.
        """
        updated_uncertainties = _numba_distance_based_propagation(
            feature_matrix, self.labeled_features, uncertainties,
            self.lambda_param, self.threshold
        )
        return updated_uncertainties
