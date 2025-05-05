"""
ClusteringRanker.py
------------------
Stateless utilities that aggregate per-item uncertainty scores
(e.g. BALD values) into a single scalar cluster score.

The heavy inner loops are JIT-accelerated with Numba if available;
otherwise they run as ordinary NumPy.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Dict

import numpy as np

# ---------------------------------------------------------------------
#  Optional Numba JIT — painless fallback
# ---------------------------------------------------------------------
try:  # pragma: no cover
    from numba import njit


    def _maybe_jit(fn: Callable) -> Callable:
        return njit(cache=True, fastmath=True)(fn)


    NUMBA_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    def _maybe_jit(fn: Callable) -> Callable:
        return fn


    NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------
#  Aggregator implementations
# ---------------------------------------------------------------------

@_maybe_jit
def _mean(u: np.ndarray) -> float:
    return u.mean()


@_maybe_jit
def _amax(u: np.ndarray) -> float:
    return u.max()


@_maybe_jit
def _power_mean(u: np.ndarray, p: float) -> float:
    return (u ** p).mean() ** (1.0 / p)


def _topk_mean(u: np.ndarray, k: int) -> float:
    """
    O(n) selection of k largest elements via argpartition
    (Numba can JIT argpartition since NumPy 1.23+).
    """
    k_eff = max(1, min(k, u.size))
    indices = np.argpartition(u, -k_eff)[-k_eff:]
    return u[indices].mean()


def _softmax_mean(u: np.ndarray, beta: float) -> float:
    """
    Numerically stable softmax with shift.
    """
    shifted = beta * (u - u.max())  # subtract max to avoid overflow
    w = np.exp(shifted)
    return float(np.dot(w, u) / w.sum())


# ---------------------------------------------------------------------
#  Registry and dispatcher
# ---------------------------------------------------------------------

def _make_power(p: float) -> Callable[[np.ndarray], float]:
    return partial(_power_mean, p=p)


def _make_topk(k: int) -> Callable[[np.ndarray], float]:
    return partial(_topk_mean, k=k)


def _make_softmax(beta: float) -> Callable[[np.ndarray], float]:
    return partial(_softmax_mean, beta=beta)


def _build_registry(
        *,
        k: int = 3,
        p: float = 4.0,
        beta: float = 2.0,
) -> Dict[str, Callable[[np.ndarray], float]]:
    """
    Return a fresh dict so that different callers can
    pass different hyperparameters without clobbering globals.
    """
    if p <= 0:
        raise ValueError("p must be > 0 for power mean")

    return {
        "mean": _mean,
        "max": _amax,
        "topk": _make_topk(k),
        "power": _make_power(p),
        "softmax": _make_softmax(beta),
    }


# ---------------------------------------------------------------------
#  Public façade
# ---------------------------------------------------------------------

def score_cluster(
        u: np.ndarray | list[float],
        *,
        method: str = "topk",
        k: int = 3,
        p: float = 4.0,
        beta: float = 2.0,
) -> float:
    """
    Parameters
    ----------
    u : 1-D array-like
        Per-item uncertainty scores (float32 or float64 preferred).
    method : {"mean","max","topk","power","softmax"}
        Aggregation heuristic.
    k, p, beta : float or int
        Hyper-parameters used by the corresponding heuristic.

    Returns
    -------
    float
        Cluster score.

    Notes
    -----
    • The caller is responsible for passing a NumPy array if they need
      JIT speed; Python lists incur a one-time conversion cost.
    • For `topk` you can set k = ceil(tau*|C|) to obtain a CVaR-τ tail.
    • For `softmax` beta≈2 applies ≈90 % weight to the top 10 % scores
      when the input is roughly z-normalised.
    """
    u_arr = np.asarray(u, dtype=np.float32)

    registry = _build_registry(k=k, p=p, beta=beta)
    try:
        fn = registry[method]
    except KeyError as e:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Available = {sorted(registry)}") from e
    return fn(u_arr)


# ---------------------------------------------------------------------
#  Helper for library introspection
# ---------------------------------------------------------------------
def describe() -> str:
    avail = ", ".join(sorted(_build_registry()))
    jit = "ON" if NUMBA_AVAILABLE else "OFF"
    return f"cluster_scoring methods: {avail}  |  numba-jit: {jit}"
