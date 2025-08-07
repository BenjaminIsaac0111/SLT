"""Utility helpers for training scripts.

This module groups small reusable functions that were previously embedded in
the fine‑tuning script. Keeping them here improves modularity and allows other
training utilities to share the implementations.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import tensorflow as tf


def set_memory_growth() -> None:
    """Enable dynamic memory allocation on all visible GPUs."""

    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:  # pragma: no cover - best effort logging
            tf.print(f"[WARN] Could not set memory growth on {gpu}: {exc}")


def compute_class_weights_from_json(json_path: Path, num_classes: int) -> list[float]:
    """Compute inverse-frequency class weights.

    The JSON file is expected to map sample identifiers to a list of mark
    dictionaries that contain a ``class_id`` field.

    Args:
        json_path: Path to the annotation file.
        num_classes: Total number of classes.

    Returns:
        A list of weights for each class. Missing classes receive a weight of
        ``0.0``.
    """

    with json_path.open() as fp:
        annotations = json.load(fp)

    counts = Counter(
        m["class_id"]
        for marks in annotations.values()
        for m in marks
        if 0 <= m["class_id"] < num_classes
    )

    total = sum(counts.values()) or 1  # avoid division by zero
    return [
        (total / (num_classes * counts[c])) if counts.get(c) else 0.0
        for c in range(num_classes)
    ]


def xla_optional(jit: bool = True) -> Callable:
    """Decorator that enables XLA if possible.

    Args:
        jit: Whether to attempt JIT compilation. If ``False`` the wrapped
            function will be executed without XLA.

    Returns:
        A decorator that wraps ``fn`` with :func:`tf.function` and optionally
        enables XLA. When compilation fails the function falls back to a
        non-JIT compiled graph while logging a warning.
    """

    def decorator(fn: Callable) -> Callable:
        if not jit:
            tf.print(f"[INFO] JIT disabled for `{fn.__name__}`; using plain graph.")
            return tf.function(fn, jit_compile=False)

        try:
            wrapped = tf.function(fn, jit_compile=True)
            tf.print(f"[INFO] XLA enabled for `{fn.__name__}`")
            return wrapped
        except (tf.errors.InvalidArgumentError, ValueError) as exc:
            tf.print(
                f"[WARN] XLA failed for `{fn.__name__}` – falling back to eager graph: {exc}"
            )
            return tf.function(fn, jit_compile=False)

    return decorator


def count_samples_from_json(json_path: Path) -> int:
    """Count the number of annotated samples in a JSON file."""

    with json_path.open("r") as file:
        annotations = json.load(file)
    return len(annotations)


__all__ = [
    "set_memory_growth",
    "compute_class_weights_from_json",
    "xla_optional",
    "count_samples_from_json",
]

