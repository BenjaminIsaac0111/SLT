from __future__ import annotations

"""Deprecated compatibility layer for annotation classes."""

from .annotations import (
    AnnotationBase,
    PointAnnotation,
    MaskAnnotation,
    annotation_from_dict,
)

# Backwards compatibility: old code imported `Annotation` directly.
Annotation = PointAnnotation

__all__ = [
    "AnnotationBase",
    "PointAnnotation",
    "MaskAnnotation",
    "annotation_from_dict",
    "Annotation",
]
