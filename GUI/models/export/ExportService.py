# -----------------------------
# export/service.py
# -----------------------------
"""Pure functions that implement the data‑manipulation logic."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from GUI.models.annotations import AnnotationBase, PointAnnotation, MaskAnnotation
import numpy as np
from .Options import ExportOptions

__all__ = ["build_grouped_annotations"]

Grouped = Dict[str, List[dict]]


def _rle_encode(mask: np.ndarray) -> List[int]:
    """Return run-length encoding for a binary mask."""
    flat = mask.astype(np.uint8).ravel()
    counts: List[int] = []
    prev = flat[0]
    length = 1
    for val in flat[1:]:
        if val == prev:
            length += 1
        else:
            counts.append(length)
            length = 1
            prev = val
    counts.append(length)
    return counts


def _should_include(anno: AnnotationBase, opts: ExportOptions) -> bool:
    """Return *True* when *anno* must be part of the export."""
    if anno.class_id in {None, -1, -2}:  # unlabeled or unsure
        return False
    if anno.class_id == -3 and not opts.include_artifacts:
        return False
    return True


def build_grouped_annotations(
        cluster_annos: Iterable[Tuple[int, AnnotationBase]],
        opts: ExportOptions,
) -> Grouped:
    """Group annotations by *filename* applying the filtering rules in *opts*.

    Parameters
    ----------
    cluster_annos
        ``[(cluster_id, annotation), ...]`` pairs.
    opts
        Export configuration flags.

    Returns
    -------
    dict[str, list[dict]]
        ``{"filename": [{"coord": [...], "class_id": ..., "cluster_id": ...}, ...]}``
    """
    grouped: Grouped = defaultdict(list)

    for cluster_id, anno in cluster_annos:
        if not _should_include(anno, opts):
            continue

        entry = {
            "class_id": int(anno.class_id),
            "cluster_id": int(cluster_id),
        }
        if isinstance(anno, MaskAnnotation) and anno.mask is not None:
            entry["mask_rle"] = _rle_encode(anno.mask)
            entry["mask_shape"] = list(anno.mask.shape)
            entry["coord"] = [int(c) for c in anno.coord]
        else:
            entry["coord"] = [int(c) for c in anno.coord]
        grouped[anno.filename].append(entry)

    return dict(grouped)
