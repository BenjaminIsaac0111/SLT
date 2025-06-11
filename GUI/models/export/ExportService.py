# -----------------------------
# export/service.py
# -----------------------------
"""Pure functions that implement the data‑manipulation logic."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from GUI.models.Annotation import Annotation
from .Options import ExportOptions

__all__ = ["build_grouped_annotations"]

Grouped = Dict[str, List[dict]]


def _should_include(anno: Annotation, opts: ExportOptions) -> bool:
    """Return *True* when *anno* must be part of the export."""
    if anno.class_id in {None, -1, -2}:  # unlabeled or unsure
        return False
    if anno.class_id == -3 and not opts.include_artifacts:
        return False
    return True


def build_grouped_annotations(
        cluster_annos: Iterable[Tuple[int, Annotation]],
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

        grouped[anno.filename].append(
            {
                "coord": [int(c) for c in anno.coord],
                "class_id": int(anno.class_id),
                "cluster_id": int(cluster_id),
            }
        )

    return dict(grouped)
