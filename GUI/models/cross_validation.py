from __future__ import annotations

"""Utilities for building stratified group cross-validation folds."""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

__all__ = ["build_cv_folds", "write_cv_folds"]


def _parse_filename(name: str) -> Tuple[str, str]:
    """Return group and class extracted from *name*."""
    stem = Path(name).stem
    parts = stem.split("_")
    return parts[0], parts[-1]


def _compute_weights(y: Iterable[str], classes: Iterable[str]) -> Dict[str, float]:
    """Return class weights for ``y`` labelled by ``classes``.

    Missing classes receive a weight of ``0.0`` so the mapping size remains
    constant regardless of class presence in the fold.
    """
    counts = pd.Series(list(y)).value_counts()
    total = counts.sum()
    n_classes = len(list(classes))
    weights = {}
    for cls in classes:
        if cls in counts and counts[cls] > 0:
            weights[cls] = total / (n_classes * counts[cls])
        else:
            weights[cls] = 0.0
    return weights


def build_cv_folds(
    image_dir: str | Path, *, n_splits: int = 3, shuffle: bool = True
) -> List[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]]:
    """Return cross-validation folds for *image_dir*.

    Each returned tuple is ``(train_df, test_df, weights)`` where ``train_df`` and
    ``test_df`` contain a ``filename`` column. ``weights`` is a mapping from class
    label to computed weight for the training set.
    """
    path = Path(image_dir).expanduser()
    files = [f for f in path.iterdir() if f.is_file()]

    records = []
    for f in files:
        group, cls = _parse_filename(f.name)
        records.append({"filename": f.name, "group": group, "class": cls})

    df = pd.DataFrame(records)
    rng = 42 if shuffle else None
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=rng)

    all_classes = sorted(df["class"].unique())

    folds = []
    for train_idx, test_idx in cv.split(df["filename"], df["class"], df["group"]):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        weights = _compute_weights(train_df["class"], all_classes)
        folds.append((train_df, test_df, weights))
    return folds


def write_cv_folds(
    image_dir: str | Path,
    output_dir: str | Path,
    *,
    n_splits: int = 3,
    shuffle: bool = True,
    progress: callable[[int], None] | None = None,
) -> None:
    """Write cross-validation folds to ``output_dir``.

    Parameters
    ----------
    image_dir:
        Directory containing images to split into folds.
    output_dir:
        Destination directory for generated files.
    progress:
        Optional callback receiving the 1-based index of the fold that has just
        been written.
    """
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("Fold_*_*.txt"):
        old.unlink()

    for i, (train_df, test_df, weights) in enumerate(
        build_cv_folds(image_dir, n_splits=n_splits, shuffle=shuffle), start=1
    ):
        train_df[["filename"]].to_csv(
            out / f"Fold_{i}_TrainingData.txt", index=False, header=False, sep="\t"
        )
        test_df[["filename"]].to_csv(
            out / f"Fold_{i}_TestData.txt", index=False, header=False, sep="\t"
        )
        pd.DataFrame({"class": list(weights.keys()), "weight": list(weights.values())}).to_csv(
            out / f"Fold_{i}_weights.txt", index=False, sep="\t"
        )
        if progress:
            progress(i)


