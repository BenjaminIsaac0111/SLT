from __future__ import annotations

"""Utilities for building stratified group cross-validation folds."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

__all__ = ["build_cv_folds", "write_cv_folds"]


def _parse_filename(name: str) -> Tuple[str, str]:
    """Return group and class extracted from *name*."""
    stem = Path(name).stem
    parts = stem.split("_")
    return parts[0], parts[-1]


def build_cv_folds(
    image_dir: str | Path, *, n_splits: int = 3, shuffle: bool = True
) -> List[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]]:
    """Return cross-validation folds for *image_dir*.

    Each returned tuple is ``(train_df, test_df, weights)`` where ``train_df`` and
    ``test_df`` contain a ``filename`` column.  ``weights`` are class weights for
    the training set computed with ``compute_class_weight``.
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

    folds = []
    for train_idx, test_idx in cv.split(df["filename"], df["class"], df["group"]):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        weights = compute_class_weight(
            "balanced", classes=np.unique(train_df["class"]), y=train_df["class"]
        )
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

    for i, (train_df, test_df, weights) in enumerate(
        build_cv_folds(image_dir, n_splits=n_splits, shuffle=shuffle), start=1
    ):
        train_df[["filename"]].to_csv(
            out / f"Fold_{i}_TrainingData.txt", index=False, header=False, sep="\t"
        )
        test_df[["filename"]].to_csv(
            out / f"Fold_{i}_TestData.txt", index=False, header=False, sep="\t"
        )
        np.savetxt(out / f"Fold_{i}_weights.txt", weights, fmt="%.6f")
        if progress:
            progress(i)


