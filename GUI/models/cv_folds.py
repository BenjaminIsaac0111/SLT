"""Utilities for creating grouped cross-validation folds."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import compute_class_weight

__all__ = ["create_grouped_folds"]


def _parse_filename(name: str) -> tuple[str, str]:
    """Return ``(group_id, class_label)`` parsed from *name*.

    The filename must contain an underscore-separated group identifier and class
    label, e.g. ``"123_img_tumour.png"``.
    """
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid filename format: {name}")
    group_id = parts[0]
    class_label = parts[-1].split(".")[0]
    return group_id, class_label


def _shuffle_train_fold(train_df: pd.DataFrame) -> pd.DataFrame:
    """Return *train_df* shuffled without changing its size."""
    return train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def create_grouped_folds(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    n_splits: int = 3,
    sample_size: Optional[int] = None,
) -> None:
    """Generate grouped stratified folds from image files in *data_dir*.

    Parameters
    ----------
    data_dir:
        Directory containing input image files.
    output_dir:
        Destination for the ``Fold_*`` text files.
    n_splits:
        Number of folds to generate.
    sample_size:
        Optional size to subsample each fold without replacement.
        Only files with common image extensions (PNG, JPG, TIFF, BMP) are
        considered; any other files are ignored.  The generated ``TrainingData``
        and ``TestData`` files contain one filename per line with no class
        labels.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only consider common image types to avoid parsing any output files
    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = [
        f
        for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_exts
    ]
    if not files:
        raise ValueError(f"No files found in {data_dir}")

    records = []
    for f in files:
        gid, cls = _parse_filename(f.name)
        records.append({"filename": f.name, "id": gid, "class": cls})
    dataset = pd.DataFrame(records)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    overall_weights = compute_class_weight(
        "balanced", classes=np.unique(dataset["class"]), y=dataset["class"]
    )
    with (output_dir / "weights.txt").open("w") as w:
        w.write(" ".join(map(str, overall_weights)))

    for i, (train_idx, test_idx) in enumerate(
        cv.split(dataset["filename"], dataset["class"], dataset["id"])
    ):
        train_df = _shuffle_train_fold(dataset.iloc[train_idx].copy())
        if sample_size is not None and len(train_df) > sample_size:
            train_df = train_df.sample(n=sample_size, replace=False, random_state=42)

        test_df = dataset.iloc[test_idx][["filename", "class"]]
        if sample_size is not None and len(test_df) > sample_size:
            test_df = test_df.sample(n=sample_size, replace=False, random_state=42)

        weights = compute_class_weight(
            "balanced", classes=np.unique(train_df["class"]), y=train_df["class"]
        )

        (output_dir / f"Fold_{i + 1}_TrainingData.txt").write_text(
            "\n".join(train_df["filename"]) + "\n"
        )
        (output_dir / f"Fold_{i + 1}_TestData.txt").write_text(
            "\n".join(test_df["filename"]) + "\n"
        )
        with (output_dir / f"Fold_{i + 1}_weights.txt").open("w") as wf:
            wf.write(" ".join(map(str, weights)))

    logging.info("Created %d folds in %s", n_splits, output_dir)
