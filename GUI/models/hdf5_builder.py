"""Utilities for creating small HDF5 datasets from training CSV files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd

__all__ = ["build_training_hdf5"]


def build_training_hdf5(
    data_dir: Union[str, Path],
    training_csv: Union[str, Path],
    model_path: Union[str, Path],
    output_file: Union[str, Path],
    *,
    sample_size: Optional[int] = None,
) -> None:
    """Create a lightweight HDF5 file listing training samples.

    Parameters
    ----------
    data_dir:
        Directory containing the image files referenced in ``training_csv``.
    training_csv:
        CSV listing image filenames and class labels.
    model_path:
        Path to the segmentation model (currently unused but logged).
    output_file:
        Destination HDF5 file to create.
    sample_size:
        Optional number of samples to keep.  If provided and smaller than the
        CSV length, a random subset without replacement is written.
    """
    data_dir = Path(data_dir)
    training_csv = Path(training_csv)
    output_file = Path(output_file)

    df = pd.read_csv(training_csv, header=None, names=["filename", "class"])
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    filenames = [str(data_dir / f) for f in df["filename"]]
    classes = df["class"].astype(str).to_numpy()

    logging.info("Creating HDF5 at %s using model %s", output_file, model_path)

    with h5py.File(output_file, "w") as h5f:
        h5f.create_dataset(
            "filenames", data=np.array(filenames, dtype=h5py.string_dtype())
        )
        h5f.create_dataset(
            "class", data=np.array(classes, dtype=h5py.string_dtype())
        )
        # Placeholder dataset for model features (zeroes for now)
        h5f.create_dataset("features", shape=(len(df), 1), dtype="float32")
    logging.info("Wrote %d entries to %s", len(df), output_file)
