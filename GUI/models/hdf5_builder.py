"""Utilities for building training HDF5 datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import yaml

__all__ = ["build_training_hdf5"]


def build_training_hdf5(
    data_dir: Union[str, Path],
    training_csv: Union[str, Path],
    model_path: Union[str, Path],
    output_file: Union[str, Path],
    *,
    sample_size: Optional[int] = None,
    resume: bool = False,
) -> None:
    """Run the MC banker inference pipeline to build a training HDF5.

    This is a thin wrapper around :func:`main_unet_mc_banker.main` that builds
    the required configuration from the provided arguments.

    Parameters
    ----------
    data_dir:
        Directory containing the training images.
    training_csv:
        CSV file listing the training images and labels.
    model_path:
        Path to the segmentation model weights.
    output_file:
        Destination HDF5 file.
    sample_size:
        Optional subsample size used to limit ``N_SAMPLES`` in the config.
    resume:
        Append to an existing file instead of overwriting.
    """

    from DeepLearning.inference import main_unet_mc_banker

    cfg = {
        "MODEL_DIR": str(Path(model_path).parent),
        "MODEL_NAME": Path(model_path).name,
        "DATA_DIR": str(data_dir),
        "TRAINING_LIST": str(training_csv),
        "BATCH_SIZE": 1,
        "SHUFFLE_BUFFER_SIZE": 1,
        "OUT_CHANNELS": 2,
        "INPUT_SIZE": [256, 256, 3],
        "MC_N_ITER": 1,
        "OUTPUT_DIR": str(Path(output_file).parent),
    }

    if sample_size is not None:
        cfg["N_SAMPLES"] = int(sample_size)

    logger = main_unet_mc_banker.setup_logging(logging.INFO)
    main_unet_mc_banker.main(cfg, logger=logger, resume=resume)

    logging.info("HDF5 written to %s", output_file)
