"""Utilities for building training HDF5 datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union



__all__ = ["build_training_hdf5"]


def build_training_hdf5(
    data_dir: Union[str, Path],
    training_csv: Union[str, Path],
    model_path: Union[str, Path],
    output_file: Union[str, Path],
    *,
    sample_size: Optional[int] = None,
    mc_iter: int = 1,
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
    mc_iter:
        Number of stochastic forward passes for MC dropout.
    resume:
        Append to an existing file instead of overwriting.
    """

    from tensorflow.keras.models import load_model
    from DeepLearning.inference import main_unet_mc_banker
    from DeepLearning.models.custom_layers import (
        DropoutAttentionBlock,
        GroupNormalization,
        SpatialConcreteDropout,
    )

    main_unet_mc_banker.set_global_seed(42)

    model = load_model(
        model_path,
        custom_objects={
            "DropoutAttentionBlock": DropoutAttentionBlock,
            "GroupNormalization": GroupNormalization,
            "SpatialConcreteDropout": SpatialConcreteDropout,
        },
        compile=False,
    )
    input_shape = list(model.input_shape[1:])
    out_channels = int(model.output_shape[-1])

    cfg = {
        "MODEL_DIR": str(Path(model_path).parent),
        "MODEL_NAME": Path(model_path).name,
        "DATA_DIR": str(data_dir),
        "TRAINING_LIST": str(training_csv),
        "BATCH_SIZE": 1,
        "SHUFFLE_BUFFER_SIZE": 1,
        "OUT_CHANNELS": out_channels,
        "INPUT_SIZE": input_shape,
        "MC_N_ITER": mc_iter,
        "OUTPUT_DIR": str(Path(output_file).parent),
    }

    if sample_size is not None:
        cfg["N_SAMPLES"] = int(sample_size)

    logger = main_unet_mc_banker.setup_logging(logging.INFO)
    main_unet_mc_banker.main(cfg, logger=logger, resume=resume)

    logging.info("HDF5 written to %s", output_file)
