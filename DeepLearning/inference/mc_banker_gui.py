"""Wrapper for running MC banker inference from external callers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import yaml


REQUIRED_KEYS = [
    "MODEL_DIR",
    "MODEL_NAME",
    "DATA_DIR",
    "FILE_LIST",
    "MC_N_ITER",
]


def _infer_model_spec(path: str) -> Tuple[list[int], int]:
    """Return ``(input_size, out_channels)`` for a saved model path."""
    from tensorflow.keras.models import load_model
    from tensorflow.keras import mixed_precision
    from DeepLearning.models.custom_layers import (
        DropoutAttentionBlock,
        GroupNormalization,
        SpatialConcreteDropout,
    )

    mixed_precision.set_global_policy("mixed_float16")
    model = load_model(
        path,
        custom_objects={
            "DropoutAttentionBlock": DropoutAttentionBlock,
            "GroupNormalization": GroupNormalization,
            "SpatialConcreteDropout": SpatialConcreteDropout,
        },
        compile=False,
    )
    input_size = list(model.input_shape[1:])
    out_channels = int(model.layers[-1].output_shape[-1])
    return input_size, out_channels


def run_from_file(
    config_path: str,
    *,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    resume: bool = False,
    log_level: str = "INFO",
) -> None:
    """Run MC banker inference using a YAML configuration file."""
    from DeepLearning.inference.main_unet_mc_banker import main, setup_logging

    logger = setup_logging(getattr(logging, log_level.upper(), logging.INFO))
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    if output_dir:
        cfg["OUTPUT_DIR"] = output_dir
    if output_file:
        cfg["OUTPUT_FILE"] = output_file

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    cfg.setdefault("BATCH_SIZE", 1)
    cfg.setdefault("SHUFFLE_BUFFER_SIZE", 256)

    if "INPUT_SIZE" not in cfg or "OUT_CHANNELS" not in cfg:
        model_path = Path(cfg["MODEL_DIR"]) / cfg["MODEL_NAME"]
        cfg["INPUT_SIZE"], cfg["OUT_CHANNELS"] = _infer_model_spec(str(model_path))

    main(cfg, logger=logger, resume=resume)


def cli() -> None:
    """Simple command line interface."""
    parser = argparse.ArgumentParser(description="Run MC banker inference")
    parser.add_argument("-c", "--config_path", required=True, help="YAML configuration file")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--output_dir", help="Override OUTPUT_DIR from YAML")
    parser.add_argument("--output_file", help="Override OUTPUT_FILE from YAML")
    parser.add_argument("--resume", action="store_true", help="Append to existing HDF5 instead of overwrite")
    args = parser.parse_args()

    run_from_file(
        args.config_path,
        output_dir=args.output_dir,
        output_file=args.output_file,
        resume=args.resume,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    cli()
