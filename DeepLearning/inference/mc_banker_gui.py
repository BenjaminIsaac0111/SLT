"""Wrapper for running MC banker inference from external callers."""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import yaml


REQUIRED_KEYS = [
    "MODEL_DIR",
    "MODEL_NAME",
    "DATA_DIR",
    "TRAINING_LIST",
    "BATCH_SIZE",
    "SHUFFLE_BUFFER_SIZE",
    "OUT_CHANNELS",
    "INPUT_SIZE",
    "MC_N_ITER",
]


def run_from_file(
    config_path: str,
    *,
    output_dir: Optional[str] = None,
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

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    main(cfg, logger=logger, resume=resume)


def cli() -> None:
    """Simple command line interface."""
    parser = argparse.ArgumentParser(description="Run MC banker inference")
    parser.add_argument("-c", "--config_path", required=True, help="YAML configuration file")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--output_dir", help="Override OUTPUT_DIR from YAML")
    parser.add_argument("--resume", action="store_true", help="Append to existing HDF5 instead of overwrite")
    args = parser.parse_args()

    run_from_file(args.config_path, output_dir=args.output_dir, resume=args.resume, log_level=args.log_level)


if __name__ == "__main__":
    cli()
