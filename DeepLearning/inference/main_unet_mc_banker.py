import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import tensorflow as tf
import yaml
from keras import Model
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model
from tqdm import tqdm

from DeepLearning.dataloader.dataloader import get_dataset_v2
from DeepLearning.models.custom_layers import (
    DropoutAttentionBlock,
    GroupNormalization,
    SpatialConcreteDropout,
)

# ----------------------------------------------------------------------------
# Global configuration --------------------------------------------------------
# ----------------------------------------------------------------------------

# 0 = INFO, 1 = WARNING, 2 = ERROR (suppress INFO + WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Run deterministically whenever possible
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")


# ----------------------------------------------------------------------------
# Logging ---------------------------------------------------------------------
# ----------------------------------------------------------------------------

def setup_logging(log_level: int = logging.DEBUG, log_file: Optional[str] = None) -> logging.Logger:
    """Configure root logger and return it."""

    logger = logging.getLogger()

    # Avoid duplicate handlers if the module is re‑imported (e.g. under pytest)
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ----------------------------------------------------------------------------
# Seeding ---------------------------------------------------------------------
# ----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set NumPy and TensorFlow RNG seeds for reproducibility."""

    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ----------------------------------------------------------------------------
# Monte‑Carlo inference utilities --------------------------------------------
# ----------------------------------------------------------------------------

# Mixed precision (convs etc. in FP16, math‑sensitive ops will be re‑cast)
mixed_precision.set_global_policy("mixed_float16")


def _predict_logits(x: tf.Tensor, model: tf.keras.Model) -> tf.Tensor:
    """Raw logits from the model *in float32* with dropout active."""

    logits_fp16 = model(x, training=True)  # honours dropout layers
    return tf.cast(logits_fp16, tf.float32)


@tf.function(jit_compile=True)
def mc_infer(
        x: tf.Tensor,
        *,
        n_iter: int,
        model: tf.keras.Model,
        num_classes: int,
        temperature: float = 1.0,
) -> Dict[str, tf.Tensor]:
    """Vectorised MC‑Dropout with epistemic / aleatoric metrics.

    Parameters
    ----------
    x
        Input batch [B, H, W, C].
    n_iter
        Number of stochastic forward passes.
    model
        Keras model returning logits.
    num_classes
        #classes for normalisation.
    temperature
        Positive scalar for post‑hoc temperature scaling.
    """

    # ---------------------------------------------------------------------
    # One fused forward pass ------------------------------------------------
    # ---------------------------------------------------------------------
    batch_size = tf.shape(x)[0]
    x_tiled = tf.repeat(x, repeats=n_iter, axis=0)  # [n_iter⋅B, ...]
    logits = _predict_logits(x_tiled, model)  # float32

    h, w = tf.shape(logits)[1], tf.shape(logits)[2]
    logits = tf.reshape(logits, (n_iter, batch_size, h, w, num_classes))  # [T, B, H, W, K]

    # Temperature scaling --------------------------------------------------
    logits_scaled = logits / tf.cast(temperature, tf.float32)
    probs = tf.nn.softmax(logits_scaled, axis=-1)  # [T, B, H, W, K]

    # Expected values across iterations -----------------------------------
    mean_probs = tf.reduce_mean(probs, axis=0)  # [B, H, W, K]
    mean_logits = tf.reduce_mean(logits_scaled, axis=0)

    # Predictive entropy (total) ------------------------------------------
    entropy = -tf.reduce_sum(mean_probs * tf.math.log(mean_probs + 1e-6), axis=-1)  # [B, H, W]
    max_entropy = tf.math.log(tf.cast(num_classes, tf.float32))
    entropy_norm = entropy / max_entropy  # 0‑1

    # Expected entropy -----------------------------------------------------
    expected_entropy = -tf.reduce_mean(
        tf.reduce_sum(probs * tf.math.log(probs + 1e-6), axis=-1), axis=0
    )

    # BALD (mutual information) -------------------------------------------
    bald = entropy - expected_entropy

    return {
        "entropy": entropy_norm,
        "bald": bald,
        "mean_probs": mean_probs,
        "mean_logits": mean_logits,
    }


# ----------------------------------------------------------------------------
# Main processing -------------------------------------------------------------
# ----------------------------------------------------------------------------

def main(config: Dict[str, Any], *, logger: logging.Logger, resume: bool = False) -> None:
    """Run MC inference, compute uncertainties, and stream them to HDF5."""

    # ---------------------------------------------------------------------
    # Seed & I/O paths -----------------------------------------------------
    # ---------------------------------------------------------------------
    set_global_seed(int(config.get("SEED", 42)))

    input_size: List[int] = config["INPUT_SIZE"]  # [H, W, C]
    if len(input_size) != 3:
        raise ValueError("INPUT_SIZE must be [H, W, C]")
    h_in, w_in, c_in = input_size

    num_classes: int = config["OUT_CHANNELS"]
    batch_size: int = config.get("BATCH_SIZE", 1)

    model_path = Path(config["MODEL_DIR"]) / f"{config['MODEL_NAME']}"
    logger.info("Loading model from %s", model_path)

    mixed_precision.set_global_policy("mixed_float16")

    model = load_model(
        model_path,
        custom_objects={
            "DropoutAttentionBlock": DropoutAttentionBlock,
            "GroupNormalization": GroupNormalization,
            "SpatialConcreteDropout": SpatialConcreteDropout,
        },
        compile=False,
    )
    logger.info("Model loaded. Switching output to penultimate layer (logits).")
    model = Model(inputs=model.input, outputs=model.layers[-2].output)

    # ---------------------------------------------------------------------
    # Dataset -------------------------------------------------------------
    # ---------------------------------------------------------------------
    ds = get_dataset_v2(
        data_dir=config["DATA_DIR"],
        filelists=config.get("FILE_LIST") or config.get("TRAINING_LIST"),
        repeat=False,
        shuffle=False,
        batch_size=batch_size,
        shuffle_buffer_size=config.get("SHUFFLE_BUFFER_SIZE", 256),
        out_channels=num_classes,
    )

    # ---------------------------------------------------------------------
    # Output file ---------------------------------------------------------
    # ---------------------------------------------------------------------
    if "OUTPUT_FILE" in config:
        output_file = Path(config["OUTPUT_FILE"])
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(config.get("OUTPUT_DIR", config["MODEL_DIR"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / (
            f"mc_{Path(config['MODEL_NAME']).stem}_inference_output.h5"
        )

    logger.info("Writing to %s", output_file)

    if not resume and output_file.exists():
        logger.error("Output file exists. Use --resume to append instead of overwrite.")
        sys.exit(1)

    hdf5_mode = "a" if resume else "w"

    # ---------------------------------------------------------------------
    # Uncertainty selection ----------------------------------------------
    # ---------------------------------------------------------------------
    uncertainty_types = [ut.lower() for ut in config.get("UNCERTAINTY_TYPES", ["variance", "bald", "entropy"])]

    # ---------------------------------------------------------------------
    # Optional cap on #samples --------------------------------------------
    # ---------------------------------------------------------------------
    n_samples_cfg = int(config.get("N_SAMPLES", -1))
    file_list_path = config.get("FILE_LIST") or config.get("TRAINING_LIST")
    if n_samples_cfg == -1:
        with open(file_list_path, "r") as f:
            total_samples = sum(1 for _ in f if _.strip())
    else:
        total_samples = n_samples_cfg

    logger.info("Total samples to process: %s", total_samples)

    # ---------------------------------------------------------------------
    # Create / open HDF5 ---------------------------------------------------
    # ---------------------------------------------------------------------
    chunk_size = int(config.get("CHUNK_SIZE", batch_size))
    if chunk_size < 1:
        chunk_size = batch_size

    with h5py.File(output_file, hdf5_mode) as h5f:
        # Create datasets if new file -----------------------------------
        if "rgb_images" not in h5f:
            maxshape_img = (None, h_in, w_in, c_in)
            maxshape_logits = (None, h_in, w_in, num_classes)

            rgb_ds = h5f.create_dataset(
                "rgb_images",
                shape=(0, h_in, w_in, c_in),
                maxshape=maxshape_img,
                dtype="uint8",
                compression="gzip",
                compression_opts=4,
                chunks=(chunk_size, h_in, w_in, c_in),
            )
            logits_ds = h5f.create_dataset(
                "logits",
                shape=(0, h_in, w_in, num_classes),
                maxshape=maxshape_logits,
                dtype="float32",
                compression="gzip",
                compression_opts=4,
                chunks=(chunk_size, h_in, w_in, num_classes),
            )
            fname_ds = h5f.create_dataset(
                "filenames",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
                compression="gzip",
                compression_opts=4,
                chunks=(chunk_size,),
            )
            unc_ds: Dict[str, h5py.Dataset] = {}
            for ut in uncertainty_types:
                ds_name = ut if ut != "total" else "total_uncertainty"
                unc_ds[ut] = h5f.create_dataset(
                    ds_name,
                    shape=(0, h_in, w_in),
                    maxshape=(None, h_in, w_in),
                    dtype="float32",
                    compression="gzip",
                    compression_opts=4,
                    chunks=(chunk_size, h_in, w_in),
                )
            h5f.attrs["last_written_index"] = 0
        else:
            rgb_ds = h5f["rgb_images"]
            logits_ds = h5f["logits"]
            fname_ds = h5f["filenames"]
            unc_ds = {
                ut: h5f[ut if ut != "total" else "total_uncertainty"]
                for ut in uncertainty_types
            }

        # -----------------------------------------------------------------
        # Resume offset ----------------------------------------------------
        # -----------------------------------------------------------------
        start_idx = int(h5f.attrs.get("last_written_index", 0))
        logger.info("Starting from index %d", start_idx)

        # Skip dataset accordingly ---------------------------------------
        if start_idx > 0:
            ds = ds.skip(start_idx)

        samples_remaining = max(total_samples - start_idx, 0)
        if samples_remaining == 0:
            logger.info("Nothing left to process; exiting.")
            return

        ds = ds.take(math.ceil(samples_remaining / batch_size))

        # -----------------------------------------------------------------
        # Main loop -------------------------------------------------------
        # -----------------------------------------------------------------
        pbar = tqdm(total=samples_remaining, unit="samples", desc="Processing")
        idx = start_idx

        for filenames, x_batch, _ in ds:  # y_batch unused for inference
            bsz_actual = x_batch.shape[0]

            # ----------------------------------- Uncertainties ----------
            uncs = mc_infer(
                x_batch,
                n_iter=int(config["MC_N_ITER"]),
                model=model,
                num_classes=num_classes,
                temperature=float(config.get("TEMPERATURE", 1.0)),
            )

            # Cast to numpy early (1 cpu copy) --------------------------
            x_uint8 = (x_batch.numpy() * 255).astype("uint8")
            filenames_str = [f.decode("utf-8") for f in filenames.numpy()]

            new_size = idx + bsz_actual

            # Resize only when required (avoid HDF5 no‑op cost) ---------
            if new_size > rgb_ds.shape[0]:
                rgb_ds.resize((new_size, h_in, w_in, c_in))
                logits_ds.resize((new_size, h_in, w_in, num_classes))
                fname_ds.resize((new_size,))
                for ut_ds in unc_ds.values():
                    ut_ds.resize((new_size, h_in, w_in))

            # Write ------------------------------------------------------
            rgb_ds[idx:new_size] = x_uint8
            logits_ds[idx:new_size] = uncs["mean_logits"].numpy().astype("float16")
            fname_ds[idx:new_size] = filenames_str

            for ut in uncertainty_types:
                data_np = uncs[ut if ut != "total" else "entropy"].numpy().astype("float16")
                unc_ds[ut][idx:new_size] = data_np

            idx += bsz_actual
            h5f.attrs["last_written_index"] = idx
            pbar.update(bsz_actual)

        pbar.close()
        logger.info("Finished writing %d samples.", idx - start_idx)


# ----------------------------------------------------------------------------
# CLI ------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MC‑Dropout inference with uncertainty outputs")
    parser.add_argument("-c", "--config_path", required=True, help="YAML configuration file")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log_file", help="Path to log file")
    parser.add_argument("--output_dir", help="Override OUTPUT_DIR from YAML")
    parser.add_argument("--output_file", help="Override OUTPUT_FILE from YAML")
    parser.add_argument("--resume", action="store_true", help="Append to existing HDF5 instead of overwrite")

    args = parser.parse_args()

    logger = setup_logging(getattr(logging, args.log_level.upper(), logging.INFO), args.log_file)

    try:
        with open(args.config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
        if args.output_dir:
            cfg["OUTPUT_DIR"] = args.output_dir
        if args.output_file:
            cfg["OUTPUT_FILE"] = args.output_file

        required = [
            "MODEL_DIR",
            "MODEL_NAME",
            "DATA_DIR",
            "FILE_LIST",
            "MC_N_ITER",
        ]
        missing = [k for k in required if k not in cfg]
        if missing:
            logger.error("Missing required config keys: %s", missing)
            sys.exit(1)

        main(cfg, logger=logger, resume=args.resume)

    except Exception:
        logger.exception("Fatal error. Exiting.")
        sys.exit(1)
