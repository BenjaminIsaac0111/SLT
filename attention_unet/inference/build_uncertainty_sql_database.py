#!/usr/bin/env python3
"""
MC‑Dropout inference with uncertainty outputs, stored in SQLite.
"""

import argparse
import logging
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
import yaml
from keras import Model
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model
from tqdm import tqdm

from attention_unet.dataloader.dataloader import get_dataset_v2
from attention_unet.models.custom_layers import (
    DropoutAttentionBlock,
    GroupNormalization,
    SpatialConcreteDropout,
)


# ----------------------------------------------------------------------------
# Logging ---------------------------------------------------------------------
# ----------------------------------------------------------------------------

def setup_logging(log_level: int = logging.DEBUG, log_file: Optional[str] = None) -> logging.Logger:
    """Configure root logger and return it."""
    logger = logging.getLogger()
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(log_level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ----------------------------------------------------------------------------
# Seeding ---------------------------------------------------------------------
# ----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set NumPy and TensorFlow RNG seeds for reproducibility."""
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ----------------------------------------------------------------------------
# MC‑Dropout inference --------------------------------------------------------
# ----------------------------------------------------------------------------

mixed_precision.set_global_policy("mixed_float16")


def _predict_logits(x: tf.Tensor, model: tf.keras.Model) -> tf.Tensor:
    """Raw logits from the model *in float32* with dropout active."""
    logits_fp16 = model(x, training=True)
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
    """
    Perform vectorized MC‑Dropout inference.
    Returns entropy (normalized), variance, BALD, mean_probs, mean_logits.
    """
    batch_size = tf.shape(x)[0]
    x_tiled = tf.repeat(x, repeats=n_iter, axis=0)
    logits = _predict_logits(x_tiled, model)
    h, w = tf.shape(logits)[1], tf.shape(logits)[2]
    logits = tf.reshape(logits, (n_iter, batch_size, h, w, num_classes))

    logits_scaled = logits / tf.cast(temperature, tf.float32)
    probs = tf.nn.softmax(logits_scaled, axis=-1)

    mean_probs = tf.reduce_mean(probs, axis=0)
    entropy = -tf.reduce_sum(mean_probs * tf.math.log(mean_probs + 1e-6), axis=-1)
    max_entropy = tf.math.log(tf.cast(num_classes, tf.float32))
    entropy_norm = entropy / max_entropy

    expected_entropy = -tf.reduce_mean(
        tf.reduce_sum(probs * tf.math.log(probs + 1e-6), axis=-1), axis=0
    )
    bald = entropy - expected_entropy
    variance = tf.reduce_sum(tf.math.reduce_variance(probs, axis=0), axis=-1)

    return {
        "entropy": entropy_norm,
        "variance": variance,
        "bald": bald,
        "mean_probs": mean_probs,
        "mean_logits": tf.reduce_mean(logits_scaled, axis=0),
    }


# ----------------------------------------------------------------------------
# SQLite storage helper ------------------------------------------------------
# ----------------------------------------------------------------------------

class SQLiteStorage:
    """Helper for storing arrays and metadata in an SQLite database."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, timeout=30)
        self._init_db()

    def _init_db(self) -> None:
        c = self.conn.cursor()
        c.executescript("""
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS samples (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename  TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shapes (
            array_id  INTEGER PRIMARY KEY,
            ndims     INTEGER NOT NULL,
            dim0      INTEGER NOT NULL,
            dim1      INTEGER,
            dim2      INTEGER,
            dim3      INTEGER,
            dtype     TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS arrays (
            array_id   INTEGER,
            sample_id  INTEGER NOT NULL REFERENCES samples(sample_id),
            name       TEXT NOT NULL,
            data       BLOB NOT NULL,
            FOREIGN KEY(array_id) REFERENCES shapes(array_id)
        );
        """)
        self.conn.commit()

    def _next_array_id(self) -> int:
        cur = self.conn.execute("SELECT IFNULL(MAX(array_id),0)+1 FROM shapes")
        return cur.fetchone()[0]

    def add_sample(self, filename: str) -> int:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO samples(filename) VALUES (?)", (filename,)
        )
        self.conn.commit()
        cur = self.conn.execute(
            "SELECT sample_id FROM samples WHERE filename=?", (filename,)
        )
        return cur.fetchone()[0]

    def add_array(self, sample_id: int, name: str, arr: np.ndarray) -> None:
        array_id = self._next_array_id()
        dims = list(arr.shape) + [None] * (4 - arr.ndim)
        self.conn.execute(
            "INSERT INTO shapes(array_id, ndims, dim0, dim1, dim2, dim3, dtype) VALUES (?,?,?,?,?,?,?)",
            (array_id, arr.ndim, dims[0], dims[1], dims[2], dims[3], str(arr.dtype))
        )
        blob = arr.tobytes()
        self.conn.execute(
            "INSERT INTO arrays(array_id, sample_id, name, data) VALUES (?,?,?,?)",
            (array_id, sample_id, name, sqlite3.Binary(blob))
        )
        self.conn.commit()

    def get_max_sample_id(self) -> int:
        cur = self.conn.execute("SELECT IFNULL(MAX(sample_id),0) FROM samples")
        return cur.fetchone()[0]

    def close(self) -> None:
        self.conn.close()


# ----------------------------------------------------------------------------
# Main processing -------------------------------------------------------------
# ----------------------------------------------------------------------------

def main(config: Dict[str, Any], logger: logging.Logger) -> None:
    set_global_seed(int(config.get("SEED", 42)))

    # Config parameters
    input_size = config["INPUT_SIZE"]  # [H, W, C]
    num_classes = config["OUT_CHANNELS"]
    batch_size = config["BATCH_SIZE"]

    model_path = Path(config["MODEL_DIR"]) / f"best_{config['MODEL_NAME']}"
    logger.info("Loading model from %s", model_path)
    model = load_model(
        model_path,
        custom_objects={
            "DropoutAttentionBlock": DropoutAttentionBlock,
            "GroupNormalization": GroupNormalization,
            "SpatialConcreteDropout": SpatialConcreteDropout,
        },
        compile=False
    )
    # Use penultimate layer for logits
    model = Model(inputs=model.input, outputs=model.layers[-2].output)

    # Dataset
    ds = get_dataset_v2(
        data_dir=config["DATA_DIR"],
        filelists=config["TRAINING_LIST"],
        repeat=False,
        shuffle=False,
        batch_size=batch_size,
        shuffle_buffer_size=config["SHUFFLE_BUFFER_SIZE"],
        out_channels=num_classes,
    )

    # Determine total samples
    n_samples_cfg = int(config.get("N_SAMPLES", -1))
    if n_samples_cfg == -1:
        with open(config["TRAINING_LIST"], "r") as f:
            total_samples = sum(1 for _ in f if _.strip())
    else:
        total_samples = n_samples_cfg
    logger.info("Total samples to process: %d", total_samples)

    # Prepare SQLite DB
    output_dir = Path(config.get("OUTPUT_DIR", config["MODEL_DIR"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(output_dir / (f"dropout_{Path(config['MODEL_NAME']).stem}.sqlite"))
    logger.info("Using SQLite database at %s", db_path)
    storage = SQLiteStorage(db_path)

    # Resume logic
    start_id = storage.get_max_sample_id()
    logger.info("Resuming from sample ID %d", start_id)
    samples_remaining = max(total_samples - start_id, 0)
    if samples_remaining == 0:
        logger.info("Nothing left to process; exiting.")
        return

    ds = ds.skip(start_id)
    ds = ds.take(math.ceil(samples_remaining / batch_size))

    pbar = tqdm(total=samples_remaining, unit="samples", desc="Processing")
    for filenames, x_batch, _ in ds:
        uncs = mc_infer(
            x_batch,
            n_iter=int(config["MC_N_ITER"]),
            model=model,
            num_classes=num_classes,
            temperature=float(config.get("TEMPERATURE", 1.0)),
        )
        x_uint8 = (x_batch.numpy() * 255).astype("uint8")
        logits_np = uncs["mean_logits"].numpy().astype("float32")
        ent_np = uncs["entropy"].numpy().astype("float32")
        var_np = uncs["variance"].numpy().astype("float32")
        bald_np = uncs["bald"].numpy().astype("float32")

        for fname_b, img, logits_arr, e_arr, v_arr, b_arr in zip(
                filenames.numpy(), x_uint8, logits_np, ent_np, var_np, bald_np
        ):
            fname = fname_b.decode("utf-8")
            sid = storage.add_sample(fname)
            storage.add_array(sid, "rgb", img)
            storage.add_array(sid, "logits", logits_arr)
            storage.add_array(sid, "entropy", e_arr)
            storage.add_array(sid, "variance", v_arr)
            storage.add_array(sid, "bald", b_arr)
            pbar.update(1)

    pbar.close()
    storage.close()
    logger.info("Finished writing %d new samples.", samples_remaining)


# ----------------------------------------------------------------------------
# CLI entry point -------------------------------------------------------------
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MC‑Dropout inference with uncertainties, stored in SQLite"
    )
    parser.add_argument(
        "-c", "--config_path", required=True,
        help="YAML configuration file"
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    parser.add_argument("--log_file", help="Path to log file")
    parser.add_argument(
        "--output_dir", help="Override OUTPUT_DIR from YAML"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing SQLite instead of overwrite"
    )
    args = parser.parse_args()

    logger = setup_logging(getattr(logging, args.log_level), args.log_file)
    try:
        with open(args.config_path, "r") as fh:
            cfg = yaml.safe_load(fh)
        if args.output_dir:
            cfg["OUTPUT_DIR"] = args.output_dir
        required = [
            "MODEL_DIR", "MODEL_NAME", "DATA_DIR", "TRAINING_LIST",
            "BATCH_SIZE", "SHUFFLE_BUFFER_SIZE", "OUT_CHANNELS",
            "INPUT_SIZE", "MC_N_ITER"
        ]
        missing = [k for k in required if k not in cfg]
        if missing:
            logger.error("Missing required config keys: %s", missing)
            sys.exit(1)
        main(cfg, logger=logger)
    except Exception:
        logger.exception("Fatal error. Exiting.")
        sys.exit(1)
