#!/usr/bin/env python3
"""train_tuned_model_refactored_with_step_logging.py
=====================================================
Fine‑tunes a segmentation/classification model with focal loss using
mixed‑precision training. This version improves *ordered, resumable*
TensorBoard logging and checkpointing compared to the earlier script.

Key Improvements Over Previous Version
--------------------------------------
* **Monotonically increasing global step** persisted in `tf.train.Checkpoint`.
* **Per‑batch and per‑epoch scalars** (`loss/batch`, `loss/epoch_avg`).
* **No duplicate steps after resume** – avoids out‑of‑order TensorBoard plots.
* **CheckpointManager** handles rotating checkpoints.
* **Graceful interrupt** still saves state (global step included).
* **Clear XLA flag semantics** via `--no_xla` to disable (default: enabled).
* **Writer flushes** after each epoch and on interrupt for timely visibility.

Only public TF 2.10.1 APIs and project modules are used.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import tensorflow as tf
from tensorflow.keras import mixed_precision, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Progbar

# ────────────────────────────────────────────────────────────────────────────────
# Project‑specific imports
# ────────────────────────────────────────────────────────────────────────────────
from DeepLearning.dataloader.dataloader import (
    get_dataset_from_json_v2,
    get_dataset_from_dir_v2,
)
from DeepLearning.losses.losses import focal_loss
from DeepLearning.models.custom_layers import (
    SpatialConcreteDropout,
    DropoutAttentionBlock,
    GroupNormalization,
)
from DeepLearning.processing.transforms import Transforms


# ────────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────────────────────

def set_memory_growth() -> None:
    """Enable dynamic memory allocation on all visible GPUs."""
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:  # noqa: BLE001
            tf.print(f"[WARN] Could not set memory growth on {gpu}: {exc}")


def compute_class_weights_from_json(json_path: Path, num_classes: int) -> list[float]:
    """Inverse‑frequency class weights — safe for missing classes."""
    with json_path.open() as fp:
        annotations = json.load(fp)

    counts = Counter(
        m["class_id"]
        for marks in annotations.values()
        for m in marks
        if 0 <= m["class_id"] < num_classes
    )

    total = sum(counts.values()) or 1  # avoid division by zero
    return [
        (total / (num_classes * counts[c])) if counts.get(c) else 0.0
        for c in range(num_classes)
    ]


def xla_optional(jit: bool = True):
    """Decorator that enables XLA if possible and logs a warning otherwise."""

    def decorator(fn):  # type: ignore
        if not jit:
            tf.print(f"[INFO] JIT disabled for `{fn.__name__}`; using plain graph.")
            return tf.function(fn, jit_compile=False)

        try:
            wrapped = tf.function(fn, jit_compile=True)
            tf.print(f"[INFO] XLA enabled for `{fn.__name__}`")
            return wrapped
        except (tf.errors.InvalidArgumentError, ValueError) as exc:
            tf.print(
                f"[WARN] XLA failed for `{fn.__name__}` – falling back to eager graph: {exc}"
            )
            return tf.function(fn, jit_compile=False)

    return decorator


@dataclass
class Config:
    """Aggregated hyper‑parameters and paths."""

    labels_json: Path
    images_dir: Path
    initial_weights: Path
    model_dir: Path
    model_name: str
    num_classes: int = 9
    batch_size: int = 4
    num_patches: int = 400
    shuffle_seed: int = 42
    shuffle_buffer_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-7
    use_xla: bool = True

    # Derived fields (set in __post_init__)
    ckpt_dir: Path | None = None
    h5_ckpt_path: Path | None = None  # optional H5 export per epoch

    def __post_init__(self):
        self.ckpt_dir = self.model_dir / f"tuned_{self.model_name}"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.h5_ckpt_path = self.ckpt_dir / f"tuned_ckpt_{self.model_name}.h5"


# ────────────────────────────────────────────────────────────────────────────────
# Trainer encapsulating workflow
# ────────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_memory_growth()
        mixed_precision.set_global_policy("mixed_float16")

        self.transforms = Transforms()
        self.class_weights = tf.constant(
            compute_class_weights_from_json(cfg.labels_json, cfg.num_classes),
            dtype=tf.float32,
        )

        self.strategy = tf.distribute.get_strategy()
        with self.strategy.scope():
            self.model = self._load_or_init_model()
            self.optimizer = self._build_optimizer()
            # Global step persisted in checkpoint (int64 to be safe for large counts)
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
            self.compiled_train_step = self._build_train_step()

            # Set up object-based checkpointing
            self.ckpt = tf.train.Checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                global_step=self.global_step,
            )
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt,
                directory=str(self.cfg.ckpt_dir / "ckpts"),
                max_to_keep=5,
            )
            latest = self.ckpt_manager.latest_checkpoint
            if latest:
                self.ckpt.restore(latest).expect_partial()
                tf.print(
                    f"[INFO] Restored checkpoint state from {latest} (global_step={int(self.global_step.numpy())})")

        # Dataset (outside strategy is acceptable if it returns per-replica elements; adjust if using distribute)
        self.dataset = self._build_dataset()
        self.steps_per_epoch = max(1, self.cfg.num_patches // self.cfg.batch_size)

        # TensorBoard writer
        logdir = self.cfg.ckpt_dir / "logs"
        logdir.mkdir(exist_ok=True, parents=True)
        self.tb_writer = tf.summary.create_file_writer(str(logdir))

        # Interrupt handling
        self._stop_training = False
        signal.signal(signal.SIGINT, self._interrupt_handler)

        self.start_wall_time = tf.timestamp()

    # ────────────────────────────────────────────────────────────────────
    # Build helpers
    # ────────────────────────────────────────────────────────────────────

    def _build_optimizer(self) -> optimizers.Optimizer:
        base_lr = self.cfg.learning_rate * self.cfg.batch_size
        opt = optimizers.Adam(learning_rate=base_lr)
        return mixed_precision.LossScaleOptimizer(opt)

    def _load_or_init_model(self):
        custom_objs = {
            "DropoutAttentionBlock": DropoutAttentionBlock,
            "GroupNormalization": GroupNormalization,
            "SpatialConcreteDropout": SpatialConcreteDropout,
        }

        # Decide whether to load initial weights or existing H5 snapshot
        if self.cfg.h5_ckpt_path.exists():
            tf.print(f"[INFO] Loading existing H5 weights {self.cfg.h5_ckpt_path}")
            return load_model(self.cfg.h5_ckpt_path, compile=False, custom_objects=custom_objs)
        tf.print(f"[INFO] Loading initial weights from {self.cfg.initial_weights}")
        return load_model(self.cfg.initial_weights, compile=False, custom_objects=custom_objs)

    @staticmethod
    def _extract_xy(*batch):
        """Return (x, y) regardless of dataset structure ((x,y) or (id,x,y))."""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                return batch[0], batch[1]
            if len(batch) == 3:
                return batch[1], batch[2]
        raise ValueError("Unexpected batch structure — expected (x,y) or (id,x,y).")

    def _build_dataset(self) -> tf.data.Dataset:
        if self.cfg.labels_json:
            ds = (
                get_dataset_from_json_v2(
                    json_path=self.cfg.labels_json,
                    images_dir=str(self.cfg.images_dir),
                    batch_size=self.cfg.batch_size,
                    repeat=True,
                    transforms=self.transforms,
                )
                .shuffle(self.cfg.shuffle_buffer_size, seed=self.cfg.shuffle_seed, reshuffle_each_iteration=True)
                .map(self._extract_xy, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            ds = (
                get_dataset_from_dir_v2(
                    images_dir=str(self.cfg.images_dir),
                    batch_size=self.cfg.batch_size,
                    repeat=True,
                    transforms=self.transforms,
                )
                .shuffle(self.cfg.shuffle_buffer_size, seed=self.cfg.shuffle_seed, reshuffle_each_iteration=True)
                .map(self._extract_xy, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
            )
        return ds

    def _build_train_step(self):
        @xla_optional(self.cfg.use_xla)
        def _train_step(batch):
            x, y = batch
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                loss = focal_loss(y_true=y, y_pred=logits, alpha_weights=self.class_weights)
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_weights)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            return loss

        return _train_step

    # ────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────

    def fit(self):
        tf.print("[INFO] Starting training…")
        for epoch in range(1, self.cfg.epochs + 1):
            if self._stop_training:
                break
            self._train_one_epoch(epoch)
            self._checkpoint(epoch)
        tf.print("[INFO] Training completed.")

    # ────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────

    def _train_one_epoch(self, epoch: int):
        prog = Progbar(self.steps_per_epoch, stateful_metrics=None, verbose=1, unit_name="step")
        running_loss = 0.0

        for step, batch in enumerate(self.dataset.take(self.steps_per_epoch), start=1):
            loss_tensor = self.compiled_train_step(batch)
            loss_value = float(loss_tensor.numpy())
            running_loss += loss_value
            avg_loss = running_loss / step

            # Increment global step (per batch)
            self.global_step.assign_add(1)
            gstep = int(self.global_step.numpy())

            # Per-batch logging
            with self.tb_writer.as_default():
                tf.summary.scalar("loss/batch", loss_value, step=gstep)
            prog.update(step, values=[("loss", loss_value), ("avg_loss", avg_loss)])

        # Epoch average logging using current global step
        with self.tb_writer.as_default():
            tf.summary.scalar("loss/epoch_avg", avg_loss, step=int(self.global_step.numpy()))
            tf.summary.scalar(
                "time/elapsed_sec", float(tf.timestamp() - self.start_wall_time), step=int(self.global_step.numpy())
            )
        self.tb_writer.flush()
        tf.print(f"[Epoch {epoch}] avg_loss = {avg_loss:.6f} (global_step={int(self.global_step.numpy())})")

    def _checkpoint(self, epoch: int):
        # Object-based checkpoint (persists global_step)
        ckpt_path = self.ckpt_manager.save(checkpoint_number=int(self.global_step.numpy()))
        tf.print(f"[INFO] Saved checkpoint: {ckpt_path}")
        # Optional: also export an H5 snapshot of the full model each epoch
        try:
            self.model.save(self.cfg.h5_ckpt_path)
        except Exception as exc:  # noqa: BLE001
            tf.print(f"[WARN] Could not save H5 snapshot: {exc}")

    def _interrupt_handler(self, *_):  # signal handler
        tf.print("\n[WARN] KeyboardInterrupt detected — saving checkpoint before exit…")
        self._stop_training = True
        self._checkpoint(epoch=-1)
        self.tb_writer.flush()


# ────────────────────────────────────────────────────────────────────────────────
# Argument parsing & entry point
# ────────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="Mixed‑precision fine‑tuning script (improved logging)")
    parser.add_argument("--labels_json", type=Path, required=True, help="Annotation JSON file")
    parser.add_argument("--images_dir", type=Path, required=True, help="Directory with image patches")
    parser.add_argument("--initial_weights", type=Path, required=True, help="Initial model .h5")
    parser.add_argument("--model_dir", type=Path, default=Path("models"), help="Directory to store checkpoints")
    parser.add_argument("--model_name", type=str, required=True, help="Base checkpoint name")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_patches", type=int, default=400)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1.0e-7)
    # Clear semantics: use XLA by default; pass --no_xla to disable.
    parser.add_argument("--no_xla", action="store_false", dest="use_xla", help="Disable XLA JIT compilation")
    args = parser.parse_args(argv)
    return Config(**vars(args))


def main(argv: Sequence[str] | None = None):
    cfg = parse_args(argv)
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
