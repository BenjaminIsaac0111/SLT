#!/usr/bin/env python3
"""train_tuned_model_refactored_with_step_logging.py
=====================================================
Fine‑tunes a segmentation/classification model with focal loss using
mixed‑precision training. This version improves *ordered, resumable*
TensorBoard logging and checkpointing compared to the earlier script.

Key Improvements Over Previous Version
--------------------------------------
* **Monotonically increasing global step** persisted in `tf.train.Checkpoint`.
* **Epoch counter persisted** so training resumes from the last completed epoch.
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
import shutil
import signal
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import tensorflow as tf
from tensorflow.keras import mixed_precision, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Progbar

# ────────────────────────────────────────────────────────────────────────────────
# Project‑specific imports (assumed to exist in your environment)
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
        except Exception as exc:
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


def count_samples_from_json(json_path: Path) -> int:
    with open(json_path, "r") as f:
        ann = json.load(f)
    return len(ann)

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
    log_every_n_batches: int = 1

    val_labels_json: Path | None = None
    val_images_dir: Path | None = None
    num_val_patches: int = 100
    validate_every: int = 1
    calibrate_every: int = 0

    # Derived fields (set in __post_init__)
    ckpt_dir: Path | None = None
    h5_ckpt_path: Path | None = None  # optional H5 export per epoch

    def __post_init__(self):
        self.ckpt_dir = self.model_dir / f"tuned_{self.model_name}"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.h5_ckpt_path = self.ckpt_dir / f"tuned_ckpt_{self.model_name}.h5"
        if self.val_images_dir is None:
            self.val_images_dir = self.images_dir


# ────────────────────────────────────────────────────────────────────
# Validation subsystem
# ────────────────────────────────────────────────────────────────────

class Validator:
    """
    Stateful validator accumulating a confusion matrix and computing
    derived metrics. Supports XLA for the per-batch update step.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 num_classes: int,
                 strategy: tf.distribute.Strategy,
                 use_xla: bool = True,
                 window_size: int = 3,
                 label_policy: str = "window_majority",  # options: center_pixel, window_majority, window_mean_argmax
                 skip_empty: bool = True):
        self.use_xla = use_xla
        self.model = model
        # 1) Create a small XLA‑compiled fn for model inference
        if self.use_xla:
            self._infer = tf.function(
                lambda images: self.model(images, training=False),
                jit_compile=True
            )
        else:
            # fallback to normal graph or even eager if you like
            self._infer = tf.function(
                lambda images: self.model(images, training=False),
                jit_compile=False
            )

        self.window_size = window_size
        self.C = num_classes
        self.strategy = strategy
        self.label_policy = label_policy
        self.skip_empty = skip_empty
        # confusion matrix variable
        with self.strategy.scope():
            self.cm_var = tf.Variable(
                tf.zeros((self.C, self.C), dtype=tf.int64),
                trainable=False,
                name="val_confusion_matrix"
            )
        self._build_batch_fn()

    def reset(self):
        self.cm_var.assign(tf.zeros_like(self.cm_var))

    def _build_batch_fn(self):
        ws = self.window_size

        def extract_labels(one_hot):
            """
            Return (labels, valid_mask)
            labels shape: (B,), int32
            valid_mask: (B,), bool
            """
            # one_hot: (B,H,W,C)
            if self.label_policy == "center_pixel":
                h = tf.shape(one_hot)[1] // 2
                w = tf.shape(one_hot)[2] // 2
                center = one_hot[:, h:h + 1, w:w + 1, :]  # (B,1,1,C)
                labels = tf.argmax(center, axis=-1, output_type=tf.int32)[:, 0, 0]
                if self.skip_empty:
                    valid = tf.reduce_max(center, axis=-1)[:, 0, 0] > 0.5
                else:
                    valid = tf.ones_like(labels, dtype=tf.bool)
                return labels, valid

            # window extraction
            h = tf.shape(one_hot)[1] // 2
            w = tf.shape(one_hot)[2] // 2
            half = ws // 2
            win = one_hot[:, h - half:h + half + 1, w - half:w + half + 1, :]  # (B,ws,ws,C)

            if self.label_policy == "window_mean_argmax":
                # Mean over window then argmax
                win_mean = tf.reduce_mean(win, axis=(1, 2))  # (B,C)
                labels = tf.argmax(win_mean, axis=-1, output_type=tf.int32)
                if self.skip_empty:
                    valid = tf.reduce_max(win_mean, axis=-1) > 0.5
                else:
                    valid = tf.ones_like(labels, dtype=tf.bool)
                return labels, valid

            # default: majority (mode) over discrete one-hot pixels
            # Convert to int class indices for all fg pixels
            pix_labels = tf.argmax(win, axis=-1, output_type=tf.int32)  # (B,ws,ws)
            fg_mask = tf.reduce_max(win, axis=-1) > 0.5  # (B,ws,ws)
            # Flatten
            pix_labels_flat = tf.reshape(pix_labels, (tf.shape(pix_labels)[0], -1))
            fg_flat = tf.reshape(fg_mask, (tf.shape(fg_mask)[0], -1))
            # We will compute bincount per sample using segment_ids trick
            B = tf.shape(pix_labels_flat)[0]
            K = tf.shape(pix_labels_flat)[1]

            # Masked values: set background class 0 where not fg (or optionally mark invalid)
            # Gather only fg indices; if none fg -> invalid
            # Gather all fg positions
            # To avoid ragged loops: use boolean_mask then compute counts via unsorted segment sum.
            idx = tf.where(fg_flat)  # (M,2) pairs (sample,row)
            sample_ids = idx[:, 0]
            label_vals = tf.gather_nd(pix_labels_flat, idx)  # (M,)
            # One-hot those labels and segment-sum
            one_hot_counts = tf.one_hot(label_vals, depth=self.C, dtype=tf.int32)  # (M,C)
            counts = tf.math.unsorted_segment_sum(one_hot_counts, sample_ids, B)  # (B,C)
            # Determine if a sample had any fg
            had_fg = tf.reduce_any(fg_flat, axis=1)
            # Mode
            labels = tf.argmax(counts, axis=-1, output_type=tf.int32)
            return labels, had_fg if self.skip_empty else tf.ones_like(labels, tf.bool)

        def batch_confusion(y_true_onehot, images):
            # images used only for forward pass
            logits = self._infer(images)  # (B,H,W,C)
            # Prediction window match training window logic: same center window
            h = tf.shape(logits)[1] // 2
            w = tf.shape(logits)[2] // 2
            half = ws // 2
            win_pred = logits[:, h - half:h + half + 1, w - half:w + half + 1, :]  # (B,ws,ws,C)
            win_pred_mean = tf.reduce_mean(win_pred, axis=(1, 2))  # (B,C)
            pred_labels = tf.argmax(win_pred_mean, axis=-1, output_type=tf.int32)  # (B,)

            true_labels, valid_mask = extract_labels(y_true_onehot)
            true_labels = tf.boolean_mask(true_labels, valid_mask)
            pred_labels = tf.boolean_mask(pred_labels, valid_mask)

            # If after masking empty
            def empty_case():
                return tf.zeros((self.C, self.C), dtype=tf.int64)

            def non_empty_case():
                # Confusion matrix via bincount of (true * C + pred)
                combined = true_labels * self.C + pred_labels
                flat_counts = tf.math.bincount(
                    combined,
                    minlength=self.C * self.C,
                    maxlength=self.C * self.C,
                    dtype=tf.int64
                )
                return tf.reshape(flat_counts, (self.C, self.C))

            return tf.cond(tf.size(true_labels) > 0, non_empty_case, empty_case)

        # Compile per-batch update
        @tf.function(jit_compile=False)
        def update_step(images, one_hot):
            batch_cm = batch_confusion(one_hot, images)
            self.cm_var.assign_add(batch_cm)

        self._update_step = update_step

    def update(self, images, one_hot):
        """
        Accumulate confusion matrix for a batch. Handles distribution.
        """
        # If distributed, run on replicas
        if isinstance(self.strategy, tf.distribute.Strategy) and \
                not isinstance(self.strategy, tf.distribute.OneDeviceStrategy):
            def replica_fn(imgs, y):
                self._update_step(imgs, y)

            self.strategy.run(replica_fn, args=(images, one_hot))
        else:
            self._update_step(images, one_hot)

    def result(self):
        cm = tf.cast(self.cm_var.read_value(), tf.float32)  # (C,C)
        tp = tf.linalg.tensor_diag_part(cm)
        pred_pos = tf.reduce_sum(cm, axis=0)
        actual_pos = tf.reduce_sum(cm, axis=1)
        precision = tf.math.divide_no_nan(tp, pred_pos)
        recall = tf.math.divide_no_nan(tp, actual_pos)
        f1 = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
        accuracy = tf.math.divide_no_nan(tf.reduce_sum(tp), tf.reduce_sum(cm))
        support = actual_pos
        macro_f1 = tf.reduce_mean(f1)
        weighted_f1 = tf.math.divide_no_nan(tf.reduce_sum(f1 * support), tf.reduce_sum(support))
        balanced_accuracy = tf.reduce_mean(recall)
        total = tf.reduce_sum(cm)
        pe = tf.reduce_sum(pred_pos * actual_pos) / (total * total)
        po = tf.reduce_sum(tp) / total
        kappa = tf.math.divide_no_nan(po - pe, 1.0 - pe)
        return {
            "confusion_matrix": tf.cast(cm, tf.int32),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_accuracy,
            "kappa": kappa,
            "support": support,
        }


# ────────────────────────────────────────────────────────────────────────────────
# Trainer encapsulating workflow
# ────────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tab10 = [
            (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
            (1.0, 0.4980392156862745, 0.054901960784313725),
            (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
            (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
            (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
            (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
            (0.8941176470588236, 0.4666666666666667, 0.7607843137254902),
            (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
            (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
            (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
        ]

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
            # Training progress variables persisted in the checkpoint
            self.global_step = tf.Variable(
                0, trainable=False, dtype=tf.int64, name="global_step"
            )
            self.epoch = tf.Variable(
                0, trainable=False, dtype=tf.int64, name="epoch"
            )
            self.compiled_train_step = self._build_train_step()

            # Persisted best‐so‐far metrics:
            self.best_val_accuracy = tf.Variable(
                -1.0, trainable=False, dtype=tf.float32, name="best_val_accuracy"
            )
            self.best_val_macro_f1 = tf.Variable(
                -1.0, trainable=False, dtype=tf.float32, name="best_val_macro_f1"
            )
            self.best_val_balanced_acc = tf.Variable(
                -1.0, trainable=False, dtype=tf.float32, name="best_val_balanced_acc"
            )
            self.best_val_weighted_f1 = tf.Variable(
                -1.0, trainable=False, dtype=tf.float32, name="best_val_weighted_f1"
            )
            self.best_val_kappa = tf.Variable(
                -1.0, trainable=False, dtype=tf.float32, name="best_val_kappa"
            )

            # Baseline metrics (pre-training evaluation)
            self.baseline_val_accuracy = tf.Variable(
                -1.0, trainable=False, dtype=tf.float32, name="baseline_val_accuracy"
            )
            self.baseline_val_macro_f1 = tf.Variable(
                -1.0, trainable=False, dtype=tf.float32, name="baseline_val_macro_f1"
            )
            self.baseline_val_balanced_acc = tf.Variable(
                -1.0, trainable=False, dtype=tf.float32, name="baseline_val_balanced_acc"
            )
            self.baseline_val_weighted_f1 = tf.Variable(
                -1.0, trainable=False, dtype=tf.float32, name="baseline_val_weighted_f1"
            )
            self.baseline_val_kappa = tf.Variable(
                -1.0, trainable=False, dtype=tf.float32, name="baseline_val_kappa"
            )

            # Include them in the checkpoint
            self.ckpt = tf.train.Checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                global_step=self.global_step,
                epoch=self.epoch,
                best_val_accuracy=self.best_val_accuracy,
                best_val_macro_f1=self.best_val_macro_f1,
                best_val_balanced_acc=self.best_val_balanced_acc,
                best_val_weighted_f1=self.best_val_weighted_f1,
                best_val_kappa=self.best_val_kappa,
                baseline_val_accuracy=self.baseline_val_accuracy,
                baseline_val_macro_f1=self.baseline_val_macro_f1,
                baseline_val_balanced_acc=self.baseline_val_balanced_acc,
                baseline_val_weighted_f1=self.baseline_val_weighted_f1,
                baseline_val_kappa=self.baseline_val_kappa,
            )
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt,
                directory=str(self.cfg.ckpt_dir / "ckpts"),
                max_to_keep=5,
            )

            # Restore (if any) – this will also restore best_val_accuracy & best_val_macro_f1
            latest = self.ckpt_manager.latest_checkpoint
            if latest:
                self.ckpt.restore(latest).expect_partial()
                tf.print(
                    f"[INFO] Restored checkpoint from {latest} "
                    f"(epoch={int(self.epoch.numpy())}, "
                    f"global_step={int(self.global_step.numpy())}, "
                    f"best_acc={self.best_val_accuracy.numpy():.4f}, "
                    f"best_macro_f1={self.best_val_macro_f1.numpy():.4f}, "
                    f"best_bal_acc={self.best_val_balanced_acc.numpy():.4f}, "
                    f"best_weighted_f1={self.best_val_weighted_f1.numpy():.4f}, "
                    f"best_kappa={self.best_val_kappa.numpy():.4f}, "
                    f"baseline_acc={self.baseline_val_accuracy.numpy():.4f}, "
                    f"baseline_macro_f1={self.baseline_val_macro_f1.numpy():.4f}, "
                    f"baseline_bal_acc={self.baseline_val_balanced_acc.numpy():.4f}, "
                    f"baseline_weighted_f1={self.baseline_val_weighted_f1.numpy():.4f}, "
                    f"baseline_kappa={self.baseline_val_kappa.numpy():.4f})"
                )
            # remember whether we resumed
            self._did_restore = latest is not None

        # Dataset (outside strategy is acceptable if it returns per-replica elements; adjust if using distribute)
        self.dataset = self._build_training_dataset()
        if self.cfg.num_patches == -1:
            total_samples = count_samples_from_json(self.cfg.labels_json)
            self.steps_per_epoch = max(1, (total_samples + self.cfg.batch_size - 1) // self.cfg.batch_size)
        else:
            self.steps_per_epoch = max(1, self.cfg.num_patches // self.cfg.batch_size)

        # Validation dataset and helper
        self.val_dataset = self._build_validation_dataset()

        if self.cfg.num_val_patches == -1:
            total_val_samples = count_samples_from_json(self.cfg.val_labels_json)
            self.val_steps = max(1, (total_val_samples + self.cfg.batch_size - 1) // self.cfg.batch_size)
        else:
            self.val_steps = max(1, self.cfg.num_val_patches // self.cfg.batch_size) if self.val_dataset else 0

        self.validator = Validator(
            self.model,
            self.cfg.num_classes,
            self.strategy,
            use_xla=self.cfg.use_xla,
        )

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

    def _build_training_dataset(self) -> tf.data.Dataset:
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

    def _build_validation_dataset(self) -> tf.data.Dataset | None:
        if not self.cfg.val_labels_json:
            return None
        if self.cfg.val_labels_json:
            ds = (
                get_dataset_from_json_v2(
                    json_path=self.cfg.val_labels_json,
                    images_dir=str(self.cfg.val_images_dir),
                    batch_size=self.cfg.batch_size,
                    repeat=False,
                    shuffle=False,
                    transforms=None,
                )
                .map(self._extract_xy, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            ds = (
                get_dataset_from_dir_v2(
                    images_dir=str(self.cfg.val_images_dir),
                    batch_size=self.cfg.batch_size,
                    repeat=False,
                    transforms=None,
                    shuffle_buffer_size=False,
                )
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

    def _extract_labels_for_calibration(self, one_hot):
        ws = self.validator.window_size
        if self.validator.label_policy == "center_pixel":
            h = tf.shape(one_hot)[1] // 2
            w = tf.shape(one_hot)[2] // 2
            center = one_hot[:, h:h + 1, w:w + 1, :]
            labels = tf.argmax(center, axis=-1, output_type=tf.int32)[:, 0, 0]
            valid = tf.reduce_max(center, axis=-1)[:, 0, 0] > 0.5 if self.validator.skip_empty else tf.ones_like(labels, dtype=tf.bool)
            return labels, valid

        h = tf.shape(one_hot)[1] // 2
        w = tf.shape(one_hot)[2] // 2
        half = ws // 2
        win = one_hot[:, h - half:h + half + 1, w - half:w + half + 1, :]
        if self.validator.label_policy == "window_mean_argmax":
            win_mean = tf.reduce_mean(win, axis=(1, 2))
            labels = tf.argmax(win_mean, axis=-1, output_type=tf.int32)
            valid = tf.reduce_max(win_mean, axis=-1) > 0.5 if self.validator.skip_empty else tf.ones_like(labels, dtype=tf.bool)
            return labels, valid

        pix_labels = tf.argmax(win, axis=-1, output_type=tf.int32)
        fg_mask = tf.reduce_max(win, axis=-1) > 0.5
        pix_labels_flat = tf.reshape(pix_labels, (tf.shape(pix_labels)[0], -1))
        fg_flat = tf.reshape(fg_mask, (tf.shape(fg_mask)[0], -1))
        idx = tf.where(fg_flat)
        sample_ids = idx[:, 0]
        label_vals = tf.gather_nd(pix_labels_flat, idx)
        one_hot_counts = tf.one_hot(label_vals, depth=self.cfg.num_classes, dtype=tf.int32)
        counts = tf.math.unsorted_segment_sum(one_hot_counts, sample_ids, tf.shape(pix_labels_flat)[0])
        had_fg = tf.reduce_any(fg_flat, axis=1)
        labels = tf.argmax(counts, axis=-1, output_type=tf.int32)
        valid = had_fg if self.validator.skip_empty else tf.ones_like(labels, dtype=tf.bool)
        return labels, valid

    # ────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────

    def fit(self):
        tf.print("[INFO] Starting training…")

        if not self._did_restore:
            # only on a fresh start
            self._evaluate_baseline()
        else:
            tf.print(
                f"[INFO] Resuming at epoch {int(self.epoch.numpy())}, "
                f"step {int(self.global_step.numpy())} – skipping baseline."
            )

        # 2. Standard training loop
        start_epoch = int(self.epoch.numpy())
        for e in range(start_epoch, self.cfg.epochs):
            if self._stop_training:
                break
            epoch = e + 1
            self._train_one_epoch(epoch)
            if epoch % self.cfg.validate_every == 0:
                self._validate_one_epoch(epoch)
            self.epoch.assign(epoch)
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

    def _validate_one_epoch(self, epoch: int):
        if self.val_dataset is None:
            return
        self.validator.reset()
        prog = Progbar(self.val_steps, stateful_metrics=None, verbose=1, unit_name="val_step")
        for step, batch in enumerate(self.val_dataset.take(self.val_steps), start=1):
            x, y = batch
            self.validator.update(x, y)
            prog.update(step)
        metrics = self.validator.result()
        gstep = int(self.global_step.numpy())

        # 1. Grab raw counts and cast to float
        cm = tf.cast(metrics["confusion_matrix"], tf.float32)  # shape (C, C)
        cm_raw_img = tf.reshape(cm, [1, self.cfg.num_classes, self.cfg.num_classes, 1])
        with self.tb_writer.as_default():
            tf.summary.image("val/confusion_matrix_raw", cm_raw_img, step=gstep)
            self.tb_writer.flush()

        # 2. Convert to row percentages for better interpretability
        row_totals = tf.reduce_sum(cm, axis=1, keepdims=True)
        cm_percent = tf.math.divide_no_nan(cm * 100.0, row_totals)
        cm_img = tf.reshape(cm_percent / 100.0, [1, self.cfg.num_classes, self.cfg.num_classes, 1])

        # 3. Write to TensorBoard and flush
        with self.tb_writer.as_default():
            tf.summary.image("val/confusion_matrix_percent", cm_img, step=gstep)
            self.tb_writer.flush()


        ece = None
        if self.cfg.calibrate_every > 0 and epoch % self.cfg.calibrate_every == 0:
            import numpy as np
            from DeepLearning.inference.temperature_calibration import compute_reliability_curve

            confs = []
            correct = []
            for step, batch in enumerate(self.val_dataset.take(self.val_steps), start=1):
                x, y = batch
                logits = self.model(x, training=False)
                h = tf.shape(logits)[1] // 2
                w = tf.shape(logits)[2] // 2
                half = self.validator.window_size // 2
                win_pred = logits[:, h - half:h + half + 1, w - half:w + half + 1, :]
                win_mean = tf.reduce_mean(win_pred, axis=(1, 2))
                probs = tf.nn.softmax(win_mean, axis=-1)
                pred_labels = tf.argmax(probs, axis=-1, output_type=tf.int32)
                conf = tf.reduce_max(probs, axis=-1)

                # extract true labels using same policy
                labels, valid = self._extract_labels_for_calibration(y)
                labels = tf.boolean_mask(labels, valid)
                pred_labels = tf.boolean_mask(pred_labels, valid)
                conf = tf.boolean_mask(conf, valid)
                confs.append(conf.numpy())
                correct.append(tf.cast(tf.equal(pred_labels, labels), tf.float32).numpy())

            if confs:
                confs_np = np.concatenate(confs)
                corr_np = np.concatenate(correct)
                _, _, _, ece_val = compute_reliability_curve(confs_np, corr_np)
                ece = float(ece_val)
                with self.tb_writer.as_default():
                    tf.summary.scalar("val/ece", ece, step=gstep)
                self.tb_writer.flush()

        current_acc = float(metrics["accuracy"].numpy())
        current_f1 = float(metrics["macro_f1"].numpy())
        current_bal_acc = float(metrics["balanced_accuracy"].numpy())
        current_weighted_f1 = float(metrics["weighted_f1"].numpy())
        current_kappa = float(metrics["kappa"].numpy())

        if current_acc > self.best_val_accuracy:
            self.best_val_accuracy = current_acc
            best_acc_path = self.cfg.ckpt_dir / "best_accuracy_model.h5"
            self.model.save(best_acc_path)
            tf.print(f"[INFO] New best accuracy {current_acc:.4f} → {best_acc_path}")

        if current_f1 > self.best_val_macro_f1:
            self.best_val_macro_f1 = current_f1
            best_f1_path = self.cfg.ckpt_dir / "best_f1_model.h5"
            self.model.save(best_f1_path)
            tf.print(f"[INFO] New best macro_f1 {current_f1:.4f} → {best_f1_path}")

        if current_bal_acc > self.best_val_balanced_acc:
            self.best_val_balanced_acc = current_bal_acc
            best_bal_path = self.cfg.ckpt_dir / "best_balanced_accuracy_model.h5"
            self.model.save(best_bal_path)
            tf.print(f"[INFO] New best balanced_accuracy {current_bal_acc:.4f} → {best_bal_path}")

        if current_weighted_f1 > self.best_val_weighted_f1:
            self.best_val_weighted_f1 = current_weighted_f1
            best_wf1_path = self.cfg.ckpt_dir / "best_weighted_f1_model.h5"
            self.model.save(best_wf1_path)
            tf.print(f"[INFO] New best weighted_f1 {current_weighted_f1:.4f} → {best_wf1_path}")

        if current_kappa > self.best_val_kappa:
            self.best_val_kappa = current_kappa
            best_kappa_path = self.cfg.ckpt_dir / "best_kappa_model.h5"
            self.model.save(best_kappa_path)
            tf.print(f"[INFO] New best kappa {current_kappa:.4f} → {best_kappa_path}")

        gstep = int(self.global_step.numpy())
        with self.tb_writer.as_default():
            tf.summary.scalar("val/best_accuracy", self.best_val_accuracy, step=gstep)
            tf.summary.scalar("val/best_macro_f1", self.best_val_macro_f1, step=gstep)
            tf.summary.scalar("val/best_balanced_accuracy", self.best_val_balanced_acc, step=gstep)
            tf.summary.scalar("val/best_weighted_f1", self.best_val_weighted_f1, step=gstep)
            tf.summary.scalar("val/best_kappa", self.best_val_kappa, step=gstep)
            self.tb_writer.flush()

        metrics = self.validator.result()
        gstep = int(self.global_step.numpy())
        with self.tb_writer.as_default():
            tf.summary.scalar("val/accuracy", metrics["accuracy"], step=gstep)
            tf.summary.scalar("val/macro_f1", metrics["macro_f1"], step=gstep)
            tf.summary.scalar("val/weighted_f1", metrics["weighted_f1"], step=gstep)
            tf.summary.scalar("val/balanced_accuracy", metrics["balanced_accuracy"], step=gstep)
            tf.summary.scalar("val/kappa", metrics["kappa"], step=gstep)
        self.tb_writer.flush()
        tf.print(
            f"[Epoch {epoch}] val_accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"balanced_acc={metrics['balanced_accuracy']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} "
            f"kappa={metrics['kappa']:.4f}"
        )

    def _evaluate_baseline(self):
        """Run validation once on the freshly loaded model to get pre‑training metrics."""
        if self.val_dataset is None:
            tf.print("[WARN] No validation dataset; skipping baseline evaluation.")
            return

        tf.print("[INFO] Evaluating baseline (pre‑training) model…")
        self.validator.reset()
        prog = Progbar(self.val_steps, stateful_metrics=None, verbose=1, unit_name="val_step")
        for step, batch in enumerate(self.val_dataset.take(self.val_steps), start=1):
            x, y = batch
            self.validator.update(x, y)
            prog.update(step)

        metrics = self.validator.result()
        gstep = int(self.global_step.numpy())  # should be 0 at this point

        # Log scalars (accuracy, f1) for baseline
        with self.tb_writer.as_default():
            tf.summary.scalar("val/accuracy", metrics["accuracy"], step=gstep)
            tf.summary.scalar("val/macro_f1", metrics["macro_f1"], step=gstep)
            tf.summary.scalar("val/weighted_f1", metrics["weighted_f1"], step=gstep)
            tf.summary.scalar("val/balanced_accuracy", metrics["balanced_accuracy"], step=gstep)
            tf.summary.scalar("val/kappa", metrics["kappa"], step=gstep)
            tf.summary.scalar("baseline/accuracy", metrics["accuracy"], step=gstep)
            tf.summary.scalar("baseline/macro_f1", metrics["macro_f1"], step=gstep)
            tf.summary.scalar("baseline/weighted_f1", metrics["weighted_f1"], step=gstep)
            tf.summary.scalar("baseline/balanced_accuracy", metrics["balanced_accuracy"], step=gstep)
            tf.summary.scalar("baseline/kappa", metrics["kappa"], step=gstep)
            self.tb_writer.flush()

        tf.print(
            f"[BASELINE] accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"balanced_acc={metrics['balanced_accuracy']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} "
            f"kappa={metrics['kappa']:.4f}"
        )

        # Persist baseline metrics and initialize "best so far"
        self.baseline_val_accuracy.assign(metrics["accuracy"])
        self.baseline_val_macro_f1.assign(metrics["macro_f1"])
        self.baseline_val_balanced_acc.assign(metrics["balanced_accuracy"])
        self.baseline_val_weighted_f1.assign(metrics["weighted_f1"])
        self.baseline_val_kappa.assign(metrics["kappa"])

        self.best_val_accuracy.assign(self.baseline_val_accuracy)
        self.best_val_macro_f1.assign(self.baseline_val_macro_f1)
        self.best_val_balanced_acc.assign(self.baseline_val_balanced_acc)
        self.best_val_weighted_f1.assign(self.baseline_val_weighted_f1)
        self.best_val_kappa.assign(self.baseline_val_kappa)

        # (Optional) Save this baseline snapshot as a reference
        baseline_path = self.cfg.ckpt_dir / "baseline_model.h5"
        self.model.save(baseline_path)
        tf.print(f"[INFO] Saved baseline model to {baseline_path}")

    def _checkpoint(self, epoch: int):
        # Object-based checkpoint (persists epoch & global_step)
        ckpt_path = self.robust_save(step=int(self.epoch.numpy()))
        tf.print(f"[INFO] Saved checkpoint: {ckpt_path}")
        # Optional: also export an H5 snapshot of the full model each epoch
        try:
            self.model.save(self.cfg.h5_ckpt_path)
        except Exception as exc:  # noqa: BLE001
            tf.print(f"[WARN] Could not save H5 snapshot: {exc}")

    def _interrupt_handler(self, *_):  # signal handler
        # Graceful interrupt: save state and flush summaries
        tf.print("[WARN] KeyboardInterrupt detected — saving checkpoint before exit… (interrupt handler)")
        self._stop_training = True
        self._checkpoint(epoch=-1)
        self.tb_writer.flush()
        tf.print("[INFO] Interrupt handling complete; exiting after current step.")

    # ────────────────────────────────────────────────────────────────────
    # Debug / visualization helpers
    # ────────────────────────────────────────────────────────────────────
    def sample_batch(self):
        """Return a *single* batch (x, y) from the training dataset for ad-hoc inspection."""
        batch = next(iter(self.dataset))
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            _, x, y = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError("Unexpected batch structure fetched from dataset.")
        return x, y

    def visualize_sample(self, save_path: str | None = None, class_colors: list | None = None, alpha: float = 0.5,
                         log_to_tb: bool = False):
        """Visualize first image/mask prediction pair.

        Handles ground-truth masks that may be one-hot (H, W, C) or single-channel (H, W).
        Also robust to stray last-dimension of size 1. Casts all images to float32 for matplotlib.

        Parameters
        ----------
        save_path : str | None
            If provided, saves the composite figure (PNG). Otherwise displays.
        class_colors : list | None
            Optional list/array of RGB triples in [0,1] of length num_classes.
        alpha : float
            Blend between input image and predicted mask for overlay panel.
        log_to_tb : bool
            If True, also logs the composite panel as a TensorBoard image summary.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import tensorflow as tf
        import io

        x, y = self.sample_batch()
        x0 = x[0:1]
        y0 = y[0]

        # Normalize ground truth shape to (H, W)
        if y0.ndim == 3:
            if y0.shape[-1] == self.cfg.num_classes:  # one-hot
                y0 = tf.argmax(y0, axis=-1)
            elif y0.shape[-1] == 1:  # redundant channel
                y0 = tf.squeeze(y0, axis=-1)
        y0 = tf.cast(y0, tf.int32)

        logits = self.model(x0, training=False)
        if logits.shape[-1] != self.cfg.num_classes:
            raise ValueError(f"Model output last dim {logits.shape[-1]} != num_classes {self.cfg.num_classes}")
        pred = tf.argmax(logits[0], axis=-1)
        pred = tf.cast(pred, tf.int32)

        num_classes = self.cfg.num_classes
        if class_colors is None:
            rng = np.random.default_rng(42)
            class_colors = rng.uniform(0.15, 0.95, size=(num_classes, 3))
            class_colors[0] = [0, 0, 0]
        class_colors = np.asarray(class_colors, dtype=np.float32)

        def colorize(mask_tensor):
            mask_np = mask_tensor.numpy() if isinstance(mask_tensor, tf.Tensor) else mask_tensor
            if mask_np.ndim == 3:
                if mask_np.shape[-1] == 1:
                    mask_np = mask_np[..., 0]
                elif mask_np.shape[-1] == num_classes:
                    mask_np = np.argmax(mask_np, axis=-1)
            h, w = mask_np.shape
            out = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(num_classes):
                out[mask_np == c] = class_colors[c]
            return out

        pred_color = colorize(pred)
        gt_color = colorize(y0)

        x_img = x0[0].numpy()
        if x_img.dtype == np.float16:
            x_img = x_img.astype(np.float32)
        if x_img.dtype != np.uint8:
            x_img = (x_img - x_img.min()) / (x_img.ptp() + 1e-8)
        if x_img.shape[-1] == 1:
            x_img = np.repeat(x_img, 3, axis=-1)
        x_img = x_img.astype(np.float32)
        overlay = ((1 - alpha) * x_img + alpha * pred_color).astype(np.float32)

        fig, axes = plt.subplots(1, 4, figsize=(16, 8))
        panels = [("Input", x_img), ("Ground Truth", gt_color), ("Prediction", pred_color), ("Overlay", overlay)]
        for ax, (title, img) in zip(axes, panels):
            ax.set_title(title)
            ax.imshow(img)
            ax.axis("off")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=180)
            tf.print(f"[DEBUG] Saved visualization to {save_path}")
        if log_to_tb:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120)
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            with self.tb_writer.as_default():
                tf.summary.image("debug/sample_panel", image, step=int(self.global_step.numpy()))
            self.tb_writer.flush()
            tf.print("[DEBUG] Logged sample panel to TensorBoard.")
        if not save_path:
            plt.show()
            plt.close(fig)
            tf.print(f"[DEBUG] Saved visualization to {save_path}")
        else:
            plt.show()

    def debug_save(self, step: int):
        import traceback, pathlib
        base = pathlib.Path(self.ckpt_manager.directory)
        before = set(base.iterdir())
        print(f"[DEBUG] Pre-save entries ({len(before)}): {[p.name for p in before]}")
        try:
            path = self.ckpt_manager.save(checkpoint_number=step)
            print(f"[DEBUG] manager.save returned: {path}")
        except Exception as e:
            print("[DEBUG] manager.save raised:", repr(e))
            traceback.print_exc()
        after = set(base.iterdir())
        new = [p for p in after - before]
        print(f"[DEBUG] New entries after attempt: {[p.name for p in new]}")
        temp_dirs = [p for p in after if p.is_dir() and p.name.endswith('_temp')]
        print(f"[DEBUG] Temp dirs currently present: {[d.name for d in temp_dirs]}")

    def robust_save(self, step, retries=3, delay=1.0):
        for a in range(1, retries + 1):
            try:
                return self.ckpt_manager.save(checkpoint_number=step)
            except tf.errors.NotFoundError as e:
                if a == retries:
                    raise
                time.sleep(delay)
                print(f"[WARN] Retry save attempt {a}/{retries} after NotFoundError: {e}")

    @staticmethod
    def cleanup_stale_ckpt_temps(root: Path):
        removed = []
        for d in root.glob("ckpt-*_*temp"):
            # Only remove if corresponding final checkpoint exists
            base = d.name.replace("_temp", "")
            data = root / f"{base}.data-00000-of-00001"
            index = root / f"{base}.index"
            if data.exists() and index.exists():
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d.name)
        if removed:
            print("[INFO] Removed stale temp dirs:", removed)


# ────────────────────────────────────────────────────────────────────────────────
# Argument parsing & entry point
# ────────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="Mixed-precision fine-tuning script")
    parser.add_argument("--labels_json", type=Path, required=True, help="Annotation JSON file")
    parser.add_argument("--images_dir", type=Path, required=True, help="Directory with image patches")
    parser.add_argument("--initial_weights", type=Path, required=True, help="Initial model .h5")
    parser.add_argument("--model_dir", type=Path, default=Path("fine_tuned_models"), help="Directory to store "
                                                                                          "checkpoints")
    parser.add_argument("--model_name", type=str, required=True, help="Base checkpoint name")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_patches", type=int, default=-1)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-7)
    parser.add_argument("--no_xla", action="store_false", dest="use_xla", help="Disable XLA JIT.")
    parser.add_argument("--log_every_n_batches", type=int, default=1, help="Scalar logging frequency (in batches)")
    parser.add_argument("--val_labels_json", type=Path, help="Validation annotation JSON file")
    parser.add_argument("--val_images_dir", type=Path, help="Directory with validation image patches")
    parser.add_argument("--num_val_patches", type=int, default=-1, help="Number of validation patches per epoch")
    parser.add_argument("--validate_every", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--calibrate_every", type=int, default=0, help="Compute calibration every N epochs (0=disable)")
    args = parser.parse_args(argv)
    return Config(**vars(args))


def main(argv: Sequence[str] | None = None):
    cfg = parse_args(argv)
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
