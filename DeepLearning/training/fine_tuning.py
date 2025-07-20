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
import shutil
import signal
from collections import Counter
from dataclasses import dataclass
from datetime import time
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
    log_every_n_batches: int = 1

    # Derived fields (set in __post_init__)
    ckpt_dir: Path | None = None
    h5_ckpt_path: Path | None = None  # optional H5 export per epoch

    def __post_init__(self):
        self.ckpt_dir = self.model_dir / f"tuned_{self.model_name}"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.h5_ckpt_path = self.ckpt_dir / f"tuned_ckpt_{self.model_name}.h5"


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
        self.model = model
        self.C = num_classes
        self.strategy = strategy
        self.use_xla = use_xla
        self.window_size = window_size
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
            # We'll gather only fg indices; if none fg -> invalid
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
            logits = self.model(images, training=False)  # (B,H,W,C)
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
        @tf.function(jit_compile=self.use_xla)
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
        return {
            "confusion_matrix": tf.cast(cm, tf.int32),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
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
            self.cleanup_stale_ckpt_temps(Path(self.ckpt_manager.directory))
            latest = self.ckpt_manager.latest_checkpoint
            if latest:
                self.ckpt.restore(latest).expect_partial()
                tf.print(
                    f"[INFO] Restored checkpoint state from {latest} (global_step={int(self.global_step.numpy())})")

        # Dataset (outside strategy is acceptable if it returns per-replica elements; adjust if using distribute)
        self.dataset = self._build_training_dataset()
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
        ckpt_path = self.robust_save(step=int(self.global_step.numpy()))
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
    parser.add_argument("--num_patches", type=int, default=10)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--no_xla", action="store_false", dest="use_xla", help="Disable XLA JIT.")
    parser.add_argument("--log_every_n_batches", type=int, default=1, help="Scalar logging frequency (in batches)")
    args = parser.parse_args(argv)
    return Config(**vars(args))


def main(argv: Sequence[str] | None = None):
    cfg = parse_args(argv)
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
