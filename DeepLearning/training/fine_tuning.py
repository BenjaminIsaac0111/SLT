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
import math
import shutil
import signal
import time
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
from DeepLearning.dataloader.dataloader import get_dataset_from_json_v2, get_dataset_from_dir_v2
from DeepLearning.losses.losses import focal_loss
from DeepLearning.models.custom_layers import (
    SpatialConcreteDropout,
    DropoutAttentionBlock,
    GroupNormalization,
)
from DeepLearning.processing.transforms import Transforms
from DeepLearning.training.utils import (
    compute_class_weights_from_json,
    count_samples_from_json,
    set_memory_growth,
    xla_optional,
)
from DeepLearning.training.validator import Validator


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

    new_labels_json: Path | None = None
    new_images_dir: Path | None = None
    warmup_steps: int = 500
    decay_schedule: str = "half_life"
    half_life: int = 1000

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

        # Datasets for old and (optionally) new data
        self.old_ds = self._make_dataset(self.cfg.labels_json, self.cfg.images_dir)
        self.new_ds = (
            self._make_dataset(self.cfg.new_labels_json, self.cfg.new_images_dir)
            if self.cfg.new_labels_json
            else None
        )
        self.old_iter = iter(self.old_ds.repeat())
        self.new_iter = iter(self.new_ds.repeat()) if self.new_ds else None

        self.old_size = count_samples_from_json(self.cfg.labels_json)
        self.new_size = (
            count_samples_from_json(self.cfg.new_labels_json)
            if self.cfg.new_labels_json
            else 0
        )

        if self.cfg.num_patches == -1:
            total_samples = self.old_size
            self.steps_per_epoch = max(
                1, (total_samples + self.cfg.batch_size - 1) // self.cfg.batch_size
            )
        else:
            self.steps_per_epoch = max(
                1, self.cfg.num_patches // self.cfg.batch_size
            )

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

    def _make_dataset(
            self,
            labels_json: Path,
            images_dir: Path,
            *,
            repeat: bool = True
    ) -> tf.data.Dataset:
        # 1 ── Build a *finite* dataset first (no repeat, no batch)
        base = get_dataset_from_json_v2(
            json_path=labels_json,
            images_dir=str(images_dir),
            batch_size=None,
            repeat=False,  # important: keep it finite for shuffling
            transforms=None,
        )

        # 2 ── Shuffle ONCE PER EPOCH
        ds = base.shuffle(
            self.cfg.shuffle_buffer_size,
            seed=self.cfg.shuffle_seed,
            reshuffle_each_iteration=True,  # now it actually triggers
        )

        # 3 ── Repeat *after* shuffle so the iterator never terminates
        if repeat:
            ds = ds.repeat()

        # 4 ── Batch and map; everything downstream sees the same API
        ds = (
            ds.batch(self.cfg.batch_size, drop_remainder=False)
            .map(self._extract_xy, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        def _transform_fn(imgs, masks):
            imgs, masks = self.transforms(imgs, masks)
            return imgs, masks

        ds = ds.map(_transform_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds

    def p_new(self, step: int) -> float:
        """Probability of sampling from the new data at a given step."""
        if not self.new_iter:
            return 0.0
        if step < self.cfg.warmup_steps:
            return 1.0
        r = self.cfg.batch_size * self.new_size / (
            self.cfg.batch_size * self.old_size + 1e-8
        )
        if self.cfg.decay_schedule == "half_life":
            tau = self.cfg.half_life
            return max(r, 0.5 ** ((step - self.cfg.warmup_steps) / tau))
        if self.cfg.decay_schedule == "linear":
            t = max(0.0, 1 - (step - self.cfg.warmup_steps) / max(1, self.cfg.half_life))
            return max(r, t)
        if self.cfg.decay_schedule == "cosine":
            t = min(1.0, (step - self.cfg.warmup_steps) / max(1, self.cfg.half_life))
            return max(r, 0.5 * (1 + math.cos(math.pi * t)))
        return r

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

        for idx in range(1, self.steps_per_epoch + 1):
            step = int(self.global_step.numpy())
            if tf.random.uniform(()) < self.p_new(step):
                batch = next(self.new_iter)
            else:
                batch = next(self.old_iter)
            loss_tensor = self.compiled_train_step(batch)
            loss_value = float(loss_tensor.numpy())
            running_loss += loss_value
            avg_loss = running_loss / idx

            self.global_step.assign_add(1)
            gstep = int(self.global_step.numpy())

            with self.tb_writer.as_default():
                tf.summary.scalar("loss/batch", loss_value, step=gstep)
                tf.summary.scalar("sampling/p_new", self.p_new(step), step=step)
            prog.update(idx, values=[("loss", loss_value), ("avg_loss", avg_loss)])

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

        if current_acc > float(self.best_val_accuracy):
            self.best_val_accuracy.assign(current_acc)
            best_acc_path = self.cfg.ckpt_dir / "best_accuracy_model.h5"
            self.model.save(best_acc_path)
            tf.print(f"[INFO] New best accuracy {current_acc:.4f} → {best_acc_path}")

        if current_f1 > float(self.best_val_macro_f1):
            self.best_val_macro_f1.assign(current_f1)
            best_f1_path = self.cfg.ckpt_dir / "best_f1_model.h5"
            self.model.save(best_f1_path)
            tf.print(f"[INFO] New best macro_f1 {current_f1:.4f} → {best_f1_path}")

        if current_bal_acc > float(self.best_val_balanced_acc):
            self.best_val_balanced_acc.assign(current_bal_acc)
            best_bal_path = self.cfg.ckpt_dir / "best_balanced_accuracy_model.h5"
            self.model.save(best_bal_path)
            tf.print(f"[INFO] New best balanced_accuracy {current_bal_acc:.4f} → {best_bal_path}")

        if current_weighted_f1 > float(self.best_val_weighted_f1):
            self.best_val_weighted_f1.assign(current_weighted_f1)
            best_wf1_path = self.cfg.ckpt_dir / "best_weighted_f1_model.h5"
            self.model.save(best_wf1_path)
            tf.print(f"[INFO] New best weighted_f1 {current_weighted_f1:.4f} → {best_wf1_path}")

        if current_kappa > float(self.best_val_kappa):
            self.best_val_kappa.assign(current_kappa)
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
            tf.summary.scalar("baseline/best_accuracy", metrics["accuracy"], step=gstep)
            tf.summary.scalar("baseline/best_macro_f1", metrics["macro_f1"], step=gstep)
            tf.summary.scalar("baseline/best_weighted_f1", metrics["weighted_f1"], step=gstep)
            tf.summary.scalar("baseline/best_balanced_accuracy", metrics["balanced_accuracy"], step=gstep)
            tf.summary.scalar("baseline/best_kappa", metrics["kappa"], step=gstep)
            tf.summary.scalar("baseline/accuracy", metrics["accuracy"], step=gstep)
            tf.summary.scalar("baseline/macro_f1", metrics["macro_f1"], step=gstep)
            tf.summary.scalar("baseline/weighted_f1", metrics["weighted_f1"], step=gstep)
            tf.summary.scalar("baseline/balanced_accuracy", metrics["balanced_accuracy"], step=gstep)
            tf.summary.scalar("baseline/kappa", metrics["kappa"], step=gstep)
            tf.summary.scalar("baseline/best_accuracy", metrics["accuracy"], step=gstep)
            tf.summary.scalar("baseline/best_macro_f1", metrics["macro_f1"], step=gstep)
            tf.summary.scalar("baseline/best_weighted_f1", metrics["weighted_f1"], step=gstep)
            tf.summary.scalar("baseline/best_balanced_accuracy", metrics["balanced_accuracy"], step=gstep)
            tf.summary.scalar("baseline/best_kappa", metrics["kappa"], step=gstep)
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
    parser.add_argument("--new_labels_json", type=Path, help="Annotation JSON for new data")
    parser.add_argument("--new_images_dir", type=Path, help="Directory with new image patches")
    parser.add_argument("--warmup_steps", type=int, default=512, help="Number of warm-up steps using only new data")
    parser.add_argument(
        "--decay_schedule",
        type=str,
        default="half_life",
        help="Decay schedule for sampling probability",
    )
    parser.add_argument("--half_life", type=int, default=1000, help="Half-life parameter for exponential schedule")
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
