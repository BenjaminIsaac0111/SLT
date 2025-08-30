#!/usr/bin/env python3
"""train_tuned_model_refactored.py
==================================
Fine‑tunes a segmentation/classification model with focal loss using
mixed‑precision training. This version improves *ordered, resumable*
TensorBoard logging, checkpointing and adds CSV logging for validation
metrics compared to the earlier script.

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
import csv
import datetime
import math
import os
import shutil
import signal
import subprocess
import time
import uuid
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
    compute_class_histogram_from_json,
    count_samples_from_json,
    set_memory_growth,
    xla_optional,
)
from DeepLearning.training.validator import Validator
from DeepLearning.training.visualization import confusion_matrix_to_image

CLASS_NAMES = [
    "Non-Informative",  # 0
    "Tumour",  # 1
    "Stroma",  # 2
    "Necrosis",  # 3
    "Vessel",  # 4
    "Inflammation",  # 5
    "Tumour-Lumen",  # 6
    "Mucin",  # 7
    "Muscle",  # 8
]


@dataclass
class Config:
    """Aggregated hyper‑parameters and paths."""

    labels_json: Path
    images_dir: Path
    initial_weights: Path
    model_dir: Path
    model_name: str
    num_classes: int = 9
    batch_size: int = 1
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
    drw_warmup_steps: int = 1024  # how long to ramp α,γ AFTER new-data warmup
    decay_schedule: str = "half_life"
    half_life: int = 1000
    focal_gamma: float = 2.0
    prior_ema: float = 0.01
    use_drw: bool = False
    use_logit_adjustment: bool = False
    use_batch_alpha: bool = False
    use_penultimate_logits: bool = False

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
        self.start_time_utc = datetime.datetime.utcnow().isoformat()

        self.transforms = Transforms()
        old_counts = tf.constant(
            compute_class_histogram_from_json(cfg.labels_json, cfg.num_classes),
            dtype=tf.float32,
        )
        self.old_hist = old_counts / tf.reduce_sum(old_counts)
        if cfg.new_labels_json:
            new_counts = tf.constant(
                compute_class_histogram_from_json(cfg.new_labels_json, cfg.num_classes),
                dtype=tf.float32,
            )
            self.new_hist = new_counts / tf.reduce_sum(new_counts)
        else:
            self.new_hist = self.old_hist

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
            self.rng = tf.random.Generator.from_seed(self.cfg.shuffle_seed)
            self.class_prior = tf.Variable(
                tf.fill([self.cfg.num_classes], 1.0 / self.cfg.num_classes),
                trainable=False,
                dtype=tf.float32,
                name="class_prior",
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
                rng=self.rng,
                class_prior=self.class_prior,
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
        self._init_csv_logging()

    # ────────────────────────────────────────────────────────────────────
    # Build helpers
    # ────────────────────────────────────────────────────────────────────

    def _init_csv_logging(self) -> None:
        """Set up CSV log files and write run metadata."""
        self.run_id_file = self.cfg.ckpt_dir / "run_id.txt"
        if self.run_id_file.exists():
            self.run_id = self.run_id_file.read_text().strip()
        else:
            self.run_id = uuid.uuid4().hex
            self.run_id_file.write_text(self.run_id)

        self.csv_dir = self.cfg.ckpt_dir
        self.val_epochs_path = self.csv_dir / "val_epochs.csv"
        self.val_confusion_path = self.csv_dir / "val_confusion_wide.csv"
        self.val_per_class_path = self.csv_dir / "val_per_class.csv"
        self.run_meta_path = self.csv_dir / "run_meta.csv"

        self.val_epochs_logged = self._load_existing_ids(self.val_epochs_path)
        self.val_confusion_logged = self._load_existing_ids(self.val_confusion_path)
        self.val_per_class_logged = self._load_existing_ids(
            self.val_per_class_path, key_tuple=True
        )

        self._write_run_meta()

    def _load_existing_ids(self, path: Path, key_tuple: bool = False) -> set:
        ids: set = set()
        if path.exists():
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if key_tuple:
                        ids.add((int(row.get("global_step", 0)), int(row.get("class_id", -1))))
                    else:
                        ids.add(int(row.get("global_step", 0)))
        return ids

    def _write_row(
        self,
        path: Path,
        fieldnames: list[str],
        row: dict,
        existing: set,
        key: tuple | int | None,
    ) -> None:
        if key is not None and key in existing:
            return
        file_exists = path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
        if key is not None:
            existing.add(key)

    def _write_run_meta(self) -> None:
        devices = tf.config.list_physical_devices("GPU")
        gpu_name = ""
        if devices:
            try:
                gpu_name = tf.config.experimental.get_device_details(devices[0]).get(
                    "device_name", ""
                )
            except Exception:  # noqa: BLE001
                gpu_name = devices[0].name
        num_gpus = len(devices)
        cuda_version = os.environ.get("CUDA_VERSION", "")
        cudnn_version = os.environ.get("CUDNN_VERSION", "")
        policy = mixed_precision.global_policy().name

        repo_root = Path(__file__).resolve().parents[2]
        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
                .decode()
                .strip()
            )
            git_dirty = int(
                bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], cwd=repo_root
                    ).strip()
                )
            )
            repo_url = (
                subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"], cwd=repo_root
                )
                .decode()
                .strip()
            )
        except Exception:  # noqa: BLE001
            git_commit = ""
            git_dirty = 0
            repo_url = ""

        existing_runs = set()
        if self.run_meta_path.exists():
            with open(self.run_meta_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    existing_runs.add(r.get("run_id"))
        if self.run_id in existing_runs:
            return

        row = {
            "run_id": self.run_id,
            "labels_json": str(self.cfg.labels_json),
            "images_dir": str(self.cfg.images_dir),
            "new_labels_json": str(self.cfg.new_labels_json or ""),
            "new_images_dir": str(self.cfg.new_images_dir or ""),
            "val_labels_json": str(self.cfg.val_labels_json or ""),
            "val_images_dir": str(self.cfg.val_images_dir or ""),
            "initial_weights": str(self.cfg.initial_weights),
            "model_dir": str(self.cfg.model_dir),
            "model_name": self.cfg.model_name,
            "old_size": self.old_size,
            "new_size": self.new_size,
            "num_classes": self.cfg.num_classes,
            "batch_size": self.cfg.batch_size,
            "num_patches": self.cfg.num_patches,
            "epochs": self.cfg.epochs,
            "learning_rate": self.cfg.learning_rate,
            "shuffle_seed": self.cfg.shuffle_seed,
            "shuffle_buffer_size": self.cfg.shuffle_buffer_size,
            "use_xla": int(self.cfg.use_xla),
            "use_drw": int(self.cfg.use_drw),
            "use_logit_adjustment": int(self.cfg.use_logit_adjustment),
            "use_batch_alpha": int(self.cfg.use_batch_alpha),
            "use_penultimate_logits": int(self.cfg.use_penultimate_logits),
            "warmup_steps": self.cfg.warmup_steps,
            "drw_warmup_steps": self.cfg.drw_warmup_steps,
            "decay_schedule": self.cfg.decay_schedule,
            "half_life": self.cfg.half_life,
            "focal_gamma": self.cfg.focal_gamma,
            "prior_ema": self.cfg.prior_ema,
            "validate_every": self.cfg.validate_every,
            "num_val_patches": self.cfg.num_val_patches,
            "calibrate_every": self.cfg.calibrate_every,
            "log_every_n_batches": self.cfg.log_every_n_batches,
            "tf_version": tf.__version__,
            "cuda_version": cuda_version,
            "cudnn_version": cudnn_version,
            "gpu_name": gpu_name,
            "num_gpus": num_gpus,
            "mixed_precision_policy": policy,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "repo_url": repo_url,
            "restored_from_ckpt": int(self._did_restore),
            "start_epoch": int(self.epoch.numpy()),
            "start_global_step": int(self.global_step.numpy()),
            "start_time_utc": self.start_time_utc,
        }
        fieldnames = list(row.keys())
        self._write_row(self.run_meta_path, fieldnames, row, set(), None)

    # Logging helpers --------------------------------------------------
    def _log_val_epoch(
        self,
        epoch: int,
        gstep: int,
        metrics: dict,
        ece: float | None,
        flags: dict,
        paths: dict,
    ) -> None:
        row = {
            "run_id": self.run_id,
            "epoch": epoch,
            "global_step": gstep,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_weighted_f1": metrics["weighted_f1"],
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_kappa": metrics["kappa"],
            "val_ece": ece if ece is not None else "",
            "is_new_best_accuracy": int(flags["acc"]),
            "is_new_best_macro_f1": int(flags["macro_f1"]),
            "is_new_best_balanced_accuracy": int(flags["balanced_acc"]),
            "is_new_best_weighted_f1": int(flags["weighted_f1"]),
            "is_new_best_kappa": int(flags["kappa"]),
            "best_accuracy_so_far": float(self.best_val_accuracy.numpy()),
            "best_macro_f1_so_far": float(self.best_val_macro_f1.numpy()),
            "best_balanced_accuracy_so_far": float(self.best_val_balanced_acc.numpy()),
            "best_weighted_f1_so_far": float(self.best_val_weighted_f1.numpy()),
            "best_kappa_so_far": float(self.best_val_kappa.numpy()),
            "best_accuracy_path": paths.get("acc", ""),
            "best_f1_path": paths.get("macro_f1", ""),
            "best_balanced_accuracy_path": paths.get("balanced_acc", ""),
            "best_weighted_f1_path": paths.get("weighted_f1", ""),
            "best_kappa_path": paths.get("kappa", ""),
        }
        fieldnames = list(row.keys())
        self._write_row(self.val_epochs_path, fieldnames, row, self.val_epochs_logged, gstep)

    def _log_val_confusion(self, epoch: int, gstep: int, cm: tf.Tensor) -> None:
        cm = tf.cast(cm, tf.int32).numpy()
        cm_dict = {f"cm_{i}_{j}": int(cm[i, j]) for i in range(cm.shape[0]) for j in range(cm.shape[1])}
        row = {"run_id": self.run_id, "epoch": epoch, "global_step": gstep}
        row.update(cm_dict)
        fieldnames = list(row.keys())
        self._write_row(
            self.val_confusion_path, fieldnames, row, self.val_confusion_logged, gstep
        )

    def _log_val_per_class(self, epoch: int, gstep: int, cm: tf.Tensor, metrics: dict) -> None:
        cm_np = tf.cast(cm, tf.float32).numpy()
        total = cm_np.sum()
        for i in range(cm_np.shape[0]):
            tp = cm_np[i, i]
            fn = cm_np[i, :].sum() - tp
            fp = cm_np[:, i].sum() - tp
            tn = total - tp - fn - fp
            tpr = metrics["recall"][i].numpy()
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = metrics["precision"][i].numpy()
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            f1 = metrics["f1"][i].numpy()
            support = metrics["support"][i].numpy()
            row = {
                "run_id": self.run_id,
                "epoch": epoch,
                "global_step": gstep,
                "class_id": i,
                "class_name": CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i),
                "support": support,
                "precision": ppv,
                "recall": tpr,
                "f1": f1,
                "tpr": tpr,
                "tnr": tnr,
                "ppv": ppv,
                "npv": npv,
            }
            fieldnames = list(row.keys())
            key = (gstep, i)
            self._write_row(
                self.val_per_class_path,
                fieldnames,
                row,
                self.val_per_class_logged,
                key,
            )

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
            model = load_model(self.cfg.h5_ckpt_path, compile=False, custom_objects=custom_objs)
        else:
            tf.print(f"[INFO] Loading initial weights from {self.cfg.initial_weights}")
            model = load_model(self.cfg.initial_weights, compile=False, custom_objects=custom_objs)

        # If the last layer is Softmax, strip it to get logits
        last = model.layers[-1]
        last_act = getattr(last, "activation", None)
        if isinstance(last, tf.keras.layers.Softmax) or last_act is tf.keras.activations.softmax:
            tf.print("[INFO] Detected terminal Softmax — using pre-Softmax logits")
            model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
        else:
            tf.print("[INFO] Model already outputs logits")

        return model

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
        def _train_step(batch, alpha_t, prior_t, gamma_t, lambda_t):
            x, y = batch

            # Enforce dtypes/shapes expected by the math
            alpha_t = tf.cast(alpha_t, tf.float32)  # [C]
            prior_t = tf.cast(prior_t, tf.float32)  # [C]
            gamma_t = tf.cast(gamma_t, tf.float32)  # scalar
            lambda_t = tf.cast(lambda_t, tf.float32)  # scalar
            y = tf.cast(y, tf.float32)  # [B,H,W,C]

            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)  # [B,H,W,C], likely float16

                # --- Logit adjustment done in fp32 ---
                # clip prior to avoid log(0); broadcast to [1,1,1,C]
                prior_t = tf.clip_by_value(prior_t, 1e-6, 1.0)
                log_adj = lambda_t * tf.math.log(1.0 / prior_t)  # [C] fp32
                log_adj = log_adj[None, None, None, :]  # [1,1,1,C]

                logits_f32 = tf.cast(logits, tf.float32) + log_adj  # [B,H,W,C] fp32

                # Softmax in fp32; focal expects probabilities, not logits
                probs = tf.nn.softmax(logits_f32, axis=-1)  # [B,H,W,C] fp32

                # Compute focal loss in fp32
                loss = focal_loss(
                    y_true=y,
                    y_pred=probs,
                    gamma=gamma_t,  # tensor scalar is fine
                    alpha_weights=alpha_t  # [C]
                )

                # Loss scaling as usual
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
        alpha_last = None
        epoch_start = tf.timestamp()

        for idx in range(1, self.steps_per_epoch + 1):
            step = int(self.global_step.numpy())
            p = self.p_new(step)
            if self.rng.uniform(shape=[], dtype=tf.float32) < p:
                batch = next(self.new_iter)
                used_new = True
            else:
                batch = next(self.old_iter)
                used_new = False

            x, y = batch
            counts = tf.reduce_sum(y, axis=[0, 1, 2])
            count_sum = tf.reduce_sum(counts)
            batch_hist = tf.math.divide_no_nan(counts, count_sum)
            batch_alpha = tf.reduce_sum(batch_hist) / tf.maximum(batch_hist, 1e-8)
            batch_alpha = tf.minimum(batch_alpha, 10.0 * tf.reduce_mean(batch_alpha))

            p_t = tf.convert_to_tensor(p, dtype=tf.float32)
            mix_hist = p_t * self.new_hist + (1.0 - p_t) * self.old_hist
            dataset_alpha = tf.reduce_sum(mix_hist) / tf.maximum(mix_hist, 1e-8)
            dataset_alpha = tf.minimum(dataset_alpha, 10.0 * tf.reduce_mean(dataset_alpha))

            if self.cfg.use_batch_alpha:
                alpha_combined = tf.cond(
                    count_sum > 0.0,
                    lambda: 0.5 * dataset_alpha + 0.5 * batch_alpha,
                    lambda: dataset_alpha,
                )
            else:
                alpha_combined = dataset_alpha

            if self.cfg.use_drw:
                step_f = tf.cast(self.global_step, tf.float32)
                post = tf.nn.relu(step_f - float(self.cfg.warmup_steps))
                ramp = tf.minimum(1.0, post / float(self.cfg.drw_warmup_steps))
                alpha_t = (1.0 - ramp) * tf.ones_like(alpha_combined) + ramp * alpha_combined
                gamma_t = ramp * self.cfg.focal_gamma
            else:
                ramp = 1.0
                alpha_t = alpha_combined
                gamma_t = self.cfg.focal_gamma

            lambda_t = ramp if self.cfg.use_logit_adjustment else 0.0
            prior_t = tf.identity(self.class_prior)

            loss_tensor = self.compiled_train_step(
                (x, y), alpha_t, prior_t, gamma_t, lambda_t
            )
            loss_value = float(loss_tensor.numpy())
            running_loss += loss_value
            avg_loss = running_loss / idx
            alpha_last = alpha_t

            if self.cfg.use_logit_adjustment:
                batch_prior = batch_hist
                self.class_prior.assign(
                    (1.0 - self.cfg.prior_ema) * self.class_prior
                    + self.cfg.prior_ema * batch_prior
                )

            self.global_step.assign_add(1)
            gstep = int(self.global_step.numpy())

            if gstep % self.cfg.log_every_n_batches == 0:
                with self.tb_writer.as_default():
                    tf.summary.scalar("loss/batch", loss_value, step=gstep)
                    tf.summary.scalar("sampling/p_new", p, step=gstep)
                    tf.summary.scalar("focal/gamma_t", gamma_t, step=gstep)
                    tf.summary.scalar("logit/lambda_t", lambda_t, step=gstep)
            prog.update(idx, values=[("loss", loss_value), ("avg_loss", avg_loss)])

        # Epoch average logging using current global step
        with self.tb_writer.as_default():
            tf.summary.scalar("loss/epoch_avg", avg_loss, step=int(self.global_step.numpy()))
            tf.summary.scalar(
                "time/elapsed_sec", float(tf.timestamp() - self.start_wall_time), step=int(self.global_step.numpy())
            )
            if alpha_last is not None:
                tf.summary.histogram("alpha_t", alpha_last, step=int(self.global_step.numpy()))
                for i, val in enumerate(tf.unstack(alpha_last)):
                    tf.summary.scalar(f"alpha_t/class_{i}", val, step=int(self.global_step.numpy()))
            if self.cfg.use_logit_adjustment:
                tf.summary.histogram("class_prior", self.class_prior, step=int(self.global_step.numpy()))
        self.tb_writer.flush()
        tf.print(
            f"[Epoch {epoch}] avg_loss = {avg_loss:.6f} (global_step={int(self.global_step.numpy())})"
        )
        secs = float(tf.timestamp() - epoch_start)
        return avg_loss, alpha_last, secs

    def _validate_one_epoch(self, epoch: int):
        if self.val_dataset is None:
            return
        self.validator.reset()

        step_f = tf.cast(self.global_step, tf.float32)
        if self.cfg.use_drw:
            post = tf.nn.relu(step_f - float(self.cfg.warmup_steps))
            ramp = tf.minimum(1.0, post / float(self.cfg.drw_warmup_steps))
        else:
            ramp = 1.0
        lambda_t = ramp if self.cfg.use_logit_adjustment else 0.0

        prog = Progbar(self.val_steps, stateful_metrics=None, verbose=1, unit_name="val_step")
        for step, batch in enumerate(self.val_dataset.take(self.val_steps), start=1):
            x, y = batch
            self.validator.update(x, y, self.class_prior, lambda_t)
            prog.update(step)
        metrics = self.validator.result()
        gstep = int(self.global_step.numpy())

        # Convert confusion matrix to percentage values for interpretability
        cm = tf.cast(metrics["confusion_matrix"], tf.float32)  # shape (C, C)
        row_totals = tf.reduce_sum(cm, axis=1, keepdims=True)
        cm_percent = tf.math.divide_no_nan(cm * 100.0, row_totals)
        cm_image = confusion_matrix_to_image(cm_percent, CLASS_NAMES)

        # Write percentage confusion matrix to TensorBoard and flush
        with self.tb_writer.as_default():
            tf.summary.image("val/confusion_matrix", cm_image, step=gstep)
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
                logits = tf.cast(logits, tf.float32)
                if self.cfg.use_logit_adjustment:
                    log_adj = lambda_t * tf.math.log(
                        1.0 / tf.clip_by_value(self.class_prior, 1e-6, 1.0)
                    )
                    logits += log_adj
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

        flags = {"acc": False, "macro_f1": False, "balanced_acc": False, "weighted_f1": False, "kappa": False}
        paths = {}

        if current_acc > float(self.best_val_accuracy):
            self.best_val_accuracy.assign(current_acc)
            best_acc_path = self.cfg.ckpt_dir / "best_accuracy_model.h5"
            self.model.save(best_acc_path)
            tf.print(f"[INFO] New best accuracy {current_acc:.4f} → {best_acc_path}")
            flags["acc"] = True
            paths["acc"] = str(best_acc_path)

        if current_f1 > float(self.best_val_macro_f1):
            self.best_val_macro_f1.assign(current_f1)
            best_f1_path = self.cfg.ckpt_dir / "best_f1_model.h5"
            self.model.save(best_f1_path)
            tf.print(f"[INFO] New best macro_f1 {current_f1:.4f} → {best_f1_path}")
            flags["macro_f1"] = True
            paths["macro_f1"] = str(best_f1_path)

        if current_bal_acc > float(self.best_val_balanced_acc):
            self.best_val_balanced_acc.assign(current_bal_acc)
            best_bal_path = self.cfg.ckpt_dir / "best_balanced_accuracy_model.h5"
            self.model.save(best_bal_path)
            tf.print(f"[INFO] New best balanced_accuracy {current_bal_acc:.4f} → {best_bal_path}")
            flags["balanced_acc"] = True
            paths["balanced_acc"] = str(best_bal_path)

        if current_weighted_f1 > float(self.best_val_weighted_f1):
            self.best_val_weighted_f1.assign(current_weighted_f1)
            best_wf1_path = self.cfg.ckpt_dir / "best_weighted_f1_model.h5"
            self.model.save(best_wf1_path)
            tf.print(f"[INFO] New best weighted_f1 {current_weighted_f1:.4f} → {best_wf1_path}")
            flags["weighted_f1"] = True
            paths["weighted_f1"] = str(best_wf1_path)

        if current_kappa > float(self.best_val_kappa):
            self.best_val_kappa.assign(current_kappa)
            best_kappa_path = self.cfg.ckpt_dir / "best_kappa_model.h5"
            self.model.save(best_kappa_path)
            tf.print(f"[INFO] New best kappa {current_kappa:.4f} → {best_kappa_path}")
            flags["kappa"] = True
            paths["kappa"] = str(best_kappa_path)

        gstep = int(self.global_step.numpy())
        with self.tb_writer.as_default():
            tf.summary.scalar("val/best_accuracy", self.best_val_accuracy, step=gstep)
            tf.summary.scalar("val/best_macro_f1", self.best_val_macro_f1, step=gstep)
            tf.summary.scalar(
                "val/best_balanced_accuracy", self.best_val_balanced_acc, step=gstep
            )
            tf.summary.scalar(
                "val/best_weighted_f1", self.best_val_weighted_f1, step=gstep
            )
            tf.summary.scalar("val/best_kappa", self.best_val_kappa, step=gstep)
            tf.summary.scalar("val/accuracy", metrics["accuracy"], step=gstep)
            tf.summary.scalar("val/macro_f1", metrics["macro_f1"], step=gstep)
            tf.summary.scalar("val/weighted_f1", metrics["weighted_f1"], step=gstep)
            tf.summary.scalar(
                "val/balanced_accuracy", metrics["balanced_accuracy"], step=gstep
            )
            tf.summary.scalar("val/kappa", metrics["kappa"], step=gstep)
            self.tb_writer.flush()
        tf.print(
            f"[Epoch {epoch}] val_accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"balanced_acc={metrics['balanced_accuracy']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} "
            f"kappa={metrics['kappa']:.4f}"
        )
        val_dict = {
            "accuracy": float(metrics["accuracy"].numpy()),
            "macro_f1": float(metrics["macro_f1"].numpy()),
            "weighted_f1": float(metrics["weighted_f1"].numpy()),
            "balanced_accuracy": float(metrics["balanced_accuracy"].numpy()),
            "kappa": float(metrics["kappa"].numpy()),
        }
        self._log_val_epoch(epoch, gstep, val_dict, ece, flags, paths)
        self._log_val_confusion(epoch, gstep, metrics["confusion_matrix"])
        self._log_val_per_class(epoch, gstep, metrics["confusion_matrix"], metrics)

    def _evaluate_baseline(self):
        """Run validation once on the freshly loaded model to get pre‑training metrics."""
        if self.val_dataset is None:
            tf.print("[WARN] No validation dataset; skipping baseline evaluation.")
            return

        tf.print("[INFO] Evaluating baseline (pre‑training) model…")
        self.validator.reset()

        step_f = tf.cast(self.global_step, tf.float32)
        if self.cfg.use_drw:
            post = tf.nn.relu(step_f - float(self.cfg.warmup_steps))
            ramp = tf.minimum(1.0, post / float(self.cfg.drw_warmup_steps))
        else:
            ramp = 1.0
        lambda_t = ramp if self.cfg.use_logit_adjustment else 0.0

        prog = Progbar(self.val_steps, stateful_metrics=None, verbose=1, unit_name="val_step")
        for step, batch in enumerate(self.val_dataset.take(self.val_steps), start=1):
            x, y = batch
            self.validator.update(x, y, self.class_prior, lambda_t)
            prog.update(step)

        metrics = self.validator.result()
        gstep = int(self.global_step.numpy())  # should be 0 at this point

        # Log scalars (accuracy, f1) for baseline and initialise best trackers
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
            tf.summary.scalar("val/best_accuracy", metrics["accuracy"], step=gstep)
            tf.summary.scalar("val/best_macro_f1", metrics["macro_f1"], step=gstep)
            tf.summary.scalar("val/best_weighted_f1", metrics["weighted_f1"], step=gstep)
            tf.summary.scalar(
                "val/best_balanced_accuracy", metrics["balanced_accuracy"], step=gstep
            )
            tf.summary.scalar("val/best_kappa", metrics["kappa"], step=gstep)
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

    def _checkpoint(self, epoch: int) -> tuple[str, int]:
        # Object-based checkpoint (persists epoch & global_step)
        ckpt_path = self.robust_save(step=int(self.epoch.numpy()))
        tf.print(f"[INFO] Saved checkpoint: {ckpt_path}")
        h5_saved = 0
        # Optional: also export an H5 snapshot of the full model each epoch
        try:
            self.model.save(self.cfg.h5_ckpt_path)
            h5_saved = 1
        except Exception as exc:  # noqa: BLE001
            tf.print(f"[WARN] Could not save H5 snapshot: {exc}")
        return ckpt_path, h5_saved

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
    parser.add_argument("--num_patches", type=int, default=64)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-7)
    parser.add_argument("--no_xla", action="store_false", dest="use_xla", help="Disable XLA JIT.")
    parser.add_argument("--log_every_n_batches", type=int, default=1, help="Scalar logging frequency (in batches)")
    parser.add_argument("--new_labels_json", type=Path, help="Annotation JSON for new data")
    parser.add_argument("--new_images_dir", type=Path, help="Directory with new image patches")
    parser.add_argument("--warmup_steps", type=int, default=512, help="Number of warm-up steps using only new data")
    parser.add_argument(
        "--drw_warmup_steps",
        type=int,
        default=1024,
        help="How long to ramp α,γ after new-data warmup",
    )
    parser.add_argument(
        "--decay_schedule",
        type=str,
        default="half_life",
        help="Decay schedule for sampling probability",
    )
    parser.add_argument("--half_life", type=int, default=1000, help="Half-life parameter for exponential schedule")
    parser.add_argument("--val_labels_json", type=Path, help="Validation annotation JSON file")
    parser.add_argument("--val_images_dir", type=Path, help="Directory with validation image patches")
    parser.add_argument("--num_val_patches", type=int, default=1024, help="Number of validation patches per epoch")
    parser.add_argument("--validate_every", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--calibrate_every", type=int, default=0, help="Compute calibration every N epochs (0=disable)")
    parser.add_argument("--use_drw", action="store_true", help="Enable deferred re-weighting for alpha and gamma")
    parser.add_argument("--use_logit_adjustment", action="store_true", help="Apply EMA prior logit adjustment")
    parser.add_argument("--use_batch_alpha", action="store_true", help="Blend per-batch histogram into alpha weights")
    parser.add_argument(
        "--use_penultimate_logits",
        action="store_true",
        help="Strip final softmax and use second-to-last layer logits",
    )
    args = parser.parse_args(argv)
    return Config(**vars(args))


def main(argv: Sequence[str] | None = None):
    cfg = parse_args(argv)
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
