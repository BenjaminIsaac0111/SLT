"""Validation utilities for model fine‑tuning.

The :class:`Validator` accumulates a confusion matrix and computes a suite of
metrics.  It is separated from the main training script to allow reuse and to
clarify the responsibilities of validation within the training pipeline.
"""

from __future__ import annotations

import tensorflow as tf


class Validator:
    """Stateful validator accumulating a confusion matrix and metrics."""

    def __init__(
        self,
        model: tf.keras.Model,
        num_classes: int,
        strategy: tf.distribute.Strategy,
        use_xla: bool = True,
        window_size: int = 3,
        label_policy: str = "window_majority",
        skip_empty: bool = True,
    ):
        self.use_xla = use_xla
        self.model = model
        if self.use_xla:
            self._infer = tf.function(
                lambda images: self.model(images, training=False),
                jit_compile=True,
            )
        else:
            self._infer = tf.function(
                lambda images: self.model(images, training=False),
                jit_compile=False,
            )

        self.window_size = window_size
        self.C = num_classes
        self.strategy = strategy
        self.label_policy = label_policy
        self.skip_empty = skip_empty
        with self.strategy.scope():
            self.cm_var = tf.Variable(
                tf.zeros((self.C, self.C), dtype=tf.int64),
                trainable=False,
                name="val_confusion_matrix",
            )
        self._build_batch_fn()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.cm_var.assign(tf.zeros_like(self.cm_var))

    # ------------------------------------------------------------------
    def _build_batch_fn(self) -> None:
        ws = self.window_size

        def extract_labels(one_hot):
            """Return integer labels and validity mask for a batch."""

            if self.label_policy == "center_pixel":
                h = tf.shape(one_hot)[1] // 2
                w = tf.shape(one_hot)[2] // 2
                center = one_hot[:, h : h + 1, w : w + 1, :]
                labels = tf.argmax(center, axis=-1, output_type=tf.int32)[:, 0, 0]
                if self.skip_empty:
                    valid = tf.reduce_max(center, axis=-1)[:, 0, 0] > 0.5
                else:
                    valid = tf.ones_like(labels, dtype=tf.bool)
                return labels, valid

            h = tf.shape(one_hot)[1] // 2
            w = tf.shape(one_hot)[2] // 2
            half = ws // 2
            win = one_hot[:, h - half : h + half + 1, w - half : w + half + 1, :]

            if self.label_policy == "window_mean_argmax":
                win_mean = tf.reduce_mean(win, axis=(1, 2))
                labels = tf.argmax(win_mean, axis=-1, output_type=tf.int32)
                if self.skip_empty:
                    valid = tf.reduce_max(win_mean, axis=-1) > 0.5
                else:
                    valid = tf.ones_like(labels, dtype=tf.bool)
                return labels, valid

            pix_labels = tf.argmax(win, axis=-1, output_type=tf.int32)
            fg_mask = tf.reduce_max(win, axis=-1) > 0.5
            pix_labels_flat = tf.reshape(pix_labels, (tf.shape(pix_labels)[0], -1))
            fg_flat = tf.reshape(fg_mask, (tf.shape(fg_mask)[0], -1))
            idx = tf.where(fg_flat)
            sample_ids = idx[:, 0]
            label_vals = tf.gather_nd(pix_labels_flat, idx)
            one_hot_counts = tf.one_hot(label_vals, depth=self.C, dtype=tf.int32)
            counts = tf.math.unsorted_segment_sum(one_hot_counts, sample_ids, tf.shape(pix_labels_flat)[0])
            had_fg = tf.reduce_any(fg_flat, axis=1)
            labels = tf.argmax(counts, axis=-1, output_type=tf.int32)
            return labels, had_fg if self.skip_empty else tf.ones_like(labels, tf.bool)

        def batch_confusion(y_true_onehot, images, class_prior, lambda_t):
            logits = self._infer(images)
            logits = tf.cast(logits, tf.float32)
            log_adj = lambda_t * tf.math.log(
                1.0 / tf.clip_by_value(class_prior, 1e-6, 1.0)
            )
            logits += log_adj
            h = tf.shape(logits)[1] // 2
            w = tf.shape(logits)[2] // 2
            half = ws // 2
            win_pred = logits[:, h - half : h + half + 1, w - half : w + half + 1, :]
            win_pred_mean = tf.reduce_mean(win_pred, axis=(1, 2))
            pred_labels = tf.argmax(win_pred_mean, axis=-1, output_type=tf.int32)

            true_labels, valid_mask = extract_labels(y_true_onehot)
            true_labels = tf.boolean_mask(true_labels, valid_mask)
            pred_labels = tf.boolean_mask(pred_labels, valid_mask)

            def empty_case():
                return tf.zeros((self.C, self.C), dtype=tf.int64)

            def non_empty_case():
                combined = true_labels * self.C + pred_labels
                flat_counts = tf.math.bincount(
                    combined,
                    minlength=self.C * self.C,
                    maxlength=self.C * self.C,
                    dtype=tf.int64,
                )
                return tf.reshape(flat_counts, (self.C, self.C))

            return tf.cond(tf.size(true_labels) > 0, non_empty_case, empty_case)

        @tf.function(jit_compile=False)
        def update_step(images, one_hot, class_prior, lambda_t):
            batch_cm = batch_confusion(one_hot, images, class_prior, lambda_t)
            self.cm_var.assign_add(batch_cm)

        self._update_step = update_step

    # ------------------------------------------------------------------
    def update(
        self,
        images,
        one_hot,
        class_prior: tf.Tensor | None = None,
        lambda_t: float = 0.0,
    ) -> None:
        """Accumulate the confusion matrix for a batch.

        Parameters
        ----------
        images: Tensor of shape `[B, H, W, C]` containing input images.
        one_hot: Tensor of shape `[B, H, W, num_classes]` with one‑hot labels.
        class_prior: Optional prior probabilities for logit adjustment.
        lambda_t: Scaling factor for logit adjustment.
        """

        if class_prior is None:
            class_prior = tf.fill([self.C], 1.0 / self.C)
        class_prior = tf.convert_to_tensor(class_prior, dtype=tf.float32)
        lambda_t = tf.convert_to_tensor(lambda_t, dtype=tf.float32)

        if isinstance(self.strategy, tf.distribute.Strategy) and not isinstance(
            self.strategy, tf.distribute.OneDeviceStrategy
        ):
            def replica_fn(imgs, y, prior, lamb):
                self._update_step(imgs, y, prior, lamb)

            self.strategy.run(
                replica_fn, args=(images, one_hot, class_prior, lambda_t)
            )
        else:
            self._update_step(images, one_hot, class_prior, lambda_t)

    # ------------------------------------------------------------------
    def result(self) -> dict[str, tf.Tensor]:
        """Return the collected metrics including the confusion matrix."""

        cm = tf.cast(self.cm_var.read_value(), tf.float32)
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


__all__ = ["Validator"]

