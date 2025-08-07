"""Debugging visualization helpers for training.

This module hosts optional utilities to sample batches and visualize model
predictions. They depend on a trainer-like object but are kept separate from the
core training loop to avoid polluting the main Trainer class.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import io

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def sample_batch(trainer) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return a single batch ``(x, y)`` from the trainer's datasets.

    The ``trainer`` is expected to expose ``new_iter``, ``old_iter``,
    ``p_new(step)`` and ``global_step`` attributes.
    """
    step = int(trainer.global_step.numpy())
    if getattr(trainer, "new_iter", None) and tf.random.uniform(()) < trainer.p_new(step):
        batch = next(trainer.new_iter)
    else:
        batch = next(trainer.old_iter)
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        _, x, y = batch
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
    else:
        raise ValueError("Unexpected batch structure fetched from dataset.")
    return x, y


def visualize_sample(
    trainer,
    save_path: str | None = None,
    class_colors: Sequence | None = None,
    alpha: float = 0.5,
    log_to_tb: bool = False,
) -> None:
    """Visualize an input, ground truth and prediction from a trainer.

    Parameters
    ----------
    trainer:
        Trainer-like object exposing ``model``, ``cfg.num_classes``, ``tb_writer``
        and dataset iterators used by :func:`sample_batch`.
    save_path:
        Optional filesystem path to save the PNG figure.
    class_colors:
        Optional sequence of RGB triples in ``[0, 1]``.
    alpha:
        Blend factor for the prediction overlay.
    log_to_tb:
        If ``True`` log the composite panel to TensorBoard using the trainer's
        ``tb_writer``.
    """
    x, y = sample_batch(trainer)
    x0 = x[0:1]
    y0 = y[0]

    # Normalize ground truth shape to (H, W)
    if y0.ndim == 3:
        if y0.shape[-1] == trainer.cfg.num_classes:  # one-hot
            y0 = tf.argmax(y0, axis=-1)
        elif y0.shape[-1] == 1:  # redundant channel
            y0 = tf.squeeze(y0, axis=-1)
    y0 = tf.cast(y0, tf.int32)

    logits = trainer.model(x0, training=False)
    if logits.shape[-1] != trainer.cfg.num_classes:
        raise ValueError(
            f"Model output last dim {logits.shape[-1]} != num_classes {trainer.cfg.num_classes}"
        )
    pred = tf.argmax(logits[0], axis=-1)
    pred = tf.cast(pred, tf.int32)

    num_classes = trainer.cfg.num_classes
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
    panels = [
        ("Input", x_img),
        ("Ground Truth", gt_color),
        ("Prediction", pred_color),
        ("Overlay", overlay),
    ]
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
        with trainer.tb_writer.as_default():
            tf.summary.image("debug/sample_panel", image, step=int(trainer.global_step.numpy()))
        trainer.tb_writer.flush()
        tf.print("[DEBUG] Logged sample panel to TensorBoard.")
    if not save_path:
        plt.show()
        plt.close(fig)
        tf.print(f"[DEBUG] Saved visualization to {save_path}")
    else:
        plt.show()


def confusion_matrix_to_image(cm_percent: tf.Tensor, class_names: Sequence[str] | None = None) -> tf.Tensor:
    """Convert a percentage confusion matrix to a TensorBoard image.

    Parameters
    ----------
    cm_percent:
        Matrix of shape ``(C, C)`` containing percentage values.
    class_names:
        Optional sequence of class labels used for axis tick labels.

    Returns
    -------
    tf.Tensor
        Image tensor of shape ``(1, H, W, 4)`` suitable for ``tf.summary.image``.
    """

    cm_np = cm_percent.numpy() if isinstance(cm_percent, tf.Tensor) else np.asarray(cm_percent)
    num_classes = cm_np.shape[0]

    fig, ax = plt.subplots(figsize=(3 + 0.5 * num_classes, 3 + 0.5 * num_classes))
    im = ax.imshow(cm_np, vmin=0, vmax=100, cmap="viridis")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    labels = class_names if class_names is not None else range(num_classes)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    fmt = ".1f"
    thresh = cm_np.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(cm_np[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_np[i, j] > thresh else "black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percent")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    return tf.expand_dims(image, 0)


__all__ = ["sample_batch", "visualize_sample", "confusion_matrix_to_image"]
