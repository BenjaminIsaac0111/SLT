"""Tests for masked focal loss over labeled pixels only."""

import tensorflow as tf
from pytest import approx

from DeepLearning.losses.losses import focal_loss


def test_masked_normalization() -> None:
    """Loss only counts labeled pixels and normalizes by their count."""
    y_true = tf.constant(
        [[
            [[1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 0, 0]],
        ]],
        dtype=tf.float32,
    )  # shape [1,2,2,3]
    y_pred = tf.constant(
        [[
            [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]],
            [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]],
        ]],
        dtype=tf.float32,
    )
    loss = focal_loss(y_true, y_pred, gamma=0.0)
    assert loss.numpy() == approx(1.0986123, abs=1e-5)


def test_zero_labeled_returns_zero() -> None:
    """If no labeled pixels are present the loss is zero."""
    y_true = tf.zeros([1, 2, 2, 3], dtype=tf.float32)
    y_pred = tf.ones_like(y_true) / 3.0
    loss = focal_loss(y_true, y_pred, gamma=2.0)
    assert loss.numpy() == approx(0.0, abs=1e-6)
