"""Tests for visualization helpers."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from DeepLearning.training.visualization import confusion_matrix_to_image


def test_confusion_matrix_to_image_shape() -> None:
    """Generated confusion matrix images have expected shape and color."""

    cm_percent = tf.constant([[50.0, 50.0], [25.0, 75.0]], dtype=tf.float32)
    image = confusion_matrix_to_image(cm_percent)

    assert image.shape[0] == 1
    assert image.shape[-1] == 4
    img_np = image[0].numpy()
    # Ensure at least one pixel has differing red and green channels (not grayscale)
    assert not np.allclose(img_np[..., 0], img_np[..., 1])

