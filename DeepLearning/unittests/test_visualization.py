"""Tests for visualization helpers."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from DeepLearning.training.visualization import confusion_matrix_to_image

CLASS_COMPONENTS = [
    "Non-Informative",  # 0
    "Tumour",  # 1
    "Stroma",  # 2
    "Necrosis",  # 3
    "Vessel",  # 4
    "Inflammation",  # 5
    "Tumour-Lumen",  # 6
    "Mucin",  # 7
    "Muscle"  # 8
]


def test_confusion_matrix_to_image_shape() -> None:
    """Generated confusion matrix images have expected shape and color."""

    cm_percent = tf.constant([[50.0, 50.0], [25.0, 75.0]], dtype=tf.float32)
    image = confusion_matrix_to_image(cm_percent)

    assert image.shape[0] == 1
    assert image.shape[-1] == 4
    img_np = image[0].numpy()
    # Ensure at least one pixel has differing red and green channels (not grayscale)
    assert not np.allclose(img_np[..., 0], img_np[..., 1])


def test_confusion_matrix_to_image_shape_multiclass() -> None:
    """
    Confusion-matrix images for the nine colorectal tissue components
    (Non-Informative … Muscle) have the expected tensor shape
    and are rendered in colour (i.e. not grayscale).

    The values are synthetic but span the full 0–100 % range so that the
    Viridis colour-map definitely produces chromatic variation.
    """

    num_classes = len(CLASS_COMPONENTS)  # 9
    # Build a 9×9 matrix whose entries sweep linearly from 0 to 100 %.
    cm_values = tf.reshape(
        tf.linspace(0.0, 100.0, num_classes ** 2),
        (num_classes, num_classes),
    )
    class_names = [CLASS_COMPONENTS[i] for i in range(num_classes)]

    image = confusion_matrix_to_image(cm_values, class_names=class_names)

    # -----------
    # Assertions
    # -----------
    # TensorBoard images come out with shape (1, H, W, 4):
    assert image.shape[0] == 1, "Batch dimension should be 1."
    assert image.shape[-1] == 4, "Last dimension should contain 4 RGBA channels."

    # Grab the single image as an ndarray and confirm it is not grayscale
    img_np = image[0].numpy()
    assert not np.allclose(
        img_np[..., 0], img_np[..., 1]
    ), "Red and green channels are identical ⇒ image is grayscale."


def test_confusion_matrix_to_image_moderate_performance() -> None:
    """
    Simulate a “moderate” model: 60% of predictions are correct (diagonal),
    the remaining 40% are equally confused among the other classes.
    Verify that (1) the output image has shape (1, H, W, 4)
    and (2) the average diagonal‐cell intensity is higher than an off‐diagonal.
    """
    # 1) Build the synthetic confusion matrix
    num_classes = len(CLASS_COMPONENTS)  # 9
    off_val = 40.0 / (num_classes - 1)
    cm = np.full((num_classes, num_classes), off_val, dtype=np.float32)
    np.fill_diagonal(cm, 60.0)

    # 2) Convert to tensor and render
    cm_tensor = tf.constant(cm, dtype=tf.float32)
    class_names = [CLASS_COMPONENTS[i] for i in range(num_classes)]
    image = confusion_matrix_to_image(cm_tensor, class_names=class_names)

    # 3) Basic shape assertions
    assert image.shape[0] == 1, "Should have batch dimension 1"
    assert image.shape[-1] == 4, "Should have 4 RGBA channels"

    # 4) Extract RGB data and compute per-cell sampling
    img = image[0].numpy()[..., :3]  # drop alpha
    H, W, _ = img.shape
    cell_h = H // num_classes
    cell_w = W // num_classes

    # sample the centre pixel of each diagonal cell
    diag_vals = []
    for i in range(num_classes):
        y = i * cell_h + cell_h // 2
        x = i * cell_w + cell_w // 2
        diag_vals.append(img[y, x].mean())

    # sample one off-diagonal cell (e.g. (0,1))
    y_off = 0 * cell_h + cell_h // 2
    x_off = 1 * cell_w + cell_w // 2
    off_val = img[y_off, x_off].mean()

    # 5) Check that diagonals are “brighter” on average than off-diagonal
    assert np.mean(diag_vals) > off_val, (
        f"Expected mean(diag)={np.mean(diag_vals):.3f} > off-diag={off_val:.3f}"
    )


def test_confusion_matrix_to_image_moderate_with_noise() -> None:
    """
    Simulate a “moderate” model with Gaussian noise:
    roughly 60% correct on the diagonal, noise σ=0.05 off-diagonal,
    and each row re-normalised to 100%. Verify that
    1) the image tensor has shape (1, H, W, 4), and
    2) the mean diagonal‐cell intensity exceeds the mean off‐diagonal intensity.
    """
    # 1) Build the synthetic confusion matrix
    num_classes = len(CLASS_COMPONENTS)  # 9
    base_diag = 0.6
    off_val = (1.0 - base_diag) / (num_classes - 1)

    cm = np.full((num_classes, num_classes), off_val, dtype=float)
    np.fill_diagonal(cm, base_diag)

    # reproducible noise
    rng = np.random.RandomState(1234)
    noise = rng.normal(loc=0.0, scale=0.05, size=cm.shape)
    cm_noisy = np.clip(cm + noise, 0.0, None)

    # row-normalise and convert to percent
    cm_noisy /= cm_noisy.sum(axis=1, keepdims=True)
    cm_percent = cm_noisy * 100.0

    # 2) Render to image
    cm_tensor = tf.constant(cm_percent, dtype=tf.float32)
    class_names = [CLASS_COMPONENTS[i] for i in range(num_classes)]
    image = confusion_matrix_to_image(cm_tensor, class_names=class_names)

    # 3) Shape assertions
    assert image.shape[0] == 1, f"Expected batch dim=1, got {image.shape[0]}"
    assert image.shape[-1] == 4, f"Expected RGBA channels, got {image.shape[-1]}"

    # 4) Sample centre pixels of each cell for intensity
    img = image[0].numpy()[..., :3]  # drop alpha channel
    H, W, _ = img.shape
    cell_h = H // num_classes
    cell_w = W // num_classes

    diag_vals = []
    off_vals = []
    for i in range(num_classes):
        for j in range(num_classes):
            y = i * cell_h + cell_h // 2
            x = j * cell_w + cell_w // 2
            intensity = img[y, x].mean()
            if i == j:
                diag_vals.append(intensity)
            else:
                off_vals.append(intensity)

    # 5) Verify diagonal is on average more intense
    mean_diag = np.mean(diag_vals)
    mean_off = np.mean(off_vals)
    assert mean_diag > mean_off, (
        f"Mean(diagonal)={mean_diag:.3f} should exceed Mean(off-diagonal)={mean_off:.3f}"
    )
