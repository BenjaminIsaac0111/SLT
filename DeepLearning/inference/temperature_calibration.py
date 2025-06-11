import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from Dataloader.dataloader import get_dataset_v2
from Model.custom_layers import GroupNormalization, SpatialConcreteDropout, DropoutAttentionBlock
from keras import Model
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model

# Suppress TensorFlow's INFO and WARNING logs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')


def setup_logging(log_level=logging.DEBUG, log_file=None):
    """
    Sets up logging configuration.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def center_pixel(tensor):
    """
    Extracts the center pixel from each image in the tensor.
    Supports tensors of shape [batch, H, W] or [batch, H, W, ...].
    Returns a tensor of shape [batch, ...] containing the pixel at the center of each image.
    """
    shape = tf.shape(tensor)
    H = shape[1]
    W = shape[2]
    return tensor[:, H // 2, W // 2, ...]


@tf.function
def predict_logits_det(x, model, temperature=1.0):
    """
    Deterministic prediction (with dropout off) to compute temperature-scaled logits.
    """
    logits = model(x, training=False)
    return logits / tf.cast(temperature, tf.float16)


def precompute_calibration_data(test_ds, model, num_batches=6600):
    """
    Precomputes logits and ground-truth labels at the center pixel from the test dataset.

    Returns:
      - logits_all: Tensor of shape [N, num_classes] with the logits.
      - labels_all: Tensor of shape [N] with the ground-truth labels.
    """
    logits_list = []
    labels_list = []

    for batch_idx, (filename_batch, x_batch, y_batch) in enumerate(test_ds):
        # Compute logits with temperature = 1.0
        logits = predict_logits_det(x_batch, model, temperature=1.0)
        center_logits = center_pixel(logits)

        # Extract ground truth at the center
        center_labels = center_pixel(y_batch)
        if center_labels.shape.rank == 4 and center_labels.shape[-1] == 1:
            center_labels = tf.squeeze(center_labels, axis=-1)

        logits_list.append(center_logits)
        labels_list.append(center_labels)

        if batch_idx >= num_batches - 1:
            break

    logits_all = tf.concat(logits_list, axis=0)
    labels_all = tf.concat(labels_list, axis=0)
    return logits_all, tf.cast(labels_all, tf.int64)


def calibrate_temperature_by_ece(logits_all, labels_all, temperature_candidates=None, n_bins=10):
    """
    Performs temperature calibration using the precomputed calibration data by minimizing the Expected Calibration
    Error (ECE).

    Returns:
      - best_temp: The temperature that minimizes the ECE.
    """
    if temperature_candidates is None:
        temperature_candidates = np.linspace(0.5, 2.0, 50)

    best_temp = 1.0
    best_ece = float('inf')

    # Convert one-hot labels to integer labels if necessary.
    # Check if labels_all is one-hot encoded by inspecting the last dimension.
    if labels_all.shape.ndims > 1 and labels_all.shape[-1] > 1:
        labels_int = tf.argmax(labels_all, axis=-1)
    else:
        labels_int = tf.cast(labels_all, tf.int64)

    for T in temperature_candidates:
        # Apply temperature scaling to the precomputed logits.
        scaled_logits = logits_all / tf.cast(T, tf.float16)
        probs = tf.nn.softmax(scaled_logits, axis=-1)
        pred_labels = tf.argmax(probs, axis=-1)
        confidences = tf.reduce_max(probs, axis=-1).numpy().flatten()
        # Use the converted integer labels in the correctness computation.
        correctness = (pred_labels == labels_int).numpy().flatten()

        # Compute the Expected Calibration Error using the computed confidences and correctness.
        _, _, _, ece = compute_reliability_curve(confidences, correctness, n_bins=n_bins)
        if ece < best_ece:
            best_ece = ece
            best_temp = T

    return best_temp


def compute_metrics_from_data(logits_all, labels_all, temperature):
    """
    Computes confidences and correctness indicators from precomputed calibration data after applying
    temperature scaling.

    Returns:
      - confidences: Flattened array of the maximum probabilities.
      - correctness: Flattened array of 1s (correct) and 0s (incorrect).
    """
    scaled_logits = tf.cast(logits_all, tf.float32) / temperature
    probs = tf.nn.softmax(scaled_logits, axis=-1)
    pred_labels = tf.argmax(probs, axis=-1)
    confidences = tf.reduce_max(probs, axis=-1)
    correctness = tf.cast(tf.equal(pred_labels, tf.argmax(labels_all, axis=-1)), tf.float32)

    return confidences.numpy().flatten(), correctness.numpy().flatten()


def compute_reliability_curve(confidences, correctness, n_bins=10):
    """
    Computes calibration statistics by binning predictions based on confidence.

    Returns:
      - bin_centers: Center value for each bin.
      - bin_accs: Empirical accuracy in each bin.
      - avg_conf: Average confidence in each bin.
      - ece: Expected calibration error.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_accs = np.zeros(n_bins)
    avg_conf = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if np.any(mask):
            bin_accs[i] = np.mean(correctness[mask])
            avg_conf[i] = np.mean(confidences[mask])
            bin_counts[i] = np.sum(mask)
        else:
            bin_accs[i] = 0.0
            avg_conf[i] = 0.0
    total = np.sum(bin_counts)
    ece = np.sum(bin_counts * np.abs(avg_conf - bin_accs)) / total if total > 0 else 0.0
    return bin_centers, bin_accs, avg_conf, ece


def plot_calibration(bin_centers, bin_accs, avg_conf, ece, title, filename=None):
    """
    Plots a reliability diagram (calibration curve).
    """
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.plot(bin_centers, bin_accs, marker='o', linestyle='-', label='Empirical Accuracy')
    plt.plot(bin_centers, avg_conf, marker='x', linestyle='--', label='Average Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'{title}\nECE: {ece:.4f}')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def main(config, logger):
    """
    Main function for temperature calibration.

    Loads the model and test dataset, precomputes calibration data from the center pixel,
    performs temperature calibration, and computes calibration plots for both the uncalibrated
    and calibrated predictions.
    """
    # Load model.
    model_path = os.path.join(config['MODEL_DIR'], 'ckpt_' + config['MODEL_NAME'])
    logger.info(f"Loading model from {model_path}")
    try:
        model = load_model(
            model_path,
            custom_objects={
                'DropoutAttentionBlock': DropoutAttentionBlock,
                'GroupNormalization': GroupNormalization,
                'SpatialConcreteDropout': SpatialConcreteDropout,
            },
            compile=False
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load the model.")
        raise e

    # Modify the model to output logits from the second-to-last layer.
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    logger.debug("Modified model to output logits from the second-to-last layer.")

    # Load the test dataset.
    if 'TESTING_LIST' not in config:
        logger.error("TESTING_LIST not found in config. Cannot perform calibration without a test set.")
        sys.exit(1)

    test_ds = get_dataset_v2(
        data_dir=config['DATA_DIR'],
        filelists=config['TESTING_LIST'],
        repeat=False,
        shuffle=False,
        batch_size=config['BATCH_SIZE'],
        shuffle_buffer_size=config['SHUFFLE_BUFFER_SIZE'],
        out_channels=config['OUT_CHANNELS']
    )
    logger.info("Test dataset loaded successfully for calibration.")

    # Precompute calibration data (logits and ground-truth labels for the center pixel).
    logger.info("Precomputing calibration data from test dataset...")
    logits_all, labels_all = precompute_calibration_data(
        test_ds, model, num_batches=config.get('NUM_CALIBRATION_BATCHES', 6600)
    )

    # Temperature calibration using the precomputed data.
    logger.info("Performing temperature calibration...")
    optimal_temperature = calibrate_temperature_by_ece(logits_all, labels_all)
    logger.info(f"Optimal temperature obtained from calibration: {optimal_temperature}")

    # Optionally write the calibrated temperature to a file.
    if 'CALIBRATION_OUTPUT' in config:
        output_path = config['CALIBRATION_OUTPUT']
        with open(output_path, 'w') as f:
            yaml.dump({"optimal_temperature": optimal_temperature}, f)
        logger.info(f"Calibrated temperature saved to {output_path}")

    print(f"Calibrated Temperature: {optimal_temperature}")

    # --- Calibration Plots using Precomputed Data ---
    num_bins = 10

    # Compute metrics for uncalibrated predictions (T = 1.0)
    confidences_uncal, correctness_uncal = compute_metrics_from_data(logits_all, labels_all, temperature=1.0)
    bin_centers_uncal, bin_accs_uncal, avg_conf_uncal, ece_uncal = compute_reliability_curve(
        confidences_uncal, correctness_uncal, n_bins=num_bins)
    plot_calibration(bin_centers_uncal, bin_accs_uncal, avg_conf_uncal, ece_uncal,
                     title='Calibration (T = 1.0 - Uncalibrated)',
                     filename=config.get('CALIBRATION_PLOT_UNCALIB', None))

    # Compute metrics for calibrated predictions (T = optimal_temperature)
    confidences_cal, correctness_cal = compute_metrics_from_data(logits_all, labels_all,
                                                                 temperature=optimal_temperature)
    bin_centers_cal, bin_accs_cal, avg_conf_cal, ece_cal = compute_reliability_curve(
        confidences_cal, correctness_cal, n_bins=num_bins)
    plot_calibration(bin_centers_cal, bin_accs_cal, avg_conf_cal, ece_cal,
                     title=f'Calibration (T = {optimal_temperature:.2f} - Calibrated)',
                     filename=config.get('CALIBRATION_PLOT_CALIB', None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Temperature Calibration Script Using Precomputed Center-Pixel Logits'
    )
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help='Path to the configuration YAML file')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to the log file')
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(log_level=log_level, log_file=args.log_file)
    logger.info("Logging is configured.")

    logger.info(f"Loading configuration from {args.config_path}")
    with open(args.config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    logger.info("Configuration loaded successfully.")

    required_keys = ['MODEL_DIR', 'MODEL_NAME', 'DATA_DIR', 'TESTING_LIST',
                     'BATCH_SIZE', 'SHUFFLE_BUFFER_SIZE', 'OUT_CHANNELS']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.error(f"Missing required config keys for calibration: {missing_keys}")
        raise KeyError(f"Missing required config keys for calibration: {missing_keys}")

    main(config, logger)
