import argparse
import logging
import math
import os
import sys

import h5py
import numpy as np
import tensorflow as tf
import yaml
from keras import Model
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model
from tqdm import tqdm

from Dataloader.dataloader import get_dataset_v2
from Model.custom_layers import GroupNormalization, SpatialConcreteDropout, DropoutAttentionBlock

# Suppress TensorFlow's INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Further suppress TensorFlow's Python logger
tf.get_logger().setLevel('ERROR')

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

model_prefix = 'dropout'


def setup_logging(log_level=logging.DEBUG, log_file=None):
    """
    Sets up the logging configuration.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


@tf.function
def mc_infer(x, n_iter=20, model=None, num_classes=10, logger=None):
    """
    Perform MC Dropout inference to obtain various uncertainty measures.
    Returns uncertainties and predictions.
    """
    if logger:
        logger.debug(f"Starting MC inference with {n_iter} iterations.")

    probs = []
    logits = []

    for i in range(n_iter):
        preds = predict_logits(x, model)  # Get logits from the model
        softmax_preds = tf.nn.softmax(preds, axis=-1)  # Apply softmax to get probabilities
        probs.append(softmax_preds)
        logits.append(preds)
        if logger and (i + 1) % 5 == 0:
            logger.debug(f"MC iteration {i + 1}/{n_iter} completed.")

    # Stack logits and probs to create tensors
    probs = tf.stack(probs, axis=0)  # Shape: [n_iter, batch_size, H, W, num_classes]
    logits = tf.stack(logits, axis=0)  # Shape: [n_iter, batch_size, H, W, num_classes]

    # Calculate mean probabilities and mean logits over MC iterations
    mean_probs = tf.reduce_mean(probs, axis=0)  # Shape: [batch_size, H, W, num_classes]
    mean_logits = tf.reduce_mean(logits, axis=0)  # Shape: [batch_size, H, W, num_classes]

    # Calculate Predictive Entropy
    entropy_mean = -tf.reduce_sum(mean_probs * tf.math.log(mean_probs + 1e-8), axis=-1)  # [batch_size, H, W]

    # Calculate Expected Entropy
    expected_entropies = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)  # [n_iter, batch_size, H, W]
    expected_entropy = tf.reduce_mean(expected_entropies, axis=0)  # [batch_size, H, W]

    # Calculate BALD (Mutual Information)
    mutual_information = entropy_mean - expected_entropy  # [batch_size, H, W]

    # Calculate Variance of Probabilities
    variance_probs = tf.math.reduce_variance(probs, axis=0)  # [batch_size, H, W, num_classes]
    variance_uncertainty = tf.reduce_sum(variance_probs, axis=-1)  # [batch_size, H, W]

    # Normalize entropy to [0, 1]
    max_entropy = tf.cast(tf.math.log(float(num_classes)), tf.float16)
    normalized_entropy = entropy_mean / max_entropy  # Normalize to [0, 1]

    return {
        'entropy': normalized_entropy,
        'variance': variance_uncertainty,
        'bald': mutual_information,
        'mean_probs': mean_probs,
        'mean_logits': mean_logits
    }


@tf.function(jit_compile=True)
def predict_logits(x, model):
    return model(x, training=True)


def main(config, logger):
    """
    Main function to run MC Inference, adjust uncertainties, and store data.
    This version uses dynamically resizable HDF5 datasets.
    """
    logger.info("Starting main processing function.")

    # Extract input dimensions from INPUT_SIZE
    input_size = config['INPUT_SIZE']
    if not isinstance(input_size, list) or len(input_size) != 3:
        logger.error("INPUT_SIZE must be a list of three integers: [HEIGHT, WIDTH, CHANNELS]")
        raise ValueError("INPUT_SIZE must be a list of three integers: [HEIGHT, WIDTH, CHANNELS]")
    INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS = input_size
    logger.debug(f"Input dimensions: Height={INPUT_HEIGHT}, Width={INPUT_WIDTH}, Channels={INPUT_CHANNELS}")

    input_shape = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
    num_classes = config['OUT_CHANNELS']
    logger.debug(f"Number of classes: {num_classes}")

    # Extract classwise uncertainty weights (if used)
    classwise_uncertainty_weights = config.get('CLASSWISE_UNCERTAINTY_WEIGHTS', None)
    if classwise_uncertainty_weights is None:
        logger.error("CLASSWISE_UNCERTAINTY_WEIGHTS not found in config.")
        raise KeyError("CLASSWISE_UNCERTAINTY_WEIGHTS not found in config.")
    classwise_uncertainty_weights = np.array(classwise_uncertainty_weights, dtype=np.float16)
    if len(classwise_uncertainty_weights) != num_classes:
        logger.error("Length of CLASSWISE_UNCERTAINTY_WEIGHTS must be equal to number of classes.")
        raise ValueError("Length of CLASSWISE_UNCERTAINTY_WEIGHTS must be equal to number of classes.")

    # Load the model
    model_path = os.path.join(config['MODEL_DIR'], f"best_{config['MODEL_NAME']}")
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

    # Modify the model to output logits (second-to-last layer)
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    logger.debug("Modified model to output logits from the second-to-last layer.")

    # Load the dataset
    logger.info("Loading dataset.")
    ds = get_dataset_v2(
        data_dir=config['DATA_DIR'],
        filelists=config['TRAINING_LIST'],
        repeat=False,
        shuffle=False,
        batch_size=config['BATCH_SIZE'],
        shuffle_buffer_size=config['SHUFFLE_BUFFER_SIZE'],
        out_channels=config['OUT_CHANNELS']
    )
    logger.info("Dataset loaded successfully.")

    # Check for output directory in config or fallback to model directory
    output_dir = config.get('OUTPUT_DIR', config['MODEL_DIR'])
    if not os.path.exists(output_dir):
        logger.info(f"Output directory {output_dir} does not exist. Creating it.")
        os.makedirs(output_dir)

    # Construct the HDF5 output file path
    output_file = os.path.join(
        output_dir,
        f"{model_prefix}_{os.path.splitext(config['MODEL_NAME'])[0]}_inference_output.h5"
    )
    logger.info(f"Output will be stored in {output_file}")

    # Try to read total_samples if provided
    try:
        N_SAMPLES = int(config.get('N_SAMPLES', -1))
    except ValueError:
        logger.error("N_SAMPLES must be an integer.")
        raise ValueError("N_SAMPLES must be an integer.")

    if N_SAMPLES == -1:
        # If not specified, we might not know total_samples upfront; if you want total_samples,
        # you must read it from the TRAINING_LIST.
        training_list_path = config['TRAINING_LIST']
        if not os.path.isfile(training_list_path):
            logger.error(f"TRAINING_LIST file not found at {training_list_path}")
            raise FileNotFoundError(f"TRAINING_LIST file not found at {training_list_path}")

        with open(training_list_path, 'r') as f:
            total_samples = sum(1 for line in f if line.strip())
        logger.info(f"Processing all available samples: {total_samples}")
    else:
        total_samples = N_SAMPLES
        logger.info(f"Total samples to process: {total_samples}")

    chunk_size = config.get('CHUNK_SIZE', 1)
    logger.debug(f"Chunk size for HDF5 datasets: {chunk_size}")

    batch_size = config['BATCH_SIZE']

    # Determine if we can resume from a checkpoint
    resume_index = 0
    hdf5_mode = 'a' if os.path.exists(output_file) else 'w'  # 'a' for append mode

    logger.info("Starting processing to compute uncertainties and store data...")

    try:
        with h5py.File(output_file, hdf5_mode) as hdf5_file:
            logger.info(f"HDF5 file opened in mode '{hdf5_mode}' for storing results.")

            # If datasets don't exist, create them as resizable
            if 'rgb_images' not in hdf5_file:
                maxshape_rgb = (None,) + input_shape
                maxshape_logits = (None,) + input_shape[:2] + (num_classes,)
                maxshape_2d = (None,) + input_shape[:2]

                rgb_image_dataset = hdf5_file.create_dataset(
                    'rgb_images',
                    shape=(0,) + input_shape,
                    maxshape=maxshape_rgb,
                    dtype='uint8',
                    compression='gzip',
                    compression_opts=4,
                    chunks=(chunk_size,) + input_shape
                )
                logits_dataset = hdf5_file.create_dataset(
                    'logits',
                    shape=(0,) + input_shape[:2] + (num_classes,),
                    maxshape=maxshape_logits,
                    dtype='float32',
                    compression='gzip',
                    compression_opts=4,
                    chunks=(chunk_size,) + input_shape[:2] + (num_classes,)
                )
                variance_dataset = hdf5_file.create_dataset(
                    'variance',
                    shape=(0,) + input_shape[:2],
                    maxshape=maxshape_2d,
                    dtype='float32',
                    compression='gzip',
                    compression_opts=4,
                    chunks=(chunk_size,) + input_shape[:2]
                )
                bald_dataset = hdf5_file.create_dataset(
                    'bald',
                    shape=(0,) + input_shape[:2],
                    maxshape=maxshape_2d,
                    dtype='float32',
                    compression='gzip',
                    compression_opts=4,
                    chunks=(chunk_size,) + input_shape[:2]
                )
                filename_dataset = hdf5_file.create_dataset(
                    'filenames',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    compression='gzip',
                    compression_opts=4,
                    chunks=(chunk_size,)
                )
            else:
                # Datasets already exist, so just open them
                rgb_image_dataset = hdf5_file['rgb_images']
                logits_dataset = hdf5_file['logits']
                variance_dataset = hdf5_file['variance']
                bald_dataset = hdf5_file['bald']
                filename_dataset = hdf5_file['filenames']

            # Check for checkpoint
            if 'last_written_index' in hdf5_file.attrs:
                resume_index = hdf5_file.attrs['last_written_index']
                logger.info(f"Resuming from index {resume_index}")
            else:
                hdf5_file.attrs['last_written_index'] = 0

            # Skip samples in the dataset if resuming
            if resume_index > 0:
                ds = ds.skip(resume_index)

            # If total_samples is known, limit dataset to remaining samples
            if total_samples > 0:
                samples_remaining = total_samples - resume_index
                if samples_remaining <= 0:
                    logger.info("All samples have already been processed. Exiting.")
                    return
                batches_remaining = math.ceil(samples_remaining / batch_size)
                ds = ds.take(batches_remaining)

                total_to_process = samples_remaining
            else:
                # If total_samples not known, you can rely solely on dataset exhaustion
                total_to_process = None

            index = resume_index
            with tqdm(total=total_to_process, desc="Processing", unit="samples",
                      disable=(total_to_process is None)) as pbar:
                for batch_idx, (filename_batch, x_batch, y_batch) in enumerate(ds):
                    current_batch_size = x_batch.shape[0]

                    # Compute uncertainties
                    uncertainties = mc_infer(
                        x_batch, n_iter=config['MC_N_ITER'], model=model, num_classes=num_classes,
                        logger=logger
                    )

                    # Extract uncertainties
                    entropy_np = uncertainties['entropy'].numpy()
                    variance_np = uncertainties['variance'].numpy()
                    bald_np = uncertainties['bald'].numpy()
                    mean_logits_np = uncertainties['mean_logits'].numpy()

                    # Adjust uncertainties
                    adjusted_variance = variance_np * (1 - entropy_np)
                    adjusted_bald = bald_np * (1 - entropy_np)

                    x_batch_np = x_batch.numpy()
                    filename_batch_np = filename_batch.numpy()

                    rgb_images_uint8 = (x_batch_np * 255).astype('uint8')

                    # If total_samples is known, ensure we don't exceed it
                    actual_batch_size = current_batch_size
                    if total_samples > 0:
                        actual_batch_size = min(current_batch_size, total_samples - index)
                        if actual_batch_size <= 0:
                            logger.warning("No samples left to write, stopping early.")
                            break

                    # Resize datasets to hold the new samples
                    new_size = index + actual_batch_size

                    rgb_image_dataset.resize((new_size,) + input_shape)
                    logits_dataset.resize((new_size,) + input_shape[:2] + (num_classes,))
                    variance_dataset.resize((new_size,) + input_shape[:2])
                    bald_dataset.resize((new_size,) + input_shape[:2])
                    filename_dataset.resize((new_size,))

                    # Write data
                    rgb_image_dataset[index:new_size] = rgb_images_uint8[:actual_batch_size]
                    logits_dataset[index:new_size] = mean_logits_np[:actual_batch_size].astype('float16')
                    variance_dataset[index:new_size] = adjusted_variance[:actual_batch_size].astype('float16')
                    bald_dataset[index:new_size] = adjusted_bald[:actual_batch_size].astype('float16')
                    filename_dataset[index:new_size] = [
                        fname.decode('utf-8') for fname in filename_batch_np[:actual_batch_size]
                    ]

                    index += actual_batch_size
                    hdf5_file.attrs['last_written_index'] = index
                    hdf5_file.flush()

                    if total_to_process is not None:
                        pbar.update(actual_batch_size)

                    # If total_samples is known and we've hit the limit, stop
                    if total_samples > 0 and index >= total_samples:
                        logger.info("Reached total_samples limit.")
                        break

            # Flush data one last time
            hdf5_file.flush()
            logger.info("All data written to HDF5 file successfully.")

    except Exception as e:
        logger.exception("An error occurred while writing to the HDF5 file.")
        raise e

    logger.info(f"Processing completed. Data stored in {output_file}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Monte Carlo Inference with Uncertainty Measures')
    parser.add_argument('-c', '--config_path', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument('--log_file', type=str, default=None, help='Path to a file to store logs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory where the output HDF5 file will be saved (overrides default in config).')
    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger = setup_logging(log_level=log_level, log_file=args.log_file)
    logger.info("Logging is configured.")

    try:
        # Load the YAML config file
        logger.info(f"Loading configuration from {args.config_path}")
        with open(args.config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        logger.info("Configuration loaded successfully.")

        # Override output directory if specified via CLI
        if args.output_dir:
            config['OUTPUT_DIR'] = args.output_dir

        # Ensure required keys are present in the config
        required_keys = ['MODEL_DIR', 'MODEL_NAME', 'DATA_DIR', 'TRAINING_LIST', 'BATCH_SIZE',
                         'SHUFFLE_BUFFER_SIZE', 'OUT_CHANNELS', 'INPUT_SIZE', 'MC_N_ITER']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            logger.error(f"Missing required config keys: {missing_keys}")
            raise KeyError(f"Missing required config keys: {missing_keys}")
        logger.debug("All required configuration keys are present.")

        # Call the main function with the config and logger
        main(config, logger)

    except Exception as e:
        logger.critical("An unrecoverable error occurred. Exiting the program.")
        sys.exit(1)
