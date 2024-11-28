import os
import logging
import yaml
import h5py
import tensorflow as tf
import numpy as np
from keras import Model
from tensorflow.keras.models import load_model
from tqdm import tqdm
from Dataloader.dataloader import get_dataset_v2
from Model.custom_layers import GroupNormalization, SpatialConcreteDropout, DropoutAttentionBlock
from tensorflow.keras import mixed_precision
import argparse
import sys
import math

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

    # Normalize uncertainties
    # For entropy-based measures, the maximum entropy occurs when the distribution is uniform
    max_entropy = tf.cast(tf.math.log(float(num_classes)), tf.float16)
    normalized_entropy = entropy_mean / max_entropy  # Normalize to [0, 1]
    normalized_mutual_information = mutual_information / max_entropy  # BALD normalized to [0, 1]

    # Variance normalization will be done in the main function using global min and max

    if logger:
        logger.debug("MC inference completed.")

    return {
        'entropy': normalized_entropy,
        'variance': variance_uncertainty,  # Return unnormalized variance
        'bald': normalized_mutual_information,
        'mean_probs': mean_probs,
        'mean_logits': mean_logits
    }


@tf.function(jit_compile=True)
def predict_logits(x, model):
    return model(x, training=True)


def main(config, logger):
    """
    Main function to run MC Inference, adjust uncertainties, and store data.
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

    # Calculate total number of samples
    try:
        N_SAMPLES = int(config.get('N_SAMPLES', -1))  # Ensure it's an integer
    except ValueError:
        logger.error("N_SAMPLES must be an integer.")
        raise ValueError("N_SAMPLES must be an integer.")

    if N_SAMPLES == -1:
        # Read the TRAINING_LIST to count total samples
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

    # Define chunk sizes
    chunk_size = config.get('CHUNK_SIZE', 1)  # Adjust chunk size as needed
    logger.debug(f"Chunk size for HDF5 datasets: {chunk_size}")

    # Calculate number of batches
    batch_size = config['BATCH_SIZE']
    num_batches = math.ceil(total_samples / batch_size)
    logger.debug(f"Number of batches to process: {num_batches}")

    # Limit the dataset to the required number of batches
    ds = ds.take(num_batches)
    logger.debug("Dataset limited to the required number of batches.")

    # First Pass: Compute global variance_min and variance_max with delta-based early stopping
    logger.info("Starting first pass to compute global variance min and max with delta-based early stopping...")

    # Initialize variables
    global_variance_min = np.inf
    global_variance_max = -np.inf
    delta_min_threshold = config.get('DELTA_MIN_THRESHOLD', 1e-6)
    delta_max_threshold = config.get('DELTA_MAX_THRESHOLD', 1e-6)
    min_batches_required = config.get('MIN_BATCHES_REQUIRED', 50)
    max_batches = config.get('MAX_BATCHES', 17600)
    stable_batches_required = config.get('STABLE_BATCHES_REQUIRED', 500)
    processed_batches = 0
    stable_batches = 0  # Counter for consecutive stable batches

    try:
        with tqdm(total=total_samples, desc="First pass", unit="samples") as pbar:
            for batch_idx, (filename_batch, x_batch, y_batch) in enumerate(ds):
                current_batch_size = x_batch.shape[0]
                # Perform MC inference
                uncertainties = mc_infer(
                    x_batch, n_iter=config['MC_N_ITER'], model=model, num_classes=num_classes, logger=logger
                )

                variance_uncertainty = uncertainties['variance'].numpy()  # [batch_size, H, W]

                batch_variance_min = variance_uncertainty.min()
                batch_variance_max = variance_uncertainty.max()

                # Update global variance min and max
                old_variance_min = global_variance_min
                old_variance_max = global_variance_max
                global_variance_min = min(global_variance_min, batch_variance_min)
                global_variance_max = max(global_variance_max, batch_variance_max)

                # Calculate deltas
                delta_min = abs(global_variance_min - old_variance_min) if old_variance_min != np.inf else np.inf
                delta_max = abs(global_variance_max - old_variance_max) if old_variance_max != -np.inf else np.inf

                # Update progress bar and show current min/max values
                pbar.set_postfix({
                    "min": f"{global_variance_min:.6f}",
                    "max": f"{global_variance_max:.6f}",
                    "delta_min": f"{delta_min:.6e}",
                    "delta_max": f"{delta_max:.6e}",
                })
                pbar.update(current_batch_size)

                processed_batches += 1

                # Early stopping condition
                if processed_batches >= min_batches_required:
                    if delta_min <= delta_min_threshold and delta_max <= delta_max_threshold:
                        stable_batches += 1  # Increment stable batches counter
                        if stable_batches >= stable_batches_required:
                            logger.info(
                                f"Variance min and max have stabilized over {stable_batches_required} consecutive "
                                f"batches after {processed_batches} total batches."
                            )
                            break
                    else:
                        stable_batches = 0  # Reset counter if deltas exceed thresholds

                # Maximum batches condition
                if processed_batches >= max_batches:
                    logger.info(f"Reached maximum number of batches ({max_batches}). Stopping first pass.")
                    break

    except Exception as e:
        logger.exception("An error occurred during the first pass.")
        raise e

    logger.info(
        f"First pass completed. Estimated global variance min: {global_variance_min}, max: {global_variance_max}"
    )

    # Second Pass: Compute uncertainties, normalize variance, adjust uncertainties, and store data
    logger.info("Starting second pass to compute uncertainties and store data...")

    # Reset the dataset iterator
    ds = get_dataset_v2(
        data_dir=config['DATA_DIR'],
        filelists=config['TRAINING_LIST'],
        repeat=False,
        shuffle=False,
        batch_size=config['BATCH_SIZE'],
        shuffle_buffer_size=config['SHUFFLE_BUFFER_SIZE'],
        out_channels=config['OUT_CHANNELS']
    )
    ds = ds.take(num_batches)
    logger.debug("Dataset iterator reset and limited to the required number of batches.")

    # Create an HDF5 file for storing results
    try:
        with h5py.File(output_file, 'w') as hdf5_file:
            logger.info("HDF5 file created for storing results.")

            # Create datasets with preallocated size to avoid resizing overhead
            rgb_image_dataset = hdf5_file.create_dataset(
                'rgb_images',
                shape=(total_samples,) + input_shape,
                dtype='uint8',
                compression='gzip',
                compression_opts=4,
                chunks=(chunk_size,) + input_shape
            )
            logits_dataset = hdf5_file.create_dataset(
                'logits',
                shape=(total_samples,) + input_shape[:2] + (num_classes,),
                dtype='float32',
                compression='gzip',
                compression_opts=4,
                chunks=(chunk_size,) + input_shape[:2] + (num_classes,)
            )
            variance_dataset = hdf5_file.create_dataset(
                'variance',
                shape=(total_samples,) + input_shape[:2],
                dtype='float32',
                compression='gzip',
                compression_opts=4,
                chunks=(chunk_size,) + input_shape[:2]
            )
            bald_dataset = hdf5_file.create_dataset(
                'bald',
                shape=(total_samples,) + input_shape[:2],
                dtype='float32',
                compression='gzip',
                compression_opts=4,
                chunks=(chunk_size,) + input_shape[:2]
            )
            filename_dataset = hdf5_file.create_dataset(
                'filenames',
                shape=(total_samples,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                compression='gzip',
                compression_opts=4,
                chunks=(chunk_size,)
            )

            logger.debug("All HDF5 datasets created successfully.")

            index = 0
            with tqdm(total=total_samples, desc="Second pass", unit="samples") as pbar:
                for batch_idx, (filename_batch, x_batch, y_batch) in enumerate(ds):
                    current_batch_size = x_batch.shape[0]
                    logger.debug(f"Processing batch {batch_idx + 1} with {current_batch_size} samples.")

                    # Perform MC inference
                    uncertainties = mc_infer(
                        x_batch, n_iter=config['MC_N_ITER'], model=model, num_classes=num_classes, logger=logger
                    )

                    # Extract uncertainties
                    entropy_np = uncertainties['entropy'].numpy()  # [batch_size, H, W]
                    variance_np = uncertainties['variance'].numpy()  # [batch_size, H, W]
                    bald_np = uncertainties['bald'].numpy()  # [batch_size, H, W]
                    mean_logits_np = uncertainties['mean_logits'].numpy()  # [batch_size, H, W, num_classes]

                    # Normalize variance using global min and max
                    normalized_variance = (variance_np - global_variance_min) / (
                            global_variance_max - global_variance_min + 1e-8)
                    # Ensure values are within [0, 1]
                    normalized_variance = np.clip(normalized_variance, 0, 1)

                    # Adjust uncertainties by weighting with (1 - entropy)
                    adjusted_variance = normalized_variance * (1 - entropy_np)
                    adjusted_bald = bald_np * (1 - entropy_np)

                    # Convert to numpy arrays
                    x_batch_np = x_batch.numpy()  # [batch_size, H, W, 3]
                    filename_batch_np = filename_batch.numpy()  # [batch_size]

                    # Scale images to uint8
                    rgb_images_uint8 = (x_batch_np * 255).astype('uint8')

                    # Determine the actual number of samples to write (handle last batch if it has fewer samples)
                    actual_batch_size = min(current_batch_size, total_samples - index)

                    # Handle cases where actual_batch_size might be zero or negative
                    if actual_batch_size <= 0:
                        logger.warning(f"Batch {batch_idx + 1}: No samples left to write.")
                        break

                    # Write data in batches
                    rgb_image_dataset[index:index + actual_batch_size] = rgb_images_uint8[:actual_batch_size]
                    logits_dataset[index:index + actual_batch_size] = mean_logits_np[:actual_batch_size].astype(
                        'float16')
                    variance_dataset[index:index + actual_batch_size] = adjusted_variance[:actual_batch_size].astype(
                        'float16')
                    bald_dataset[index:index + actual_batch_size] = adjusted_bald[:actual_batch_size].astype('float16')
                    filename_dataset[index:index + actual_batch_size] = [
                        fname.decode('utf-8') for fname in filename_batch_np[:actual_batch_size]
                    ]

                    logger.debug(f"Batch {batch_idx + 1}: Data written to HDF5 file.")

                    index += actual_batch_size  # Update index

                    # Update progress bar
                    pbar.update(actual_batch_size)

            # Handle cases where fewer samples were written than expected
            if index < total_samples:
                logger.warning(f"Expected to process {total_samples} samples, but only {index} were processed.")

            # Flush data to ensure all data is written
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
