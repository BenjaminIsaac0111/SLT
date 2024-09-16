import h5py
import numpy as np
import tensorflow as tf
from keras import Model
from keras.activations import softmax
from tensorflow.keras.models import load_model
from tqdm import tqdm
from Dataloader.dataloader import get_dataset
from Model.custom_layers import GroupNormalization, SpatialConcreteDropout, DropoutAttentionBlock
from cfg.config import load_config
from tensorflow.keras import mixed_precision

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# Configuration and paths
config_path = (r'C:\Users\wispy\OneDrive - University of Leeds\PhD '
               r'Projects\Attention-UNET\cfg\unet_training_experiments\unet_devel_local.yaml')
cfg = load_config(config_path)
model_path = f"{cfg['MODEL_DIR']}/dropout_{cfg['MODEL_NAME']}/dropout_ckpt_{cfg['MODEL_NAME']}"


def mc_infer(x, n_iter=20):
    probs = []
    logits = []

    for i in range(n_iter):
        preds = predict_logits(x)  # Get logits from the model
        softmax_preds = tf.nn.softmax(preds, axis=-1)  # Apply softmax to get probabilities
        probs.append(softmax_preds)
        logits.append(preds)

    # Stack logits and probs to create tensors
    probs = tf.stack(probs, axis=0)
    logits = tf.stack(logits, axis=0)

    # Calculate epistemic uncertainty as the variance of the predictions
    epistemic_uncertainty = tf.math.reduce_variance(probs, axis=0)

    # Calculate mean predictions and mean logits
    mean_pred = tf.reduce_mean(probs, axis=0)
    logits_mean = tf.reduce_mean(logits, axis=0)

    return epistemic_uncertainty, mean_pred, logits_mean


@tf.function
def predict_logits(x):
    return model(x, training=True)


# Load the model
model = load_model(
    model_path,
    custom_objects={
        'DropoutAttentionBlock': DropoutAttentionBlock,
        'GroupNormalization': GroupNormalization,
        'SpatialConcreteDropout': SpatialConcreteDropout,
    },
    compile=False
)

model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Load the dataset
ds = get_dataset(
    cfg=cfg,
    repeat=False,
    shuffle=False,
    transforms=None,
    filelists=cfg['TRAINING_LIST'],
    batch_size=1
)

# Get the shape from the first batch in the dataset to dynamically determine shapes
first_batch = next(iter(ds))
input_shape = first_batch[1].shape[1:]  # Assuming x has shape [batch_size, H, W, C]
logits_shape = (input_shape[0], input_shape[1], model.output.shape[-1])  # [H, W, num_classes]

# Create an HDF5 file
# calibration_data_file = fr"Z:\PathologyData\{cfg['MODEL_NAME']}_COLLECTED_UNCERTAINTIES.h5"
calibration_data_file = fr"C:\Users\wispy\OneDrive - University of Leeds\DATABACKUP\{cfg[
    'MODEL_NAME']}_COLLECTED_UNCERTAINTIES_2.h5"

with h5py.File(calibration_data_file, 'w') as hdf5_file:
    # Create datasets within the HDF5 file
    logits_dataset = hdf5_file.create_dataset(
        'logits', shape=(0,) + logits_shape, maxshape=(None,) + logits_shape,
        dtype='float32'
    )
    y_dataset = hdf5_file.create_dataset(
        'ground_truths', shape=(0,) + logits_shape, maxshape=(None,) + logits_shape,
        dtype='float32'
    )
    epistemic_uncertainties = hdf5_file.create_dataset(
        'epistemic_uncertainty', shape=(0,) + logits_shape,
        maxshape=(None,) + logits_shape, dtype='float32'
    )

    rgb_dataset = hdf5_file.create_dataset(
        'rgb_images', shape=(0,) + input_shape, maxshape=(None,) + input_shape,
        dtype='float32'
    )
    filename_dataset = hdf5_file.create_dataset(
        'filenames', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype()
    )

    for filename, x, y in tqdm(ds.take(64), desc="Processing samples"):
        epistemic_uncertainty, mean_pred, logits = mc_infer(x)

        # Append the full data to the HDF5 datasets
        logits_dataset.resize(logits_dataset.shape[0] + 1, axis=0)
        logits_dataset[-1] = logits.numpy()[0]

        y_dataset.resize(y_dataset.shape[0] + 1, axis=0)
        y_dataset[-1] = y.numpy()[0]

        epistemic_uncertainties.resize(epistemic_uncertainties.shape[0] + 1, axis=0)
        epistemic_uncertainties[-1] = epistemic_uncertainty.numpy()

        rgb_dataset.resize(rgb_dataset.shape[0] + 1, axis=0)
        rgb_dataset[-1] = x.numpy()[0]  # Store the original RGB image

        filename_dataset.resize(filename_dataset.shape[0] + 1, axis=0)
        filename_dataset[-1] = filename.numpy()[0].decode('utf-8')  # Store the filename as string

    hdf5_file.close()
