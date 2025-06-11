import argparse
import os
import sys
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

from attention_unet.models.custom_layers import AttentionBlock, PixelShuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set before tf import else tf vomits logging INFO...

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow_addons.layers import GroupNormalization

from attention_unet.config.config import load_config
from attention_unet.dataloader.dataloader import get_dataset
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help=r'The extractor config file (YAML). Will use the default_configuration if none supplied.',
        default=Path('configurations/configuration.yaml'))
    args = parser.parse_args()
    cfg = load_config(args.config)

    checkpoint_dir = f"{cfg['MODEL_DIR']}/{cfg['MODEL_NAME']}/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(f'{checkpoint_dir}/model_outputs/'):
        os.makedirs(f'{checkpoint_dir}/model_outputs/')

    data = get_dataset(
        cfg=cfg,
        repeat=False,
        shuffle=False,
        transforms=None,
        filelists=[cfg['TRAINING_LIST'], cfg['TESTING_LIST']],
        batch_size=1
    )

    training_filelist = open(cfg['TRAINING_LIST']).readlines()
    test_filelist = open(cfg['TESTING_LIST']).readlines()
    filelist = [f.strip() for f in training_filelist] + [f.strip() for f in test_filelist]

    checkpoint = f"{checkpoint_dir}best_{cfg['MODEL_NAME']}"
    tf.print(f"Loading from model checkpoint {checkpoint}")
    model = load_model(
        filepath=checkpoint,
        compile=False,
        custom_objects={
            'GroupNormalization': GroupNormalization,
            'PixelShuffle': PixelShuffle,
            'AttentionBlock': AttentionBlock
        }
    )
    model.compile()

    # Modify model to output features from multiple layers
    logits_layer = model.get_layer('output_conv').output
    second_to_last_layer_output = model.get_layer('decoder_residual_add_0').output
    combined_model = Model(inputs=model.input, outputs=[logits_layer, second_to_last_layer_output, model.output])

    ds = iter(data)

    # Get the shape of the feature vectors from a single batch
    x, y, filename = next(ds)
    tf.print((filename, x.shape, y.shape))
    logits, second_to_last_layer_feature, _ = combined_model.predict(x, verbose=0)

    # Determine the shape of the feature vectors
    h, w = logits.shape[1:3]
    logits_shape = logits[:, h // 2, w // 2, :].shape[1:]
    second_to_last_layer_shape = second_to_last_layer_feature[:, h // 2, w // 2, :].shape[1:]

    ds = iter(data)  # Reset

    # Open the .h5 file for writing
    with h5py.File(f"{checkpoint_dir}/model_outputs/features.h5", "w") as hf:
        # Initialize datasets with maximum shape
        last_layer_logits = hf.create_dataset("logits", shape=(0,) + logits_shape, maxshape=(None,) + logits_shape,
                                              chunks=True, dtype=np.float32)
        second_to_last_layer_dset = hf.create_dataset("second_to_last_layer_features",
                                                      shape=(0,) + second_to_last_layer_shape,
                                                      maxshape=(None,) + second_to_last_layer_shape, chunks=True,
                                                      dtype=np.float32)
        ground_truths_dset = hf.create_dataset("ground_truths", shape=(0,), maxshape=(None,), chunks=True, dtype='int')
        predictions_dset = hf.create_dataset("predictions", shape=(0,), maxshape=(None,), chunks=True, dtype='int')
        correctness_dset = hf.create_dataset("correctness", shape=(0,), maxshape=(None,), chunks=True, dtype=np.float32)
        filelist_dset = hf.create_dataset("filelist", shape=(0,), maxshape=(None,), chunks=True,
                                          dtype=h5py.special_dtype(vlen=str))

        for i, (x, y, filename) in enumerate(
                tqdm(ds, total=data.cardinality().numpy(), desc="Processing", unit="batch", file=sys.stdout)):
            # Get features and predictions from the combined model
            logits, second_to_last_layer_feature, preds = combined_model.predict(x, verbose=0)
            # Extract the feature vector from the center pixel
            first_layer_center_pixel = logits[:, h // 2, w // 2, :].flatten()
            second_to_last_layer_center_pixel = second_to_last_layer_feature[:, h // 2, w // 2, :].flatten()

            # Get ground truth and prediction for the center pixel
            y_test_center = tf.argmax(y[:, h // 2, w // 2, :], axis=-1).numpy().flatten()
            preds_center = tf.argmax(preds[:, h // 2, w // 2, :], axis=-1).numpy().flatten()

            # Calculate correctness (1 for correct prediction, 0 for incorrect)
            correctness = (preds_center == y_test_center).astype(np.float32)

            # Extend datasets and append new data
            last_layer_logits.resize(last_layer_logits.shape[0] + 1, axis=0)
            last_layer_logits[-1] = first_layer_center_pixel

            second_to_last_layer_dset.resize(second_to_last_layer_dset.shape[0] + 1, axis=0)
            second_to_last_layer_dset[-1] = second_to_last_layer_center_pixel

            ground_truths_dset.resize(ground_truths_dset.shape[0] + 1, axis=0)
            ground_truths_dset[-1] = y_test_center

            predictions_dset.resize(predictions_dset.shape[0] + 1, axis=0)
            predictions_dset[-1] = preds_center

            correctness_dset.resize(correctness_dset.shape[0] + 1, axis=0)
            correctness_dset[-1] = correctness

            filelist_dset.resize(filelist_dset.shape[0] + 1, axis=0)
            filelist_dset[-1] = filename.numpy()[0].decode('utf-8')
