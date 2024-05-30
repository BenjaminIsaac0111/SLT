import argparse
import csv
import os
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, classification_report, cohen_kappa_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set before tf import else tf vomits logging INFO...

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Progbar
from tensorflow_addons.layers import GroupNormalization

from TeaPotts.config import load_config
from TeaPotts.dataloader import get_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help=r'The extractor config file (YAML). Will use the default_configuration if none supplied.',
        default=Path('configurations/configuration.yaml'))
    args = parser.parse_args()
    cfg = load_config(args.config)

    checkpoint_dir = f"./ckpt/{cfg['MODEL_NAME']}/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(f'{checkpoint_dir}/outputs/'):
        os.makedirs(f'{checkpoint_dir}/outputs/')

    testing_data = get_dataset(
        cfg=cfg,
        repeat=False,
        transforms=None,
        shuffle=False,
        strategy=None,
        filelist=cfg['TESTING_LIST']
    )
    filelist = open(cfg['TESTING_LIST']).readlines()

    checkpoint = f"{checkpoint_dir}ckpt_{cfg['MODEL_NAME']}"
    tf.print(f"Loading from model checkpoint {checkpoint}")
    model = load_model(
        filepath=checkpoint,
        compile=False,
    )
    model.compile()
    model.summary()

    # Define the center and radius of the circle
    width = 512
    center_x = width // 2
    height = 1024
    center_y = height // 2

    radius = 2

    # Create tensors of x and y coordinates
    x = tf.range(width, dtype=tf.float32)
    y = tf.range(height, dtype=tf.float32)
    x, y = tf.meshgrid(x, y)

    # Compute the distance of each pixel from the center
    dist = tf.math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Create a boolean mask of pixels inside the circle
    mask = tf.math.less_equal(dist, radius)

    ds = iter(testing_data)
    patches = []
    point_true = []
    point_pred = []

    for i, patch in enumerate(ds):
        x_test, y_test = patch
        preds = model.predict(x_test, verbose=0)
        n, h, w, c = preds.shape
        preds = tf.argmax(tf.math.bincount(tf.argmax(preds, axis=-1)[mask[tf.newaxis, ...]].numpy()))
        y_test = tf.argmax(y_test[:, h // 2, w // 2, :], axis=-1).numpy()
        point_pred.append(preds)
        point_true.append(y_test[0])

        pbar = Progbar(target=len(testing_data))
        pbar.update(i + 1)

    with open(f"{checkpoint_dir}report.txt", mode="w") as txt_file:
        print("Classification report:\n", file=txt_file)
        print(classification_report(point_true, point_pred, zero_division=True), file=txt_file)
        print("Balanced accuracy score: {:.2f}".format(balanced_accuracy_score(point_true, point_pred)), file=txt_file)
        print("Cohen kappa score: {:.2f}".format(cohen_kappa_score(point_true, point_pred)), file=txt_file)

    with open(f"{checkpoint_dir}predictions.csv", mode='w', newline="") as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["Pred", "True"])
        # Write the data rows by zipping the two lists
        writer.writerows(zip(point_pred, point_true))
