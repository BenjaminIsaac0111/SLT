import argparse
import csv
import os
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from Model.custom_layers import AttentionBlock, PixelShuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set before tf import else tf vomits logging INFO...

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Progbar
from tensorflow_addons.layers import GroupNormalization

from cfg.config import load_config
from Dataloader.dataloader import get_dataset

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

    test_data = get_dataset(
        cfg=cfg,
        repeat=False,
        shuffle=False,
        transforms=None,
        filelists=cfg['TESTING_LIST'],
        batch_size=cfg['VAL_BATCH_SIZE']  # Use the larger batch size for validation
    )

    filelist = open(cfg['TESTING_LIST']).readlines()

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
    model.summary()

    # Define the center and radius of the circle
    width = 512
    center_x = width // 2
    height = 1024
    center_y = height // 2

    radius = 0

    # Create tensors of x and y coordinates
    x = tf.range(width, dtype=tf.float32)
    y = tf.range(height, dtype=tf.float32)
    x, y = tf.meshgrid(x, y)

    # Compute the distance of each pixel from the center
    dist = tf.math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Create a boolean mask of pixels inside the circle
    mask = tf.math.less_equal(dist, radius)

    ds = iter(test_data)
    patches = []
    point_true = []
    point_pred = []

    for i, patch in enumerate(ds):
        x_test, y_test = patch
        preds = model.predict(x_test, verbose=0)
        n, h, w, c = preds.shape
        preds = tf.argmax(tf.math.bincount(tf.argmax(preds, axis=-1)[mask[tf.newaxis, ...]].numpy())).numpy()
        y_test = tf.argmax(y_test[:, h // 2, w // 2, :], axis=-1).numpy()
        point_pred.append(preds)
        point_true.append(y_test[0])

        pbar = Progbar(target=len(test_data))
        pbar.update(i + 1)

    with open(f"{checkpoint_dir}/model_outputs/report.txt", mode="w") as txt_file:
        print("Classification report:\n", file=txt_file)
        print(classification_report(point_true, point_pred, zero_division=True), file=txt_file)
        print("Balanced accuracy score: {:.2f}".format(balanced_accuracy_score(point_true, point_pred)), file=txt_file)
        print("Cohen kappa score: {:.2f}".format(cohen_kappa_score(point_true, point_pred)), file=txt_file)

    with open(f"{checkpoint_dir}/model_outputs/predictions.csv", mode='w', newline="") as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["Pred", "True"])
        # Write the data rows by zipping the two lists
        writer.writerows(zip(map(int, point_pred), map(int, point_true)))

    # Confusion Matrix
    conf_matrix = confusion_matrix(point_true, point_pred)
    conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    unique_labels = np.unique(point_true)

    # Plotting confusion matrix with raw counts using matplotlib
    fig, ax = plt.subplots(figsize=(10, 7))
    cax = ax.matshow(conf_matrix, cmap='Blues')
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Counts)')
    ax.set_xticks(np.arange(len(unique_labels)))
    ax.set_yticks(np.arange(len(unique_labels)))
    ax.set_xticklabels(unique_labels)
    ax.set_yticklabels(unique_labels)
    plt.savefig(f"{checkpoint_dir}/model_outputs/confusion_matrix_counts.png")
    plt.close()

    # Plotting confusion matrix with percentages using matplotlib
    fig, ax = plt.subplots(figsize=(10, 7))
    cax = ax.matshow(conf_matrix_percentage, cmap='Blues')
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(conf_matrix_percentage):
        ax.text(j, i, f'{val:.2f}%', ha='center', va='center', color='red')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Percentage)')
    ax.set_xticks(np.arange(len(unique_labels)))
    ax.set_yticks(np.arange(len(unique_labels)))
    ax.set_xticklabels(unique_labels)
    ax.set_yticklabels(unique_labels)
    plt.savefig(f"{checkpoint_dir}/model_outputs/confusion_matrix_percentage.png")
    plt.close()

    # Save confusion matrix counts as CSV
    conf_matrix_csv_path = f"{checkpoint_dir}/model_outputs/confusion_matrix_counts.csv"
    with open(conf_matrix_csv_path, mode='w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow([""] + list(unique_labels))  # Write the header row
        for i, row in enumerate(conf_matrix):
            writer.writerow([unique_labels[i]] + list(row))  # Write the data rows

    # Save confusion matrix percentages as CSV
    conf_matrix_percentage_csv_path = f"{checkpoint_dir}/model_outputs/confusion_matrix_percentage.csv"
    with open(conf_matrix_percentage_csv_path, mode='w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow([""] + list(unique_labels))  # Write the header row
        for i, row in enumerate(conf_matrix_percentage):
            writer.writerow([unique_labels[i]] + list(row))  # Write the data rows
