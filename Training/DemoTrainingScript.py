import csv
import os
import random
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import mixed_precision

from Model.models import MCDropoutUnetBuilder, EnsembleMCDropoutUnetBuilder

# Set global seeds for reproducibility.
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Set mixed precision and enable XLA.
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

# -----------------------------
# Global Parameters & Hyperparameters
# -----------------------------
EPOCHS = 5
MC_PASSES = 5  # Number of stochastic passes for MC Concrete dropout uncertainty.
BATCH_SIZE = 32
# Base directory for saving all results.
BASE_RESULTS_DIR = "all_results"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

# -----------------------------
# Dataset Configurations
# -----------------------------
datasets_to_test = {
    "oxford_iiit_pet": {
        "tfds_name": "oxford_iiit_pet",
        "input_size": (64, 64),
        "num_classes": 3,  # labels 0,1,2 after subtracting 1.
        "preprocess_fn": None  # use default preprocessing below.
    },
}


# -----------------------------
# Generic Preprocessing Function
# -----------------------------
def default_preprocess(example, input_size, num_classes):
    image = tf.cast(example['image'], tf.float32) / 255.0
    mask = example['segmentation_mask']
    mask = tf.squeeze(mask, axis=-1)
    if num_classes == 3:
        mask = mask - 1
    image = tf.image.resize(image, input_size)
    mask = tf.image.resize(tf.expand_dims(tf.cast(mask, tf.float32), axis=-1),
                           input_size,
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)
    return image, mask


def get_dataset(config, split: str) -> tf.data.Dataset:
    ds = tfds.load(config["tfds_name"], split=split, as_supervised=False)
    preprocess_fn = config.get("preprocess_fn", None)
    if preprocess_fn is None:
        ds = ds.map(lambda ex: default_preprocess(ex, config["input_size"], config["num_classes"]),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # For training, shuffle with a fixed seed; for test, do not shuffle.
    if split == "train":
        ds = ds.cache().shuffle(500, seed=42).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------------
# Segmentation Metrics Functions
# -----------------------------
def compute_iou(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    ious = []
    for c in range(num_classes):
        true_c = (y_true == c)
        pred_c = (y_pred == c)
        intersection = np.logical_and(true_c, pred_c).sum()
        union = np.logical_or(true_c, pred_c).sum()
        ious.append(1.0 if union == 0 else intersection / union)
    return np.mean(ious)


def compute_dice(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    dices = []
    for c in range(num_classes):
        true_c = (y_true == c).astype(np.float32)
        pred_c = (y_pred == c).astype(np.float32)
        intersection = np.sum(true_c * pred_c)
        sum_pixels = np.sum(true_c) + np.sum(pred_c)
        dices.append(1.0 if sum_pixels == 0 else 2 * intersection / sum_pixels)
    return np.mean(dices)


def compute_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-10
    entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
    return entropy


# -----------------------------
# MC Dropout Inference Functions
# -----------------------------
def mc_dropout_predict(model: keras.Model, x: tf.Tensor, mc_passes: int = MC_PASSES) -> Tuple[np.ndarray, np.ndarray]:
    predictions = []
    for m in range(mc_passes):
        print(f"MC Pass {m + 1}/{mc_passes} ...", end="\r")
        preds = model(x, training=True)
        predictions.append(preds)
    print("MC inference for current batch complete.")
    predictions = tf.stack(predictions, axis=0)
    mean_probs = tf.reduce_mean(predictions, axis=0)
    var_probs = tf.math.reduce_variance(predictions, axis=0)
    return mean_probs.numpy(), var_probs.numpy()


def ensemble_predict(model: EnsembleMCDropoutUnetBuilder, x: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    # Deterministic predictions from each ensemble member.
    preds = [submodel(x, training=False) for submodel in model.submodels]
    preds_tensor = tf.stack(preds, axis=0)
    overall_mean = tf.reduce_mean(preds_tensor, axis=0)
    ensemble_variance = tf.math.reduce_variance(preds_tensor, axis=0)
    ensemble_uncertainty = tf.reduce_mean(ensemble_variance, axis=-1).numpy()
    return overall_mean.numpy(), ensemble_uncertainty


def mc_dropout_predict_combined(model: EnsembleMCDropoutUnetBuilder,
                                x: tf.Tensor,
                                mc_passes: int = MC_PASSES) -> Tuple[np.ndarray, np.ndarray]:
    ensemble_preds = []
    for m in range(mc_passes):
        preds = [submodel(x, training=True) for submodel in model.submodels]
        preds = tf.stack(preds, axis=0)
        ensemble_preds.append(preds)
    ensemble_preds = tf.stack(ensemble_preds, axis=0)
    dropout_var_per_member = tf.math.reduce_variance(ensemble_preds, axis=0)
    dropout_uncertainty = tf.reduce_mean(dropout_var_per_member, axis=[0, -1]).numpy()
    mean_preds_per_member = tf.reduce_mean(ensemble_preds, axis=0)
    overall_mean = tf.reduce_mean(mean_preds_per_member, axis=0)
    overall_entropy = compute_entropy(overall_mean.numpy())
    entropies = [compute_entropy(mean_preds_per_member[i].numpy()) for i in range(mean_preds_per_member.shape[0])]
    avg_member_entropy = np.mean(np.stack(entropies, axis=0), axis=0)
    ensemble_uncertainty = overall_entropy - avg_member_entropy
    dropout_min, dropout_max = np.min(dropout_uncertainty), np.max(dropout_uncertainty)
    ensemble_min, ensemble_max = np.min(ensemble_uncertainty), np.max(ensemble_uncertainty)
    dropout_norm = (dropout_uncertainty - dropout_min) / (dropout_max - dropout_min + 1e-10)
    ensemble_norm = (ensemble_uncertainty - ensemble_min) / (ensemble_max - ensemble_min + 1e-10)
    combined_uncertainty = dropout_norm + ensemble_norm
    return overall_mean.numpy(), combined_uncertainty


# -----------------------------
# Reliability & Plotting Functions
# -----------------------------
def compute_reliability_data(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15):
    pred_class = np.argmax(probs, axis=-1)
    pred_conf = np.max(probs, axis=-1)
    correctness = (pred_class == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(pred_conf, bin_edges, right=True)
    bin_confidences = []
    bin_accuracies = []
    ece = 0.0
    n = len(pred_conf)
    for b in range(1, num_bins + 1):
        in_bin = (bin_indices == b)
        n_bin = in_bin.sum()
        avg_conf = pred_conf[in_bin].mean() if n_bin > 0 else 0.0
        avg_acc = correctness[in_bin].mean() if n_bin > 0 else 0.0
        bin_confidences.append(avg_conf)
        bin_accuracies.append(avg_acc)
        ece += np.abs(avg_acc - avg_conf) * (n_bin / n)
    return np.array(bin_confidences), np.array(bin_accuracies), ece


def plot_reliability_diagram(bin_confidences: np.ndarray, bin_accuracies: np.ndarray, ece: float,
                             num_bins: int = 15, title: str = "Reliability Diagram", save_path=None):
    plt.figure(figsize=(6, 6))
    plt.plot(bin_confidences, bin_accuracies, marker='o', label='Reliability')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(f"{title}\nECE = {ece:.4f}")
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.legend(loc='lower right')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_results(image, mask, mean_pred, combined_uncertainty, idx: int, save_path: str = None,
                 num_classes: int = None):
    plt.figure(figsize=(16, 3))
    plt.subplot(1, 5, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")
    plt.subplot(1, 5, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Ground Truth")
    plt.axis("off")
    plt.subplot(1, 5, 3)
    pred_class = np.argmax(mean_pred, axis=-1)
    plt.imshow(pred_class, cmap='jet')
    plt.title("Prediction")
    plt.axis("off")
    plt.subplot(1, 5, 4)
    plt.imshow(combined_uncertainty, cmap='hot')
    plt.title("Uncertainty")
    plt.axis("off")
    plt.subplot(1, 5, 5)
    plt.imshow(combined_uncertainty, cmap='hot')
    plt.title("Adjusted Uncertainty")
    plt.axis("off")
    plt.suptitle(f"Example {idx}")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_uncertainty_vs_correctness(uncertainties, ious, dices, num_bins: int = 10, save_path: str = None):
    uncertainties = np.array(uncertainties)
    ious = np.array(ious)
    dices = np.array(dices)
    bins = np.linspace(uncertainties.min(), uncertainties.max(), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    iou_means, iou_stds = [], []
    dice_means, dice_stds = [], []
    for i in range(num_bins):
        indices = np.where((uncertainties >= bins[i]) & (uncertainties < bins[i + 1]))[0]
        if len(indices) > 0:
            iou_means.append(np.mean(ious[indices]))
            iou_stds.append(np.std(ious[indices]))
            dice_means.append(np.mean(dices[indices]))
            dice_stds.append(np.std(dices[indices]))
        else:
            iou_means.append(0)
            iou_stds.append(0)
            dice_means.append(0)
            dice_stds.append(0)
    plt.figure(figsize=(12, 5))
    bar_width = bins[1] - bins[0]
    plt.subplot(1, 2, 1)
    plt.bar(bin_centers, iou_means, width=bar_width, alpha=0.6, label='Mean IoU')
    plt.errorbar(bin_centers, iou_means, yerr=iou_stds, fmt='none', ecolor='black', capsize=5, label='Std. Dev.')
    plt.xlabel("Avg Uncertainty")
    plt.ylabel("IoU")
    plt.title("Uncertainty vs IoU")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.bar(bin_centers, dice_means, width=bar_width, alpha=0.6, color='green', label='Mean Dice')
    plt.errorbar(bin_centers, dice_means, yerr=dice_stds, fmt='none', ecolor='black', capsize=5, label='Std. Dev.')
    plt.xlabel("Avg Uncertainty")
    plt.ylabel("Dice")
    plt.title("Uncertainty vs Dice")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# -----------------------------
# Ablation Study Pipeline for a Single Dataset
# -----------------------------
def run_pipeline_for_dataset(config_name: str, config: dict, ablation_mode: str = "combined") -> Dict:
    """
    ablation_mode: "dropout", "ensemble", or "combined"
    """
    print(f"\n--- Running pipeline for dataset: {config_name} | Mode: {ablation_mode} ---")
    results_dir = os.path.join(BASE_RESULTS_DIR, f"{config_name}_{ablation_mode}")
    os.makedirs(results_dir, exist_ok=True)
    ds_train = get_dataset(config, split="train")
    ds_val = get_dataset(config, split="test")
    if ablation_mode == "dropout":
        builder = MCDropoutUnetBuilder(
            input_size=(config["input_size"][0], config["input_size"][1], 3),
            num_classes=config["num_classes"],
            num_levels=2,
            num_conv_per_level=2,
            num_filters=32,
            regularisation=None,
            use_attention=True,
            activation=tf.keras.layers.LeakyReLU(),
            return_logits=False
        )
        inference_fn = mc_dropout_predict
    else:
        builder = EnsembleMCDropoutUnetBuilder(
            n_models=3,
            input_size=(config["input_size"][0], config["input_size"][1], 3),
            num_classes=config["num_classes"],
            num_levels=2,
            num_conv_per_level=2,
            num_filters=32,
            regularisation=None,
            use_attention=True,
            activation=tf.keras.layers.LeakyReLU(),
            return_logits=False
        )
        if ablation_mode == "ensemble":
            inference_fn = ensemble_predict
        else:
            inference_fn = mc_dropout_predict_combined

    model = builder.build_model()
    model.summary()
    base_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS)
    model.save(filepath=os.path.join(results_dir, f"model_{config_name}.tf"), save_format="tf")
    uncertainties = []
    ious = []
    dices = []
    results_list = []
    all_mean_probs = []
    all_labels = []
    print("Starting inference on the test set...")
    total_batches = sum(1 for _ in ds_val)
    for batch_idx, (images, masks) in enumerate(ds_val):
        mean_probs, uncertainty_map = inference_fn(model, images)
        all_mean_probs.append(mean_probs.reshape(-1, config["num_classes"]))
        all_labels.append(masks.numpy().reshape(-1))
        print(f"Processing batch {batch_idx + 1}/{total_batches} ...")
        batch_size = images.shape[0]
        for i in range(batch_size):
            gt_mask = masks[i].numpy()
            mean_pred = mean_probs[i]
            pred_mask = np.argmax(mean_pred, axis=-1)
            iou = compute_iou(gt_mask, pred_mask, config["num_classes"])
            dice = compute_dice(gt_mask, pred_mask, config["num_classes"])
            avg_uncertainty = np.mean(uncertainty_map[i])
            uncertainties.append(avg_uncertainty)
            ious.append(iou)
            dices.append(dice)
            results_list.append([avg_uncertainty, iou, dice])
        print(f"Batch {batch_idx + 1} processed.")
    print("Inference complete.")
    all_mean_probs = np.concatenate(all_mean_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    bin_confidences, bin_accuracies, ece = compute_reliability_data(
        probs=all_mean_probs, labels=all_labels, num_bins=15
    )
    scatter_plot_path = os.path.join(results_dir, "reliability_diagram.png")
    plot_reliability_diagram(bin_confidences, bin_accuracies, ece,
                             num_bins=15,
                             title=f"Reliability Diagram for {config_name} | {ablation_mode}",
                             save_path=scatter_plot_path)
    scatter_plot_path = os.path.join(results_dir, "uncertainty_vs_correctness.png")
    plot_uncertainty_vs_correctness(uncertainties, ious, dices, save_path=scatter_plot_path)
    print(f"Scatter plot saved to {scatter_plot_path}")
    csv_path = os.path.join(results_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image_Index", "Uncertainty", "IoU", "Dice"])
        for idx, (unc, iou, d) in enumerate(results_list):
            writer.writerow([idx, unc, iou, d])
    print(f"Metrics saved to {csv_path}")
    # Save example visualizations using a fixed seed for reproducibility.
    fixed_example_seed = 42
    tf.random.set_seed(fixed_example_seed)
    np.random.seed(fixed_example_seed)
    for images, masks in ds_val.take(1):
        mean_probs, uncertainty_map = inference_fn(model, images)
        for i in range(min(3, images.shape[0])):
            img = images[i].numpy()
            gt_mask = masks[i].numpy()
            mean_pred = mean_probs[i]
            save_path = os.path.join(results_dir, f"example_{i}.png")
            plot_results(img, gt_mask, mean_pred, uncertainty_map[i],
                         i, save_path=save_path, num_classes=config["num_classes"])
            print(f"Example {i} saved to {save_path}")
    dataset_metrics = {
        "Dataset": config_name,
        "Avg_Uncertainty": np.mean(uncertainties),
        "Std_Uncertainty": np.std(uncertainties),
        "Avg_IoU": np.mean(ious),
        "Std_IoU": np.std(ious),
        "Avg_Dice": np.mean(dices),
        "Std_Dice": np.std(dices)
    }
    return dataset_metrics


# -----------------------------
# Main Loop: Run Ablation Studies and Consolidate Results
# -----------------------------
def main():
    ablation_modes = ["dropout", "ensemble", "combined"]
    consolidated_results = []
    for mode in ablation_modes:
        print(f"\n=== Running Ablation: {mode} ===")
        for dataset_name, config in datasets_to_test.items():
            dataset_metrics = run_pipeline_for_dataset(dataset_name, config, ablation_mode=mode)
            consolidated_results.append((mode, dataset_metrics))
    consolidated_csv_path = os.path.join(BASE_RESULTS_DIR, "consolidated_results.csv")
    with open(consolidated_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Ablation_Mode", "Dataset", "Avg_Uncertainty", "Std_Uncertainty",
                         "Avg_IoU", "Std_IoU", "Avg_Dice", "Std_Dice"])
        for mode, res in consolidated_results:
            writer.writerow([mode, res["Dataset"], res["Avg_Uncertainty"], res["Std_Uncertainty"],
                             res["Avg_IoU"], res["Std_IoU"], res["Avg_Dice"], res["Std_Dice"]])
    print(f"Consolidated results saved to {consolidated_csv_path}")
    print("\n--- Consolidated Results ---")
    for mode, res in consolidated_results:
        print(f"[{mode}] {res['Dataset']}: Avg IoU = {res['Avg_IoU']:.4f}, Avg Dice = {res['Avg_Dice']:.4f}, "
              f"Avg Uncertainty = {res['Avg_Uncertainty']:.4f}")


if __name__ == '__main__':
    main()
