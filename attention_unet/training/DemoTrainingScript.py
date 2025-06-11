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

from attention_unet.models.models import MCDropoutUnetBuilder, EnsembleMCDropoutUnetBuilder

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
EPOCHS = 8
MC_PASSES = 8  # Number of stochastic passes for MC Concrete dropout uncertainty.
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
        "input_size": (128, 128),
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
    ds = tfds.load(config["tfds_name"], split=split, as_supervised=False,
                   # data_dir=r"Z:\tensorflow_datasets"  # Comment out to use default download path.
                   )
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
    var_probs = tf.reduce_sum(tf.math.reduce_variance(predictions, axis=0), axis=-1)
    return mean_probs.numpy(), var_probs.numpy()


def ensemble_predict(model: EnsembleMCDropoutUnetBuilder, x: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    # Deterministic predictions from each ensemble member.
    preds = [submodel(x, training=False) for submodel in model.submodels]
    preds_tensor = tf.stack(preds, axis=0)
    overall_mean = tf.reduce_mean(preds_tensor, axis=0)
    ensemble_variance = tf.math.reduce_variance(preds_tensor, axis=0)
    ensemble_uncertainty = tf.reduce_mean(ensemble_variance, axis=-1).numpy()
    return overall_mean.numpy(), ensemble_uncertainty


def mc_dropout_predict_combined_v2(model: EnsembleMCDropoutUnetBuilder,
                                   x: tf.Tensor,
                                   mc_passes: int = MC_PASSES) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute combined uncertainty based on the law of total variance.
    For each ensemble member, perform multiple MC dropout passes to obtain:
        - The within-model variance (aleatoric uncertainty).
        - The predictive mean.
    Then, compute:
        - The variance of the ensemble means (epistemic uncertainty).
    The total uncertainty is the sum of the within-model and between-model variances.
    """
    ensemble_means = []
    ensemble_dropout_vars = []
    for submodel in model.submodels:
        # Run MC dropout for the current ensemble member.
        dropout_preds = [submodel(x, training=True) for _ in range(mc_passes)]
        dropout_preds = tf.stack(dropout_preds, axis=0)
        # Mean prediction for this submodel.
        mean_pred = tf.reduce_mean(dropout_preds, axis=0)
        # Average dropout variance per pixel (across the channel dimension).
        dropout_var = tf.reduce_mean(tf.math.reduce_variance(dropout_preds, axis=0), axis=-1)
        ensemble_means.append(mean_pred)
        ensemble_dropout_vars.append(dropout_var)

    ensemble_means = tf.stack(ensemble_means, axis=0)
    ensemble_dropout_vars = tf.stack(ensemble_dropout_vars, axis=0)

    # Compute the between-model (ensemble) uncertainty.
    variance_between = tf.math.reduce_variance(ensemble_means, axis=0)
    variance_between = tf.reduce_mean(variance_between, axis=-1)

    # Compute the average within-model (MC dropout) uncertainty across ensemble members.
    variance_within = tf.reduce_mean(ensemble_dropout_vars, axis=0)

    total_uncertainty = variance_within + variance_between
    overall_mean = tf.reduce_mean(ensemble_means, axis=0)

    return overall_mean.numpy(), total_uncertainty.numpy()


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
                             title: str = "Reliability Diagram", save_path=None):
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


def plot_results(image, mask, mean_pred, uncertainty, save_path: str = None):
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
    plt.imshow(uncertainty, cmap='hot')
    plt.title("Uncertainty")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_uncertainty_vs_correctness(uncertainties, ious, dices, num_quantiles: int = 20, save_path: str = None):
    """
    Plots the relationship between uncertainty and segmentation performance (IoU and Dice)
    using quantile-based binning. This version uses a line plot for the mean metrics with a
    shaded area indicating ±1 standard deviation.

    Parameters:
        uncertainties (list or np.ndarray): Array of uncertainty values for each image.
        ious (list or np.ndarray): Array of IoU values corresponding to the images.
        dices (list or np.ndarray): Array of Dice coefficient values corresponding to the images.
        num_quantiles (int): Number of quantile bins to use.
        save_path (str): Base path to save plots; if None, plots are displayed instead.
    """
    uncertainties = np.array(uncertainties)
    ious = np.array(ious)
    dices = np.array(dices)

    # Compute quantile bin edges
    quantile_edges = np.quantile(uncertainties, np.linspace(0, 1, num_quantiles + 1))
    # Ensure last bin includes its upper boundary
    quantile_edges[-1] += 1e-6  # Small epsilon to include max value

    # Compute bin centers dynamically
    quantile_centers = (quantile_edges[:-1] + quantile_edges[1:]) / 2.0

    # Initialize lists to store mean and std for each quantile
    iou_means, iou_stds = [], []
    dice_means, dice_stds = [], []

    # Compute mean and standard deviation per bin
    for i in range(num_quantiles):
        lower_edge = quantile_edges[i]
        upper_edge = quantile_edges[i + 1]
        indices = np.where((uncertainties >= lower_edge) & (uncertainties < upper_edge))[0]

        if len(indices) > 0:
            iou_means.append(np.mean(ious[indices]))
            iou_stds.append(np.std(ious[indices]))
            dice_means.append(np.mean(dices[indices]))
            dice_stds.append(np.std(dices[indices]))
        else:
            # Use NaN to avoid misleading 0s in visualization
            iou_means.append(np.nan)
            iou_stds.append(np.nan)
            dice_means.append(np.nan)
            dice_stds.append(np.nan)

    # IoU Plot: Line plot with shaded ±1 standard deviation area.
    plt.figure(figsize=(6, 5))
    plt.plot(quantile_centers, iou_means, marker='o', color='blue', label='Mean IoU')
    plt.fill_between(quantile_centers,
                     np.array(iou_means) - np.array(iou_stds),
                     np.array(iou_means) + np.array(iou_stds),
                     color='blue', alpha=0.2, label='±1 Std Dev')
    plt.xlabel("Uncertainty (Quantile Bin Centers)")
    plt.ylabel("IoU")
    plt.ylim(0.0, 1.0)
    plt.title("Uncertainty vs IoU (Line Plot with Std Dev)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{os.path.splitext(save_path)[0]}_iou_line.png", dpi=300)
        plt.close()
    else:
        plt.show()

    # Dice Plot: Line plot with shaded ±1 standard deviation area.
    plt.figure(figsize=(6, 5))
    plt.plot(quantile_centers, dice_means, marker='o', color='green', label='Mean Dice')
    plt.fill_between(quantile_centers,
                     np.array(dice_means) - np.array(dice_stds),
                     np.array(dice_means) + np.array(dice_stds),
                     color='green', alpha=0.2, label='±1 Std Dev')
    plt.xlabel("Uncertainty (Quantile Bin Centers)")
    plt.ylabel("Dice")
    plt.ylim(0.0, 1.0)
    plt.title("Uncertainty vs Dice (Line Plot with Std Dev)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{os.path.splitext(save_path)[0]}_dice_line.png", dpi=300)
        plt.close()
    else:
        plt.show()


# -----------------------------
# Ablation Study Pipeline for a Single Dataset
# -----------------------------
def run_pipeline_for_dataset(config_name: str, config: dict, ablation_mode: str = "combined",
                             force_train: bool = False) -> Dict:
    """
    Runs the training and evaluation pipeline for a given dataset configuration.

    Parameters:
        config_name (str): Name of the dataset.
        config (dict): Dataset configuration parameters.
        ablation_mode (str): One of "dropout", "ensemble", or "combined" indicating the model type.
        force_train (bool): If True, retrains the model even if a saved model exists.

    Returns:
        Dict: A dictionary containing performance metrics for the dataset.
    """
    print(f"\n--- Running pipeline for dataset: {config_name} | Mode: {ablation_mode} ---")
    results_dir = os.path.join(BASE_RESULTS_DIR, f"{config_name}_{ablation_mode}")
    os.makedirs(results_dir, exist_ok=True)

    # Define model file path.
    model_filepath = os.path.join(results_dir, f"model_{config_name}.tf")

    # Check if a pre-trained model exists.
    if os.path.exists(model_filepath) and not force_train:
        print("Pre-trained model found. Loading model...")
        model = tf.keras.models.load_model(model_filepath)
        # Ensure inference_fn is defined even when loading a model.
        if ablation_mode == "dropout":
            inference_fn = mc_dropout_predict
        elif ablation_mode == "ensemble":
            inference_fn = ensemble_predict
        else:
            inference_fn = mc_dropout_predict_combined_v2
    else:
        print("No pre-trained model found or force_train=True. Training a new model from scratch...")
        ds_train = get_dataset(config, split="train")
        ds_val = get_dataset(config, split="test")

        if ablation_mode == "dropout":
            builder = MCDropoutUnetBuilder(
                input_size=(config["input_size"][0], config["input_size"][1], 3),
                num_classes=config["num_classes"],
                num_levels=4,
                num_conv_per_level=4,
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
                inference_fn = mc_dropout_predict_combined_v2

        model = builder.build_model()
        model.summary()
        base_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
            run_eagerly=False
        )
        model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS)
        model.save(filepath=model_filepath, save_format="tf")

    # Continue with inference and evaluation using the (loaded or newly trained) model.
    uncertainties = []
    ious = []
    dices = []
    results_list = []
    all_mean_probs = []
    all_labels = []

    print("Starting inference on the test set...")
    ds_val = get_dataset(config, split="test")
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
            plot_results(img, gt_mask, mean_pred, uncertainty_map[i], save_path=save_path)
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
    ablation_modes = [
        "dropout",
        "ensemble",
        "combined"
    ]
    consolidated_results = []
    # Change this flag to True to force retraining even if a saved model exists.
    force_train = False
    for mode in ablation_modes:
        print(f"\n=== Running Ablation: {mode} ===")
        for dataset_name, config in datasets_to_test.items():
            dataset_metrics = run_pipeline_for_dataset(dataset_name, config, ablation_mode=mode,
                                                       force_train=force_train)
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
