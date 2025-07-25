import argparse
import os
import time

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

# Use a high-resolution setting for figures
matplotlib.rcParams['figure.dpi'] = 300

# Define class components mapping
CLASS_COMPONENTS = {
    0: 'Non-Informative',
    1: 'Tumour',
    2: 'Stroma',
    3: 'Necrosis',
    4: 'Vessel',
    5: 'Inflammation',
    6: 'Tumour-Lumen',
    7: 'Mucin',
    8: 'Muscle'
}


# Enable Numba's fast math optimizations
@numba.jit(nopython=True, fastmath=True)
def compute_confidence_and_accuracy_numba(logits, y):
    """
    Compute confidence and accuracy using Numba for speed.
    Assumes logits and y are NumPy arrays with shape (batch_size, num_classes).
    """
    batch_size, num_classes = logits.shape
    confidence = np.empty(batch_size, dtype=np.float32)
    accuracy = np.empty(batch_size, dtype=np.float32)
    predicted_class = np.empty(batch_size, dtype=np.int32)
    ground_truth_class = np.empty(batch_size, dtype=np.int32)
    probabilities = np.empty((batch_size, num_classes), dtype=np.float32)

    for i in range(batch_size):
        # Apply softmax in a numerically stable way
        max_logit = logits[i].max()
        exp_logits = np.exp(logits[i] - max_logit)
        probs = exp_logits / exp_logits.sum()
        probabilities[i, :] = probs  # Store probabilities

        # Confidence
        confidence[i] = probs.max()

        # Predicted and ground truth classes
        predicted_class[i] = np.argmax(probs)
        ground_truth_class[i] = np.argmax(y[i])

        # Accuracy
        accuracy[i] = 1.0 if predicted_class[i] == ground_truth_class[i] else 0.0

    return confidence, accuracy, ground_truth_class, predicted_class, probabilities


def calculate_confidence_and_accuracy(logits, y, t=1.0):
    """
    Calculate confidence and accuracy based on the center pixel of the image.
    Optimized to use precomputed center pixel data and Numba-accelerated function.
    """
    # Apply temperature scaling
    logits = logits / t  # Shape: (batch_size, num_classes)

    # Utilize the Numba-accelerated function
    confidence, accuracy, ground_truth_class, predicted_class, probabilities = compute_confidence_and_accuracy_numba(
        logits, y)

    return confidence, accuracy, ground_truth_class, predicted_class, probabilities


def compute_entropy(probs):
    """Compute entropy of probability distributions."""
    epsilon = 1e-8
    entropies = -np.sum(probs * np.log(probs + epsilon), axis=-1)
    return entropies


def plot_accuracy_vs_uncertainty(uncertainties, accuracies, model_name, uncertainty_type, class_name=None, n_bins=20,
                                 save_path=None, pearson=None, spearman=None):
    """Plot accuracy vs uncertainty, including correlation statistics and model name."""
    # Set the style for publication-quality plots
    sns.set(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # Convert to pandas DataFrame for easier binning with qcut
    df = pd.DataFrame({
        'Uncertainty': uncertainties,
        'Accuracy': accuracies
    })

    # Define minimum samples per bin to ensure sufficient data
    min_samples_per_bin = 1  # Adjust based on your dataset size

    # Attempt to create quantile-based bins (equal count)
    try:
        df['Uncertainty Bin'], bins = pd.qcut(df['Uncertainty'], q=n_bins, retbins=True, duplicates='drop')
    except ValueError as e:
        print(f"Warning: {e}. Reducing number of bins.")
        n_bins = n_bins - 1
        df['Uncertainty Bin'], bins = pd.qcut(df['Uncertainty'], q=n_bins, retbins=True, duplicates='drop')

    # Compute bin statistics
    bin_stats = df.groupby('Uncertainty Bin')['Accuracy'].agg(['mean', 'std', 'count']).reset_index()

    # Filter out bins with insufficient samples
    bin_stats = bin_stats[bin_stats['count'] >= min_samples_per_bin]

    # Calculate bin centers
    bin_centers = bin_stats['Uncertainty Bin'].apply(lambda x: x.mid).values
    accuracies_in_bin = bin_stats['mean'].values
    accuracies_std_in_bin = bin_stats['std'].values / np.sqrt(bin_stats['count'].values)

    # Plot accuracy vs uncertainty with error bars
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        bin_centers,
        accuracies_in_bin,
        yerr=accuracies_std_in_bin,
        fmt='o-',
        color='blue',
        ecolor='lightgray',
        elinewidth=2,
        capsize=4,
        markersize=5,
        label='Accuracy'
    )

    # Add correlation statistics to the plot
    if pearson is not None and spearman is not None:
        plt.annotate(f'Pearson r: {pearson:.2f}\nSpearman ρ: {spearman:.2f}',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, horizontalalignment='left', verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

    # Labels and title
    plt.xlabel('Normalized Uncertainty')
    plt.ylabel('Accuracy')

    # Include class name in the title if provided
    if class_name:
        plt.title(f'Accuracy vs. {uncertainty_type.capitalize()} Uncertainty for {class_name} ({model_name})')
    else:
        plt.title(f'Accuracy vs. {uncertainty_type.capitalize()} Uncertainty ({model_name})')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Accuracy vs Uncertainty plot saved to {save_path}")

    plt.close()


def plot_precision_recall_vs_uncertainty(uncertainties, accuracies, model_name, uncertainty_type, save_path=None):
    """Plot Precision vs Recall for different uncertainty thresholds."""
    # Sort uncertainties and associated accuracies
    sorted_indices = np.argsort(uncertainties)
    sorted_uncertainties = uncertainties[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]

    # Compute precision and recall
    precision, recall, thresholds = precision_recall_curve(sorted_accuracies, -sorted_uncertainties)

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'{uncertainty_type.capitalize()} Uncertainty')

    # Labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve vs. {uncertainty_type.capitalize()} Uncertainty ({model_name})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Precision-Recall plot saved to {save_path}")

    plt.close()


def plot_uncertainty_calibration(uncertainties, accuracies, model_name, uncertainty_type, n_bins=10, save_path=None):
    """Plot uncertainty calibration curve."""
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(uncertainties, bins=bin_edges, right=True) - 1

    # Initialize arrays to hold bin accuracies and mean uncertainties
    bin_accuracies = np.zeros(n_bins)
    bin_mean_uncertainties = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    # Aggregate data in bins
    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_counts[i] = np.sum(bin_mask)
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(1 - accuracies[bin_mask])
            bin_mean_uncertainties[i] = np.mean(uncertainties[bin_mask])

    # Filter bins with at least one sample
    valid_bins = bin_counts > 0
    bin_accuracies = bin_accuracies[valid_bins]
    bin_mean_uncertainties = bin_mean_uncertainties[valid_bins]

    # Calculate calibration MSE
    calibration_mse = np.mean((bin_mean_uncertainties - bin_accuracies) ** 2)

    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(bin_mean_uncertainties, bin_accuracies, 's-', label=f'{uncertainty_type.capitalize()} Uncertainty')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Uncertainty')
    plt.ylabel('Observed Error Rate')
    plt.title(f'Uncertainty Calibration ({model_name})\nCalibration MSE: {calibration_mse:.4f}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Uncertainty Calibration plot saved to {save_path}")

    plt.close()


def perform_subgroup_analysis(uncertainties_dict, accuracies, ground_truth_classes, model_name, n_bins=20,
                              save_dir=None):
    """Perform class-wise analysis of accuracy vs uncertainty."""
    unique_classes = np.unique(ground_truth_classes)
    for cls in unique_classes:
        cls_mask = ground_truth_classes == cls
        cls_accuracies = accuracies[cls_mask]

        if len(cls_accuracies) == 0:
            continue  # Skip if no samples for the class

        # Get class name from mapping
        class_name = CLASS_COMPONENTS.get(cls, f'Class {cls}')

        for uncertainty_type, uncertainties in uncertainties_dict.items():
            cls_uncertainties = uncertainties[cls_mask]

            # Compute correlations
            pearson_corr, _ = pearsonr(cls_uncertainties, cls_accuracies)
            spearman_corr, _ = spearmanr(cls_uncertainties, cls_accuracies)

            # Plot Accuracy vs Uncertainty
            plot_path = os.path.join(save_dir, f'accuracy_vs_{uncertainty_type}_class_{class_name}.png')
            plot_accuracy_vs_uncertainty(
                cls_uncertainties,
                cls_accuracies,
                model_name,
                uncertainty_type,
                class_name=class_name,
                n_bins=n_bins,
                save_path=plot_path,
                pearson=pearson_corr,
                spearman=spearman_corr
            )


def main(config):
    # Start time
    start_time = time.time()

    model_prefix = 'dropout'
    model_name = f"{model_prefix}_{config['MODEL_NAME']}"
    model_dir = os.path.join(config['MODEL_DIR'])
    h5_file = os.path.join(model_dir, f"{model_name.split('.')[0]}_inference_output.h5")

    print(f"Loading data from {h5_file}...")
    # Open the HDF5 file
    with (h5py.File(h5_file, 'r') as hdf5_file):
        # Access the datasets
        logits_dataset = hdf5_file['logits']
        y_dataset = hdf5_file['ground_truths']
        epistemic_uncertainty_dataset = hdf5_file['epistemic_uncertainty']
        aleatoric_uncertainty_dataset = hdf5_file['aleatoric_uncertainty']  # Added to read stored aleatoric uncertainty

        # Determine the total number of samples
        N = logits_dataset.shape[0]
        print(f"Total number of samples: {N}")

        # Precompute center indices
        H, W = logits_dataset.shape[1], logits_dataset.shape[2]
        center_h, center_w = H // 2, W // 2

        # Preallocate arrays for performance
        all_confidences = np.empty(N, dtype=np.float32)
        all_accuracies = np.empty(N, dtype=np.float32)
        all_epistemic_uncertainties = np.empty(N, dtype=np.float32)
        all_aleatoric_uncertainties = np.empty(N, dtype=np.float32)  # Added to store stored aleatoric uncertainties
        all_ground_truth_classes = np.empty(N, dtype=np.int32)
        all_predicted_classes = np.empty(N, dtype=np.int32)

        # Process data in batches with a progress bar
        # Increased batch_size for better performance; adjust based on available memory
        batch_size = 1000  # Adjusted from 1 to 1000
        total_batches = (N + batch_size - 1) // batch_size  # Calculate total number of batches

        for batch_start in tqdm(range(0, N, batch_size), desc="Processing batches", total=total_batches):
            batch_end = min(batch_start + batch_size, N)
            current_batch_size = batch_end - batch_start

            # Read only the center pixel data to minimize I/O
            batch_logits = logits_dataset[batch_start:batch_end, center_h, center_w, :]  # Shape: (batch_size, C)
            batch_y = y_dataset[batch_start:batch_end, center_h, center_w, :]  # Shape: (batch_size, C)
            batch_epistemic_uncertainties = epistemic_uncertainty_dataset[batch_start:batch_end, center_h,
                                            center_w]  # Shape: (batch_size,)
            batch_aleatoric_uncertainties = aleatoric_uncertainty_dataset[batch_start:batch_end, center_h,
                                            center_w]  # Shape: (batch_size,)

            # Calculate confidence, accuracy, and probabilities
            confidence, accuracies, ground_truth_class, predicted_class, probabilities =
            calculate_confidence_and_accuracy(
                batch_logits, batch_y
            )

            # Collect results
            all_confidences[batch_start:batch_end] = confidence
            all_accuracies[batch_start:batch_end] = accuracies
            all_epistemic_uncertainties[batch_start:batch_end] = batch_epistemic_uncertainties.astype(np.float32)
            all_aleatoric_uncertainties[batch_start:batch_end] = batch_aleatoric_uncertainties.astype(np.float32)
            all_ground_truth_classes[batch_start:batch_end] = ground_truth_class
            all_predicted_classes[batch_start:batch_end] = predicted_class

    print("Normalizing uncertainties...")
    # Normalize uncertainties using Min-Max Scaling for each uncertainty type
    uncertainties_dict = {
        'aleatoric': all_aleatoric_uncertainties,
        'epistemic': all_epistemic_uncertainties
    }

    for key in uncertainties_dict.keys():
        u_min = uncertainties_dict[key].min()
        u_max = uncertainties_dict[key].max()
        uncertainties_dict[key] = (uncertainties_dict[key] - u_min) / (
                u_max - u_min + 1e-8)  # Added epsilon to avoid division by zero
        print(f"{key.capitalize()} uncertainties normalized to range [0, 1] using min={u_min:.4f} and max={u_max:.4f}.")

    print("Performing statistical analysis and plotting...")
    # Create directories for plots
    plot_dirs = {
        'accuracy_vs_uncertainty': os.path.join(model_dir, 'accuracy_vs_uncertainty'),
        'precision_recall': os.path.join(model_dir, 'precision_recall'),
        'calibration': os.path.join(model_dir, 'calibration'),
        'subgroup_analysis': os.path.join(model_dir, 'subgroup_analysis')
    }
    for dir_path in plot_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Iterate over each uncertainty type
    for uncertainty_type, uncertainties in uncertainties_dict.items():
        print(f"Processing {uncertainty_type} uncertainty...")

        # Calculate Pearson and Spearman correlations
        pearson_corr, _ = pearsonr(uncertainties, all_accuracies)
        spearman_corr, _ = spearmanr(uncertainties, all_accuracies)
        print(f"{uncertainty_type.capitalize()} Uncertainty - Pearson Correlation: {pearson_corr:.4f}")
        print(f"{uncertainty_type.capitalize()} Uncertainty - Spearman Correlation: {spearman_corr:.4f}")

        # Plot Accuracy vs Uncertainty
        auc_plot_save_path = os.path.join(plot_dirs['accuracy_vs_uncertainty'],
                                          f'accuracy_vs_{uncertainty_type}_uncertainty.png')
        plot_accuracy_vs_uncertainty(
            uncertainties,
            all_accuracies,
            model_name,
            uncertainty_type,
            n_bins=10,
            save_path=auc_plot_save_path,
            pearson=pearson_corr,
            spearman=spearman_corr
        )

        # Plot Precision-Recall Curve
        pr_plot_save_path = os.path.join(plot_dirs['precision_recall'],
                                         f'precision_recall_{uncertainty_type}_uncertainty.png')
        plot_precision_recall_vs_uncertainty(
            uncertainties,
            all_accuracies,
            model_name,
            uncertainty_type,
            save_path=pr_plot_save_path
        )

        # Plot Uncertainty Calibration Curve
        calib_plot_save_path = os.path.join(plot_dirs['calibration'], f'calibration_{uncertainty_type}_uncertainty.png')
        plot_uncertainty_calibration(
            uncertainties,
            all_accuracies,
            model_name,
            uncertainty_type,
            n_bins=10,
            save_path=calib_plot_save_path
        )

    print("Performing Subgroup (Class-Wise) Analysis...")
    perform_subgroup_analysis(
        uncertainties_dict,
        all_accuracies,
        all_ground_truth_classes,
        model_name,
        n_bins=20,
        save_dir=plot_dirs['subgroup_analysis']
    )

    print(f"All plots saved in {model_dir}")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced and Optimized Plot Accuracy vs Uncertainty with YAML config')
    parser.add_argument('-c', '--config_path', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    # Load the YAML config file
    with open(args.config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Call the main function with the config
    main(config)
