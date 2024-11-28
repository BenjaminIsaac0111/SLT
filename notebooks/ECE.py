import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def calculate_ece(confidences, accuracies, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Adjusted binning to include boundary cases
        if i == 0:
            in_bin = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            acc_mean = np.mean(accuracies[in_bin])
            conf_mean = np.mean(confidences[in_bin])
            ece += bin_count * np.abs(acc_mean - conf_mean)

    # Normalize by the total number of samples
    ece /= len(confidences)

    return ece


def calculate_classwise_ece(confidences, accuracies, labels, n_bins=10):
    """Calculate Expected Calibration Error (ECE) for each class."""
    bins = np.linspace(0, 1, n_bins + 1)
    n_classes = np.max(labels) + 1  # Assuming labels are 0-indexed
    ece_per_class = np.zeros(n_classes)

    for class_idx in range(n_classes):
        class_confidences = confidences[labels == class_idx]
        class_accuracies = accuracies[labels == class_idx]

        ece = 0.0
        for i in range(n_bins):
            if i == 0:
                in_bin = (class_confidences >= bins[i]) & (class_confidences <= bins[i + 1])
            else:
                in_bin = (class_confidences > bins[i]) & (class_confidences <= bins[i + 1])

            bin_count = np.sum(in_bin)
            if bin_count > 0:
                acc_mean = np.mean(class_accuracies[in_bin])
                conf_mean = np.mean(class_confidences[in_bin])
                ece += bin_count * np.abs(acc_mean - conf_mean)

        ece /= len(class_confidences)
        ece_per_class[class_idx] = ece

    return ece_per_class


@tf.function(jit_compile=True)
def calculate_confidence_and_accuracy(logits, y, t=1.0):
    """Calculate confidence and accuracy of the model predictions."""
    logits = logits / t
    preds = tf.nn.softmax(logits, axis=-1)
    confidence = tf.reduce_max(preds, axis=-1)
    predicted_class = tf.argmax(preds, axis=-1)
    ground_truth = tf.argmax(y, axis=-1)
    accuracy = predicted_class == ground_truth
    return confidence, accuracy


def plot_calibration_curve(confidences, accuracies, n_bins=10, title_prefix='All Classes'):
    """Plot a calibration curve (reliability diagram) with bars showing deviation from the identity line,
    and a histogram of the distribution of instances per bin, in a single plot."""
    bins = np.linspace(0, 1, n_bins + 1)
    accuracies_in_bin = []
    confidences_in_bin = []
    residuals_in_bin = []
    instances_per_bin = []

    for i in range(n_bins):
        # Adjusted binning to include boundary cases
        if i == 0:
            in_bin = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])

        bin_count = np.sum(in_bin)
        instances_per_bin.append(bin_count)
        if bin_count > 0:
            acc_mean = np.mean(accuracies[in_bin])
            conf_mean = np.mean(confidences[in_bin])
            accuracies_in_bin.append(acc_mean)
            confidences_in_bin.append(conf_mean)
            residuals_in_bin.append(acc_mean - conf_mean)

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plotting the calibration curve
    ax1.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration (Identity)', color='gray')
    ax1.plot(confidences_in_bin, accuracies_in_bin, marker='o', color='blue', label='Model Calibration')
    for i in range(len(confidences_in_bin)):
        ax1.bar(confidences_in_bin[i], residuals_in_bin[i], width=0.02, bottom=confidences_in_bin[i], color='red',
                alpha=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{title_prefix} Reliability Diagram (Calibration Curve) with Histogram')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plotting the histogram on the same figure
    ax2 = ax1.twinx()
    ax2.hist(confidences, bins=bins, color='skyblue', edgecolor='black', alpha=0.3)
    ax2.set_ylabel('Number of Instances')
    ax2.set_ylim(0, max(instances_per_bin) * 1.1)

    plt.show()


def plot_classwise_calibration_curve(confidences, accuracies, labels, n_bins=10):
    """Plot calibration curves for each class."""
    bins = np.linspace(0, 1, n_bins + 1)
    n_classes = np.max(labels) + 1

    for class_idx in range(n_classes):
        class_confidences = confidences[labels == class_idx]
        class_accuracies = accuracies[labels == class_idx]

        accuracies_in_bin = []
        confidences_in_bin = []
        residuals_in_bin = []
        instances_per_bin = []

        for i in range(n_bins):
            if i == 0:
                in_bin = (class_confidences >= bins[i]) & (class_confidences <= bins[i + 1])
            else:
                in_bin = (class_confidences > bins[i]) & (class_confidences <= bins[i + 1])

            bin_count = np.sum(in_bin)
            instances_per_bin.append(bin_count)
            if bin_count > 0:
                acc_mean = np.mean(class_accuracies[in_bin])
                conf_mean = np.mean(class_confidences[in_bin])
                accuracies_in_bin.append(acc_mean)
                confidences_in_bin.append(conf_mean)
                residuals_in_bin.append(acc_mean - conf_mean)

        fig, ax1 = plt.subplots(figsize=(10, 8))

        ax1.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration (Identity)', color='gray')
        ax1.plot(confidences_in_bin, accuracies_in_bin, marker='o', color='blue', label='Model Calibration')
        for i in range(len(confidences_in_bin)):
            ax1.bar(confidences_in_bin[i], residuals_in_bin[i], width=0.02, bottom=confidences_in_bin[i], color='red',
                    alpha=0.5)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Reliability Diagram (Calibration Curve) for Class {class_idx}')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.hist(class_confidences, bins=bins, color='skyblue', edgecolor='black', alpha=0.3)
        ax2.set_ylabel('Number of Instances')
        ax2.set_ylim(0, max(instances_per_bin) * 1.1)

        plt.show()


def find_optimal_temperature(logits, y, n_bins=10, temperature_range=(0.5, 2.0), num_temperatures=50):
    """Find the optimal temperature that minimises the Expected Calibration Error (ECE)."""
    temperatures = np.linspace(temperature_range[0], temperature_range[1], num_temperatures, dtype=np.float32)
    ece_scores = []

    for t in temperatures:
        confidence, accuracy = calculate_confidence_and_accuracy(logits, y, t)
        ece = calculate_ece(confidence, accuracy, n_bins)
        ece_scores.append(ece)

    optimal_temperature = temperatures[np.argmin(ece_scores)]

    # Plot ECE vs Temperature
    plt.figure(figsize=(8, 6))
    plt.plot(temperatures, ece_scores, marker='o')
    plt.axvline(optimal_temperature, color='r', linestyle='--', label=f'Optimal Temperature: {optimal_temperature:.2f}')
    plt.xlabel('Temperature')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('ECE vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_temperature.astype(np.float32)


def find_optimal_temperature_per_class(logits, y, labels, n_bins=10, temperature_range=(0.1, 5), num_temperatures=50):
    """Find the optimal temperature for each class that minimizes the Expected Calibration Error (ECE)."""
    temperatures = np.linspace(temperature_range[0], temperature_range[1], num_temperatures, dtype=np.float32)
    n_classes = np.max(labels) + 1
    optimal_temperatures = np.zeros(n_classes)

    for class_idx in range(n_classes):
        class_logits = logits[labels == class_idx]
        class_y = y[labels == class_idx]
        ece_scores = []

        for t in temperatures:
            confidence, accuracy = calculate_confidence_and_accuracy(class_logits, class_y, t)
            ece = calculate_ece(confidence, accuracy, n_bins)
            ece_scores.append(ece)

        optimal_temperature = temperatures[np.argmin(ece_scores)]
        optimal_temperatures[class_idx] = optimal_temperature

    return optimal_temperatures.astype(np.float32)
