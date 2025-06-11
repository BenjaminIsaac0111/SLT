import argparse
from pathlib import Path
from time import time

import h5py
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_data(h5_file_path):
    with h5py.File(h5_file_path, "r") as hf:
        logits = np.array(hf["logits"])
        ground_truths = np.array(hf["ground_truths"])
        correctness = np.array(hf["correctness"])
        file_names = np.array(hf["filelist"]).astype(str)
    return logits, ground_truths, correctness, file_names


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, average='weighted', zero_division=0)
    recall = recall_score(y, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y, predictions, average='weighted', zero_division=0)
    report = classification_report(y, predictions, zero_division=0)
    confusion = confusion_matrix(y, predictions)
    return accuracy, precision, recall, f1, report, confusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help=r'The extractor config file (YAML). Will use the default configuration if none supplied.',
        default=Path('configurations/configuration.yaml'))
    args = parser.parse_args()
    cfg = load_config(args.config)

    checkpoint_dir = Path(cfg['MODEL_DIR']) / cfg['MODEL_NAME']
    h5_file_path = checkpoint_dir / 'model_outputs/validation_features.h5'

    logits, ground_truths, correctness, file_names = load_data(h5_file_path)

    # Load pre-trained models
    model_paths = {
        'LogisticRegression': checkpoint_dir / 'model_outputs/best_model_LogisticRegression_logits.joblib',
        'NaiveBayes': checkpoint_dir / 'model_outputs/best_model_NaiveBayes_logits.joblib',
        'DecisionTree': checkpoint_dir / 'model_outputs/best_model_DecisionTree_logits.joblib',
        'XGBoost': checkpoint_dir / 'model_outputs/best_model_XGBoost_logits.joblib'
    }

    results = []

    for name, model_path in model_paths.items():
        print(f"Evaluating {name}...")
        model = joblib.load(model_path)

        start_time = time()
        accuracy, precision, recall, f1, report, confusion = evaluate_model(model, logits, correctness)
        end_time = time()
        elapsed_time = end_time - start_time

        print(f"Results for {name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(confusion)

        result = {
            'Model': name,
            'Time (s)': elapsed_time,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        results.append(result)

    # Generate CSV report
    results_df = pd.DataFrame(results)
    results_df.to_csv(checkpoint_dir / 'model_outputs/meta_model_validation_report.csv', index=False)

    print("Model evaluation on validation data complete. Results saved to CSV.")
