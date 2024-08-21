import argparse
import h5py
import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

def plot_umap(features, labels, correctness, title, save_path):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5, alpha=0.6, label='Correct')

    incorrect_indices = correctness == 0
    plt.scatter(embedding[incorrect_indices, 0], embedding[incorrect_indices, 1], c='red', s=5, alpha=0.8,
                label='Incorrect')

    plt.colorbar(scatter, label='Labels')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
    h5_file_path = checkpoint_dir / 'model_outputs/features.h5'

    with h5py.File(h5_file_path, "r") as hf:
        first_layer_features = np.array(hf["logits"])
        second_to_last_layer_features = np.array(hf["second_to_last_layer_features"])
        ground_truths = np.array(hf["ground_truths"])
        predictions = np.array(hf["predictions"])
        correctness = np.array(hf["correctness"])

    plot_umap(
        features=first_layer_features,
        labels=ground_truths,
        correctness=correctness,
        title='UMAP of logits',
        save_path=checkpoint_dir / 'model_outputs/umap_logits.png'
    )

    plot_umap(
        features=second_to_last_layer_features,
        labels=ground_truths,
        correctness=correctness,
        title='UMAP of Second-to-Last Layer Features',
        save_path=checkpoint_dir / 'model_outputs/umap_second_to_last_layer_features.png'
    )
