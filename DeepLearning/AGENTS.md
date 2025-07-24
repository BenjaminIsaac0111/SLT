# Deep Learning Suite Guidelines

This directory hosts the TensorFlow based training and inference code.

## Directory layout
- `config/` – configuration loader and example YAML file.
- `dataloader/` – dataset preparation utilities.
- `models/` – network architectures and custom layers.
- `training/` – scripts for model training.
- `inference/` – inference tools and uncertainty estimation.
- `losses/` – custom loss functions.
- `processing/` – preprocessing and augmentation routines.

## Coding conventions
- Target **Python 3.8** and follow PEP8 with 4 spaces per indent.
- Provide type hints and docstrings for public modules, classes and functions.
- Keep top‑level scripts importable by guarding execution under
  `if __name__ == "__main__"`.

## Tests
- Add unit tests for new functionality under `DeepLearning/unittests`.
- Execute the full test suite with `pytest` from the repository root.

## Notes
- Use the YAML loader in `config/` for defining training parameters and keep
  `example_config.yaml` updated when new fields are introduced.
- Training and inference scripts should rely on TensorFlow 2.10.* and follow
  the existing usage of `mixed_precision` and `MirroredStrategy`.
