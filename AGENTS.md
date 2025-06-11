# Codex Agent Instructions

This repository implements an Attention U-Net model and a PyQt5 GUI for medical image segmentation.

## Running tests
Run the unit tests with `pytest` from the repository root. All changes to Python code should keep the tests passing.

## Coding style
- Target **Python 3.8**.
- Follow PEP8 using 4 spaces per indent.
- Include docstrings for public modules, classes and functions.

## Repository layout

- `DeepLearning/config/` – loaders for config files
- `DeepLearning/Model/` – network architectures and custom layers.
- `DeepLearning/Training/` – training scripts.
- `DeepLearning/Inference/` – inference utilities.
- `DeepLearning/Processing/` – data transforms and augmentation.
- `DeepLearning/Dataloader/` – dataset helpers.
- `GUI/` – Qt-based graphical interface.
- See `GUI/AGENTS.md` for contribution guidelines specific to the GUI.
