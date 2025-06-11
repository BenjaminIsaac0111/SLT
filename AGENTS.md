# Codex Agent Instructions

This repository implements an Attention U-Net model and a PyQt5 GUI for medical image segmentation.

## Running tests
Run the unit tests with `pytest` from the repository root. All changes to Python code should keep the tests passing.

## Coding style
- Target **Python 3.8**.
- Follow PEP8 using 4 spaces per indent.
- Include docstrings for public modules, classes and functions.

## Repository layout
- `Model/` – network architectures and custom layers.
- `Training/` – training scripts.
- `Inference/` – inference utilities.
- `Processing/` – data transforms and augmentation.
- `Dataloader/` – dataset helpers.
- `GUI/` – Qt-based graphical interface.
- See `GUI/AGENTS.md` for contribution guidelines specific to the GUI.
