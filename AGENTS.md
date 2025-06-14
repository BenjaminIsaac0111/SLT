# Codex Agent Instructions

This repository implements an Attention U-Net model and a PyQt5 GUI for medical image segmentation.

## Requirements
Install the Python dependencies before running any tests. The recommended
environment specification is provided in `environment.yml` and can be created
with:

```bash
conda env create -f environment.yml
conda activate SLT
```

If using `pip`, ensure Python 3.8 is installed and install the following
packages:

- tensorflow==2.10.*
- tensorflow-datasets<4.9
- pyaml==23.5.8
- zstandard
- numpy
- pandas
- scikit-learn==1.3.0
- matplotlib
- seaborn
- scipy
- pillow
- h5py
- tqdm
- numba
- joblib
- PyQt5
- pydantic
- pytest
- pyinstaller

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
