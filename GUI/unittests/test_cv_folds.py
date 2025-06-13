import pandas as pd
from pathlib import Path

from GUI.models.cv_folds import create_grouped_folds


def test_create_grouped_folds(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # create simple dataset with 3 groups and 2 classes
    filenames = [
        "1_a_tumour.png",
        "1_b_stroma.png",
        "2_a_tumour.png",
        "2_b_stroma.png",
        "3_a_tumour.png",
        "3_b_stroma.png",
    ]
    for name in filenames:
        (data_dir / name).write_text("x")

    out_dir = tmp_path / "out"
    create_grouped_folds(data_dir, out_dir, n_splits=3)

    # weights for whole dataset
    assert (out_dir / "weights.txt").exists()

    for i in range(1, 4):
        tfile = out_dir / f"Fold_{i}_TrainingData.txt"
        vfile = out_dir / f"Fold_{i}_TestData.txt"
        wfile = out_dir / f"Fold_{i}_weights.txt"
        assert tfile.exists()
        assert vfile.exists()
        assert wfile.exists()
        train_lines = tfile.read_text().strip().splitlines()
        test_lines = vfile.read_text().strip().splitlines()
        # each fold should have one group in test set -> two samples
        assert len(test_lines) == 2
        # training set should contain remaining four samples
        assert len(train_lines) == 4


def test_create_folds_ignore_non_images(tmp_path: Path) -> None:
    """Running generation twice in the same directory should succeed."""
    data_dir = tmp_path
    # first run will place outputs beside images
    for name in [
        "1_a_tumour.png",
        "1_b_stroma.png",
        "2_a_tumour.png",
        "2_b_stroma.png",
    ]:
        (data_dir / name).write_text("x")

    create_grouped_folds(data_dir, data_dir, n_splits=2)
    # second run should skip the generated CSV/TXT files
    create_grouped_folds(data_dir, data_dir, n_splits=2)
