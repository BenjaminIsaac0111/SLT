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
        tfile = out_dir / f"Fold_{i}_TrainingData.csv"
        vfile = out_dir / f"Fold_{i}_TestData.csv"
        wfile = out_dir / f"Fold_{i}_weights.txt"
        assert tfile.exists()
        assert vfile.exists()
        assert wfile.exists()

        train_df = pd.read_csv(tfile, header=None, names=["fname", "cls"])
        test_df = pd.read_csv(vfile, header=None, names=["fname", "cls"])
        assert set(train_df["cls"]) <= {"tumour", "stroma"}
        assert set(test_df["cls"]) <= {"tumour", "stroma"}
        # each fold should have one group in test set -> two samples
        assert len(test_df) == 2
        # training set should contain remaining four samples
        assert len(train_df) == 4
