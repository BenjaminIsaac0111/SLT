import numpy as np
from pathlib import Path
from GUI.models.cross_validation import build_cv_folds, write_cv_folds


def create_files(tmp_path: Path) -> None:
    for g in range(3):
        for c in [0, 1]:
            (tmp_path / f"{g}_img_{c}.png").touch()


def test_build_cv_folds_group_preservation(tmp_path: Path):
    create_files(tmp_path)
    folds = build_cv_folds(tmp_path, n_splits=3, shuffle=False)
    assert len(folds) == 3
    seen_groups = set()
    for train_df, test_df in folds:
        test_groups = {name.split("_")[0] for name in test_df["filename"]}
        assert len(test_groups) == 1
        assert test_groups.isdisjoint(seen_groups)
        seen_groups.update(test_groups)
        for name in test_df["filename"]:
            assert name.split("_")[0] in test_groups
        train_groups = {name.split("_")[0] for name in train_df["filename"]}
        assert test_groups.isdisjoint(train_groups)
    assert len(seen_groups) == 3


def test_write_cv_folds_progress(tmp_path: Path):
    create_files(tmp_path)
    out_dir = tmp_path / "out"
    calls = []

    def _progress(i: int) -> None:
        calls.append(i)

    write_cv_folds(tmp_path, out_dir, n_splits=3, shuffle=False, progress=_progress)
    assert calls == [1, 2, 3]
    for i in range(1, 4):
        assert (out_dir / f"Fold_{i}_TrainingData.txt").exists()
        assert (out_dir / f"Fold_{i}_TestData.txt").exists()
        assert not (out_dir / f"Fold_{i}_weights.txt").exists()


def test_write_cv_folds_overwrite(tmp_path: Path):
    create_files(tmp_path)
    out_dir = tmp_path / "out"
    write_cv_folds(tmp_path, out_dir, n_splits=3, shuffle=False)
    (out_dir / "Fold_4_TestData.txt").touch()

    write_cv_folds(tmp_path, out_dir, n_splits=2, shuffle=False)
    assert not (out_dir / "Fold_4_TestData.txt").exists()
    assert (out_dir / "Fold_1_TestData.txt").exists()
    assert (out_dir / "Fold_2_TestData.txt").exists()

