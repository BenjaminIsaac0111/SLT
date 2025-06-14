import numpy as np
from pathlib import Path
from GUI.models.cross_validation import build_cv_folds


def create_files(tmp_path: Path) -> None:
    for g in range(3):
        for c in [0, 1]:
            (tmp_path / f"{g}_img_{c}.png").touch()


def test_build_cv_folds_group_preservation(tmp_path: Path):
    create_files(tmp_path)
    folds = build_cv_folds(tmp_path, n_splits=3, shuffle=False)
    assert len(folds) == 3
    seen_groups = set()
    for train_df, test_df, weights in folds:
        test_groups = {name.split("_")[0] for name in test_df["filename"]}
        assert len(test_groups) == 1
        assert test_groups.isdisjoint(seen_groups)
        seen_groups.update(test_groups)
        assert weights.shape[0] == 2
        for name in test_df["filename"]:
            assert name.split("_")[0] in test_groups
        train_groups = {name.split("_")[0] for name in train_df["filename"]}
        assert test_groups.isdisjoint(train_groups)
    assert len(seen_groups) == 3

