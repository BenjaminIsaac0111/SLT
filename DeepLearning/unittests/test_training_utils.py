"""Tests for training utility helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from DeepLearning.training.utils import (
    compute_class_histogram_from_json,
    count_samples_from_json,
)


def test_compute_class_histogram_from_json(tmp_path: Path) -> None:
    """Histogram counts reflect class frequency."""

    data = {
        "img1": [{"class_id": 0}, {"class_id": 1}],
        "img2": [{"class_id": 1}, {"class_id": 1}],
    }
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps(data))

    hist = compute_class_histogram_from_json(json_path, num_classes=2)
    assert hist == [1, 3]


def test_count_samples_from_json(tmp_path: Path) -> None:
    """Counting entries in the annotation file returns correct value."""

    data = {"a": [], "b": []}
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps(data))

    assert count_samples_from_json(json_path) == 2

