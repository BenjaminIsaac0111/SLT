import json
from pathlib import Path

import pytest

from GUI.models.io.Persistence import (
    _encode,
    _decode,
    save_state,
    load_state,
    migrate,
    ProjectState,
)
from GUI.configuration.configuration import LATEST_SCHEMA_VERSION


@pytest.fixture
def example_state() -> ProjectState:
    annotation = {
        "image_index": 0,
        "filename": "img0.png",
        "coord": [0, 0],
        "logit_features": [0.1, 0.2],
        "uncertainty": 0.5,
        "adjusted_uncertainty": 0.5,
        "class_id": -1,
        "cluster_id": 1,
        "model_prediction": None,
        "mask_rle": None,
        "mask_shape": None,
    }
    return ProjectState(
        schema_version=LATEST_SCHEMA_VERSION,
        data_backend="hdf5",
        data_path="/tmp/data",
        clusters={"1": [annotation]},
        cluster_order=[1],
        selected_cluster_id=1,
        annotation_method="Local Uncertainty Maxima",
    )


def test_encode_decode_round_trip():
    original = "{\"a\": 1}"
    encoded = _encode(original, level=3)
    decoded = _decode(encoded)
    assert decoded == original


def test_save_and_load_state(tmp_path: Path, example_state: ProjectState):
    path = tmp_path / "project.slt"
    save_state(example_state, path)
    loaded = load_state(path)
    assert loaded.dict() == example_state.dict()


def test_save_creates_directories(tmp_path: Path, example_state: ProjectState):
    path = tmp_path / "a" / "b" / "project.slt"
    save_state(example_state, path)
    assert path.exists()
    loaded = load_state(path)
    assert loaded.dict() == example_state.dict()


def test_saved_file_contents_match_state(tmp_path: Path, example_state: ProjectState):
    """Verify raw project file JSON matches the state dict."""
    path = tmp_path / "project.slt"
    save_state(example_state, path)
    raw_json = _decode(path.read_bytes())
    assert json.loads(raw_json) == example_state.dict()


def test_migrate_from_v2():
    raw = {
        "schema_version": 2,
        "data_backend": "hdf5",
        "data_path": "data",
        "clusters": {"1": []},
        "cluster_order": ["1"],
        "selected_cluster_id": None,
        # intentionally omit 'uncertainty'
        "annotation_method": "Local Uncertainty Maxima",
    }
    migrated = migrate(raw)
    assert migrated["schema_version"] == LATEST_SCHEMA_VERSION
    assert migrated["cluster_order"] == [1]
    assert migrated["uncertainty"] == "bald"

