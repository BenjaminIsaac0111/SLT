from pathlib import Path
from unittest.mock import patch
import logging
import pandas as pd
import sys
import types

from GUI.models.hdf5_builder import build_training_hdf5


def test_build_training_hdf5_invokes_mc(tmp_path: Path) -> None:
    csv_file = tmp_path / "train.csv"
    pd.DataFrame({"filename": ["a.png"], "class": ["x"]}).to_csv(
        csv_file, index=False, header=False
    )

    captured = {}

    fake_module = types.ModuleType("banker")

    def fake_main(cfg, *, logger, resume=False):
        captured["cfg"] = cfg

    fake_module.main = fake_main
    fake_module.setup_logging = lambda *a, **k: logging.getLogger("test")

    with patch.dict(sys.modules, {"DeepLearning.inference.main_unet_mc_banker": fake_module}):
        build_training_hdf5(
            tmp_path,
            csv_file,
            "/models/best_model.h5",
            tmp_path / "out.h5",
            sample_size=5,
        )

    cfg = captured["cfg"]
    assert cfg["MODEL_DIR"] == "/models"
    assert cfg["MODEL_NAME"] == "best_model.h5"
    assert cfg["OUTPUT_DIR"] == str(tmp_path)
    assert cfg["N_SAMPLES"] == 5
