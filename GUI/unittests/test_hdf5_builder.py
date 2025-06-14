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
    fake_module.set_global_seed = lambda *a, **k: None

    fake_model = types.SimpleNamespace(
        input_shape=(None, 256, 256, 3), output_shape=(None, 256, 256, 2)
    )

    tf_mod = types.ModuleType("tensorflow")
    keras_mod = types.ModuleType("tensorflow.keras")
    models_mod = types.ModuleType("tensorflow.keras.models")
    captured_kwargs = {}

    def fake_load_model(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return fake_model

    models_mod.load_model = fake_load_model
    keras_mod.models = models_mod
    tf_mod.keras = keras_mod

    with patch.dict(sys.modules, {
        "DeepLearning.inference.main_unet_mc_banker": fake_module,
        "tensorflow": tf_mod,
        "tensorflow.keras": keras_mod,
        "tensorflow.keras.models": models_mod,
        "DeepLearning.models.custom_layers": types.SimpleNamespace(
            DropoutAttentionBlock=object,
            GroupNormalization=object,
            SpatialConcreteDropout=object,
        ),
    }):
        build_training_hdf5(
            tmp_path,
            csv_file,
            "/models/best_model.h5",
            tmp_path / "out.h5",
            sample_size=5,
            mc_iter=7,
        )

    cfg = captured["cfg"]
    assert cfg["MODEL_DIR"] == "/models"
    assert cfg["MODEL_NAME"] == "best_model.h5"
    assert cfg["OUTPUT_DIR"] == str(tmp_path)
    assert cfg["N_SAMPLES"] == 5
    assert cfg["MC_N_ITER"] == 7
    assert cfg["INPUT_SIZE"] == [256, 256, 3]
    assert cfg["OUT_CHANNELS"] == 2
    assert {
        "DropoutAttentionBlock",
        "GroupNormalization",
        "SpatialConcreteDropout",
    } <= set(captured_kwargs["custom_objects"].keys())
