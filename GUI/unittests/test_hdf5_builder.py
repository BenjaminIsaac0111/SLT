from pathlib import Path
import h5py
import pandas as pd
from PIL import Image

from GUI.models.hdf5_builder import build_training_hdf5


def test_build_training_hdf5(tmp_path: Path) -> None:
    data_dir = tmp_path / "imgs"
    data_dir.mkdir()
    # create tiny images
    names = []
    for i in range(3):
        name = f"{i}_tumour.png"
        img = Image.new("RGB", (2, 2))
        img.save(data_dir / name)
        names.append(name)

    csv_file = tmp_path / "train.csv"
    pd.DataFrame({"filename": names, "class": ["tumour"] * 3}).to_csv(
        csv_file, index=False, header=False
    )

    out_file = tmp_path / "out.h5"
    build_training_hdf5(data_dir, csv_file, "model.h5", out_file, sample_size=2)

    with h5py.File(out_file, "r") as h5f:
        assert len(h5f["filenames"]) == 2
        assert list(h5f["class"].asstr()) == ["tumour", "tumour"]
