import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from omnilearned.dataloader import prepare_dataset


@pytest.fixture(scope="session")
def test_dataset_name():
    return "toy"


@pytest.fixture(scope="session")
def test_dataset_path(tmp_path_factory):
    path = Path(
        os.environ.get("OMNILEARNED_TOY_PATH", tmp_path_factory.mktemp("public_toy"))
    )
    for split in ("train", "val", "test"):
        prepare_dataset("toy", path, split, check_remote=False)
    return path


@pytest.fixture
def toy_local(tmp_path):
    rng = np.random.default_rng(2)
    for split in ("train", "val", "test"):
        folder = tmp_path / "toy" / split
        folder.mkdir(parents=True)
        for fid, maximum in enumerate((8, 11)):
            data = np.zeros((4, maximum, 4), np.float32)
            glob = np.zeros((4, 3), np.float32)
            target = np.zeros_like(data)
            for row, count in enumerate((0, 1, 3, 5)):
                data[row, :count] = rng.normal(size=(count, 4))
                data[row, :count, 2] += 3
                glob[row] = [fid, row, count / 100]
                target[row, :count] = 0.5
            with h5py.File(folder / f"file_{fid}.h5", "w") as f:
                f["data"], f["pid"], f["global"], f["data_pid"] = (
                    data,
                    np.arange(4) % 2,
                    glob,
                    target,
                )
    return tmp_path
