import shutil
from unittest.mock import patch

import h5py
import pytest
import requests
from typer.testing import CliRunner

from omnilearned.cli import app
from omnilearned.dataloader import load_data, prepare_dataset


@pytest.mark.remote
def test_download(test_dataset_path, tmp_path):
    shutil.copytree(test_dataset_path / "toy", tmp_path / "toy")
    result = CliRunner().invoke(
        app, ["dataloader", "--dataset", "toy", "--folder", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    for split in ("train", "val", "test"):
        with h5py.File(next((tmp_path / "toy" / split).glob("*.h5"))) as f:
            assert len(f["data"]) == len(f["pid"])


def test_missing_exits(tmp_path):
    with patch("omnilearned.dataloader.remote_files", return_value=None):
        result = CliRunner().invoke(
            app, ["dataloader", "--dataset", "absent", "--folder", str(tmp_path)]
        )
    assert result.exit_code == 1 and "unavailable" in result.output


def test_missing_file_download(toy_local):
    folder = toy_local / "toy" / "train"
    original = (folder / "file_0.h5").read_bytes()
    payload = (folder / "file_1.h5").read_bytes()
    (folder / "file_1.h5").unlink()
    response = requests.Response()
    response.status_code = 200
    response._content = payload
    response._content_consumed = True
    with (
        patch(
            "omnilearned.dataloader.remote_files",
            return_value=["file_0.h5", "file_1.h5"],
        ),
        patch("omnilearned.dataloader.requests.get", return_value=response) as get,
    ):
        prepare_dataset("toy", toy_local)
    assert (
        get.call_count == 1
        and (folder / "file_0.h5").read_bytes() == original
        and (folder / "file_1.h5").read_bytes() == payload
    )


def test_order_and_batched_reads(toy_local):
    loader = load_data(
        "toy", toy_local, batch=3, num_workers=0, check_remote=False, shuffle=False
    )
    identities = [
        (int(f), int(r))
        for b in loader
        for f, r in zip(b["file_id"], b["row_id"], strict=False)
    ]
    assert identities == [(f, r) for f in range(2) for r in range(4)]
    samples = loader.dataset.__getitems__([5, 1, 5, 0])
    assert [s["file_id"] for s in samples] == [1, 0, 1, 0]
    assert [s["row_id"] for s in samples] == [1, 1, 1, 0]


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("mode", ["classifier", "pretrain", "pretrain_regression"])
def test_labels_match_upstream(tmp_path, split, mode):
    import numpy as np
    import torch

    from omnilearned.data_spec import LABEL_OFFSETS, PRETRAIN_DATASETS

    expected = []
    for name in PRETRAIN_DATASETS:
        folder = tmp_path / name / split
        folder.mkdir(parents=True)
        labels = np.array([0, 1]) + LABEL_OFFSETS.get(name, 0)
        expected.extend(labels.tolist())
        with h5py.File(folder / "events.h5", "w") as f:
            f["data"] = np.ones((2, 3, 4), dtype=np.float32)
            f["pid"] = labels
    loader = load_data(
        "pretrain",
        tmp_path,
        dataset_type=split,
        mode=mode,
        batch=3,
        shuffle=False,
        check_remote=False,
    )
    actual = torch.cat([b["y"] for b in loader]).tolist()
    assert actual == expected
    assert min(actual) >= 0 and max(actual) < 210
    loader.dataset.close()
    for name in PRETRAIN_DATASETS:
        single = load_data(
            name,
            tmp_path,
            dataset_type=split,
            mode=mode,
            batch=2,
            shuffle=False,
            check_remote=False,
        )
        assert next(iter(single))["y"].tolist() == [0, 1]
        single.dataset.close()
        with h5py.File(tmp_path / name / split / "events.h5") as f:
            np.testing.assert_array_equal(
                f["pid"][:], np.array([0, 1]) + LABEL_OFFSETS.get(name, 0)
            )
