"""Upstream train/evaluate CLI on an explicit single-process device."""

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from omnilearned.cli import app
from omnilearned.sbi.device import mps_available

runner = CliRunner()


def write_custom_dataset(root, n=96, p=12):
    rng = np.random.default_rng(0)
    for split in ["train", "test", "val"]:
        d = root / "custom" / split
        d.mkdir(parents=True)
        x = rng.normal(size=(n, p, 4)).astype(np.float32)
        x[:, :, 2] = np.abs(x[:, :, 2]) + 0.5  # legacy mask uses feature 2
        x[:, 9:, :] = 0.0  # padding
        glob = np.stack([rng.uniform(size=n), np.full(n, 9 / 100.0)], -1)
        with h5py.File(d / "part.h5", "w") as f:
            f["data"] = x
            f["pid"] = np.zeros(n, dtype=np.int64)
            f["global"] = glob.astype(np.float32)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not mps_available(),
                                                         reason="no MPS"))],
)
def test_train_and_generate_single_process(tmp_path, device):
    write_custom_dataset(tmp_path)
    out = tmp_path / "out"
    common = ["--dataset", "custom", "--path", str(tmp_path), "--mode", "generator",
              "--conditional", "--num-cond", "2", "--num-feat", "4",
              "--num-workers", "0", "--device", device, "--size", "small",
              "--batch", "32",
              # num_classes > 1 hits a pre-existing upstream issue: generate()
              # passes float labels to the class embedding (all devices).
              "--num-classes", "1"]
    res = runner.invoke(app, ["train", "-o", str(out), "--epoch", "1",
                              "--iterations", "2", "--lr", "1e-4"] + common)
    assert res.exit_code == 0, res.output + repr(res.exception)
    assert f"Training on device: {device}" in res.output
    res = runner.invoke(app, ["evaluate", "-i", str(out), "-o", str(out)] + common)
    assert res.exit_code == 0, res.output + repr(res.exception)
    with h5py.File(out / "generated__custom_0.h5") as f:
        gen = f["data"][...]
    assert gen.shape[0] == 96 and np.isfinite(gen).all()
