import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

from omnilearned.cli import app
from omnilearned.dataloader import prepare_dataset
from omnilearned.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


def test_submit_dry_run_and_sbatch(monkeypatch, tmp_path):
    script = tmp_path / "dry.slurm"
    args = [
        "submit",
        "--nodes",
        "2",
        "--module",
        "",
        "--script",
        str(script),
        "--dry-run",
        "--",
        "train",
        "--dataset",
        "top",
        "--path",
        "/data/path with spaces",
        "--save-tag",
        "test",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    contents = script.read_text()
    assert (
        "#SBATCH --nodes=2" in contents
        and "--path '/data/path with spaces'" in contents
    )
    assert "--save-tag test" in contents and "LOCAL_RANK=$SLURM_LOCALID" in contents
    import subprocess

    subprocess.run(["bash", "-n", str(script)], check=True)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "12345\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args.remove("--dry-run")
    args[args.index("--script") + 1] = str(tmp_path / "live.slurm")
    result = runner.invoke(app, args)
    assert result.exit_code == 0 and "Submitted job 12345" in result.output
    assert calls == [["sbatch", "--parsable", str(tmp_path / "live.slurm")]]
    result = runner.invoke(app, args)
    assert result.exit_code == 1  # Never silently overwrite an existing job script.


def test_index_exists_only_policy(toy_local):
    prepare_dataset("toy", toy_local, "train", check_remote=False)
    folder = toy_local / "toy/train"
    index, manifest = folder / "file_index.npy", folder / "file_manifest.json"
    original = index.read_bytes(), index.stat().st_mtime_ns, manifest.read_bytes()
    file = folder / "file_0.h5"
    os.utime(file, ns=(file.stat().st_atime_ns, file.stat().st_mtime_ns + 1_000_000))
    with h5py.File(file, "a") as f:
        f.attrs["extra"] = "metadata changed"
    prepare_dataset("toy", toy_local, "train", check_remote=False)
    assert original == (
        index.read_bytes(),
        index.stat().st_mtime_ns,
        manifest.read_bytes(),
    )
    with h5py.File(folder / "new.h5", "w") as f:
        f["data"], f["pid"] = np.ones((2, 3, 4)), np.zeros(2)
    prepare_dataset("toy", toy_local, "train", check_remote=False)
    assert len(np.load(index)) == 8
    index.unlink()  # User explicitly requests rebuilding by removing it.
    prepare_dataset("toy", toy_local, "train", check_remote=False)
    assert len(np.load(index)) == 10
    manifest.unlink()
    with pytest.raises(RuntimeError, match="exists without"):
        prepare_dataset("toy", toy_local, "train", check_remote=False)
    assert len(np.load(index)) == 10


def test_metrics_ties_and_zero_background():
    labels = np.array([0, 0, 1, 1])
    prediction = np.array([[0.8, 0.2], [0.8, 0.2], [0.1, 0.9], [0.1, 0.9]])
    result = binary_metrics(labels, prediction)
    assert result["accuracy"] == result["auc"] == 1
    for p in result["working_points"]:
        assert p["signal_efficiency"] == 1 and np.isinf(p["background_rejection"])
    tied = binary_metrics(labels, np.full((4, 2), 0.5))
    assert tied["auc"] == 0.5
    assert all(p["background_rejection"] == 1 for p in tied["working_points"])
