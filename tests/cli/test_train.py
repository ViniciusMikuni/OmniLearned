import json

import pytest
import torch
from typer.testing import CliRunner

from omnilearned.cli import app

runner = CliRunner()


def invoke(args):
    if "--num-workers" not in args:
        args = [*args, "--num-workers", 0]
    result = runner.invoke(app, list(map(str, args)))
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    return result


@pytest.mark.remote
def test_train(test_dataset_name, test_dataset_path, tmp_path):
    result = invoke(
        [
            "train",
            "--dataset",
            test_dataset_name,
            "--path",
            test_dataset_path,
            "--output_dir",
            tmp_path,
            "--epoch",
            2,
            "--batch",
            2,
            "--iterations",
            2,
            "--val-iterations",
            2,
            "--no-check-remote",
        ]
    )
    assert "Training Complete" in result.output
    loss = json.loads((tmp_path / "training_.json").read_text())
    assert len(loss["val_loss"]) == 2 and 0 < loss["val_loss"][-1] < 1


@pytest.mark.parametrize(
    "mode",
    [
        "classifier",
        "regression",
        "generator",
        "pretrain",
        "pretrain_regression",
        "pretrain_decoder",
        "encoder",
        "segmentation",
        "ftag",
    ],
)
def test_all_modes_and_resume(mode, toy_local, tmp_path):
    if mode == "ftag":
        import h5py
        import numpy as np

        for path in toy_local.glob("toy/*/*.h5"):
            with h5py.File(path, "a") as f:
                del f["data_pid"]
                f["data_pid"] = np.zeros(f["data"].shape[:2], np.int64)
    out = tmp_path / "model"
    args = [
        "train",
        "--dataset",
        "toy",
        "--path",
        toy_local,
        "--output_dir",
        out,
        "--mode",
        mode,
        "--epoch",
        1,
        "--iterations",
        1,
        "--val-iterations",
        1,
        "--batch",
        2,
        "--num-part",
        11,
        "--no-check-remote",
        "--qk-norm",
    ]
    if "regression" in mode:
        args += ["--num-classes", 1]
    if mode == "segmentation":
        args += ["--num-gen-classes", 4]
    if mode == "ftag":
        args += ["--num-gen-classes", 2]
    invoke(args)
    file = next(out.glob("*latest.pth"))
    saved = torch.load(file, weights_only=False)
    assert (
        saved["epoch"] == 1
        and saved["global_step"] == 1
        and saved["ema"]
        and saved["rng_by_rank"]
    )
    if mode == "encoder":
        assert any(k.startswith("decoder.") for k in saved["model"])
    args[args.index("--epoch") + 1] = 2
    invoke([*args, "--resuming"])
    assert torch.load(file, weights_only=False)["epoch"] == 2


def test_config_precedence(toy_local, tmp_path):
    config = tmp_path / "test.yaml"
    config.write_text(
        f"dataset: toy\npath: {toy_local}\noutput_dir: {tmp_path / 'model'}\nepoch: 1\nbatch: 2\niterations: 1\nval_iterations: 1\ncheck_remote: false\nsize: lite\n"
    )
    invoke(["train", "--config", config, "--epoch", 999])
    assert json.loads((tmp_path / "model" / "resolved_.json").read_text())["epoch"] == 1
    config.write_text("no_such_parameter: 42")
    result = runner.invoke(app, ["train", "--config", str(config)])
    assert result.exit_code != 0 and "Unknown config option" in result.output


def test_best_checkpoint(monkeypatch, toy_local, tmp_path):
    import omnilearned.train as training

    values = iter([0.4, 0.3, 0.5])

    def run_pass(
        model,
        loader,
        c,
        device,
        high_level=False,
        optimizer=None,
        scheduler=None,
        scaler=None,
        ema=None,
    ):
        return {
            "loss": 0.6 if optimizer is not None else next(values),
            "loss_class": 0.0,
            "loss_gen": 0.0,
        }, int(optimizer is not None)

    monkeypatch.setattr(training, "pass_epoch", run_pass)
    out = tmp_path / "best"
    invoke(
        [
            "train",
            "--dataset",
            "toy",
            "--path",
            toy_local,
            "--output_dir",
            out,
            "--epoch",
            3,
            "--iterations",
            1,
            "--val-iterations",
            1,
            "--batch",
            2,
            "--no-check-remote",
        ]
    )
    assert torch.load(out / "best_model_.pt", weights_only=False)["epoch"] == 2
    assert torch.load(out / "best_model__latest.pth", weights_only=False)["epoch"] == 3


def test_exact_resume(monkeypatch, toy_local, tmp_path):
    import omnilearned.train as training

    args = [
        "train",
        "--dataset",
        "toy",
        "--path",
        toy_local,
        "--epoch",
        2,
        "--iterations",
        1,
        "--val-iterations",
        1,
        "--batch",
        2,
        "--size",
        "lite",
        "--no-check-remote",
        "--mode",
        "pretrain",
        "--num-workers",
        0,
    ]
    full, resumed = tmp_path / "full", tmp_path / "resume"
    invoke([*args, "--output_dir", full])
    original = training.pass_epoch
    calls = 0

    def interrupted(*a, **k):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("Simulated interruption")
        return original(*a, **k)

    monkeypatch.setattr(training, "pass_epoch", interrupted)
    result = runner.invoke(app, list(map(str, [*args, "--output_dir", resumed])))
    assert result.exit_code == 1 and "Simulated interruption" in result.output
    monkeypatch.setattr(training, "pass_epoch", original)
    invoke([*args, "--output_dir", resumed, "--resuming"])
    expected = torch.load(full / "best_model__latest.pth", weights_only=False)
    actual = torch.load(resumed / "best_model__latest.pth", weights_only=False)
    for k in expected["model"]:
        torch.testing.assert_close(
            expected["model"][k], actual["model"][k], rtol=0, atol=0
        )
