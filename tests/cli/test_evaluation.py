import h5py
import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from omnilearned.cli import app


def invoke(args):
    # Keep CPU integration tests independent of multiprocessing start methods.
    if "--num-workers" not in args:
        args = [*args, "--num-workers", 0]
    result = CliRunner().invoke(app, list(map(str, args)))
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"


def test_evaluation_order_and_generation_schema(toy_local, tmp_path):
    model, output = tmp_path / "model", tmp_path / "output"
    shared = [
        "--dataset",
        "toy",
        "--path",
        toy_local,
        "--no-check-remote",
        "--batch",
        2,
    ]
    invoke(
        [
            "train",
            *shared,
            "--output_dir",
            model,
            "--epoch",
            1,
            "--iterations",
            1,
            "--val-iterations",
            1,
        ]
    )
    invoke(["evaluate", *shared, "--input_dir", model, "--output_dir", output])
    with np.load(next(output.glob("*.npz"))) as saved:
        assert saved["file_id"].tolist() == [0] * 4 + [1] * 4
        assert saved["row_id"].tolist() == list(range(4)) * 2
        np.testing.assert_allclose(saved["prediction"].sum(-1), 1, atol=1e-6)
    invoke(
        [
            "train",
            *shared,
            "--output_dir",
            model,
            "--save-tag",
            "gen",
            "--mode",
            "generator",
            "--num-classes",
            1,
            "--epoch",
            1,
            "--iterations",
            1,
            "--val-iterations",
            1,
        ]
    )
    invoke(
        [
            "evaluate",
            *shared,
            "--input_dir",
            model,
            "--save-tag",
            "gen",
            "--output_dir",
            output,
            "--mode",
            "generator",
            "--nsteps",
            2,
        ]
    )
    files = sorted(output.glob("generated_gen*.h5"))
    assert len(files) == 1  # One streamed output per rank.
    with h5py.File(files[0]) as new:
        assert new["data"].shape == (8, 11, 4)
        assert new["source_row"][:].tolist() == list(range(4)) * 2
        for i, (filename, row) in enumerate(
            zip(new["source_file"].asstr()[:], new["source_row"][:], strict=True)
        ):
            with h5py.File(filename) as old:
                np.testing.assert_array_equal(new["pid"][i], old["pid"][row])
                np.testing.assert_array_equal(new["global"][i], old["global"][row])
                count = (0, 1, 3, 5)[row]
                assert np.count_nonzero(new["data"][i, count:]) == 0
                assert np.isfinite(new["data"][i]).all()
                np.testing.assert_array_equal(
                    new["data_pid"][i, : old["data"].shape[1]], old["data_pid"][row]
                )


def test_high_level(toy_local, tmp_path):
    model, output = tmp_path / "hlmodel", tmp_path / "hlout"
    invoke(
        [
            "train-hl",
            "--dataset",
            "toy",
            "--path",
            toy_local,
            "--output_dir",
            model,
            "--num-feat",
            2,
            "--num-cond",
            1,
            "--epoch",
            1,
            "--iterations",
            1,
            "--val-iterations",
            1,
            "--batch",
            2,
            "--no-check-remote",
        ]
    )
    np.savez(toy_local / "conditions.npz", cond=np.arange(5, dtype=np.float32))
    invoke(
        [
            "evaluate-hl",
            "--dataset",
            "toy",
            "--path",
            toy_local,
            "--condition-file",
            "conditions.npz",
            "--input_dir",
            model,
            "--output_dir",
            output,
            "--batch",
            2,
            "--nsteps",
            2,
        ]
    )
    with h5py.File(next(output.glob("*.h5"))) as saved:
        assert saved["global"].shape == (5, 3)
        np.testing.assert_array_equal(saved["source_row"][:], np.arange(5))


def test_auxiliary_feature_columns(toy_local, tmp_path):
    for file in toy_local.glob("toy/*/*.h5"):
        with h5py.File(file, "a") as f:
            x = f["data"][:]
            pid = np.where(x[..., 2:3] != 0, 2, 0)
            data = np.concatenate((x, pid, np.repeat(pid, 2, -1) * 0.3), -1)
            data[:, [0, -1]] = data[:, [-1, 0]]
            del f["data"]
            f["data"] = data
    model, output = tmp_path / "auxmodel", tmp_path / "auxout"
    shared = [
        "--dataset",
        "toy",
        "--path",
        toy_local,
        "--no-check-remote",
        "--batch",
        2,
        "--mode",
        "generator",
        "--num-classes",
        1,
        "--use-pid",
        "--pid-idx",
        4,
        "--use-add",
        "--num-add",
        2,
    ]
    invoke(
        [
            "train",
            *shared,
            "--output_dir",
            model,
            "--epoch",
            1,
            "--iterations",
            1,
            "--val-iterations",
            1,
        ]
    )
    invoke(
        [
            "evaluate",
            *shared,
            "--input_dir",
            model,
            "--output_dir",
            output,
            "--nsteps",
            2,
        ]
    )
    files = list(output.glob("generated*.h5"))
    assert len(files) == 1
    with h5py.File(files[0]) as new:
        assert new["data"].shape == (8, 11, 7)
        for i, (filename, row) in enumerate(
            zip(new["source_file"].asstr()[:], new["source_row"][:], strict=True)
        ):
            with h5py.File(filename) as old:
                y = old["data"][row]
                x = new["data"][i, : len(y)]
                np.testing.assert_array_equal(x[..., 4:], y[..., 4:])
                assert np.count_nonzero(x[y[..., 2] == 0]) == 0
                assert np.count_nonzero(new["data"][i, len(y) :]) == 0


@pytest.mark.parametrize("mode", ["encoder", "pretrain_decoder"])
def test_checkpoint_token_count(mode, tmp_path):
    from omnilearned.evaluate import load_for_evaluation
    from omnilearned.runtime import make_model

    config = dict(
        size="lite",
        num_feat=4,
        num_tokens=3,
        num_part=5,
        mode=mode,
        interaction=False,
        local_interaction=False,
    )
    original = make_model(config).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": config, "model": original.state_dict()}, path)
    evaluation = dict(config, num_tokens=4, mode="encoder", checkpoint_path=str(path))
    restored = load_for_evaluation(evaluation, "cpu")
    assert restored.num_tokens == 3
    x, y = torch.randn(2, 5, 4), torch.zeros(2)
    with torch.no_grad():
        torch.testing.assert_close(restored(x, y)["latent"], original(x, y)["latent"])
