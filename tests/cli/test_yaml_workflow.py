"""Validate the config matrix, pairing and real shell argument forwarding."""

import os
import subprocess
from pathlib import Path

import typer

from omnilearned.cli import app
from omnilearned.config import resolve
from omnilearned.submit import validated_args
from omnilearned.utils import get_model_parameters

ROOT = Path(__file__).resolve().parents[2]


def test_config_matrix():
    commands = typer.main.get_command(app).commands
    paths = sorted((ROOT / "configs").glob("*.yaml"))
    assert paths
    resolved = {}
    for path in paths:
        command = "evaluate" if path.stem.startswith("evaluate_") else "train"
        with commands[command].make_context(command, ["--config", str(path)]) as ctx:
            c = resolve(ctx.params)
        resolved[path.stem] = c
        allowed = {"classifier", "regression", "segmentation", "generator", "encoder"}
        if command == "train":
            allowed |= {"ftag", "pretrain", "pretrain_regression", "pretrain_decoder"}
        assert c["mode"] in allowed, path.name
        assert c["batch"] > 0 and c["num_feat"] > 0, path.name
        assert c["num_tokens"] > 0, path.name
        get_model_parameters(c["size"])

    # Check matching recipes, without requiring every dataset/size combination.
    # Cross-dataset evaluation and CLI-supplied save tags are supported.
    for name, evaluation in resolved.items():
        if not name.startswith("evaluate_"):
            continue
        stem = name.removeprefix("evaluate_")
        train = resolved.get(stem)
        if train is None or train["save_tag"] != evaluation["save_tag"]:
            continue
        assert train["outdir"] == evaluation["indir"], name
        for key in ("dataset", "size", "num_classes", "num_gen_classes"):
            assert train[key] == evaluation[key], (name, key)
        expected_mode = {
            "ftag": "classifier",
            "pretrain": "classifier",
            "pretrain_regression": "regression",
            "pretrain_decoder": "encoder",
        }.get(train["mode"], train["mode"])
        assert evaluation["mode"] == expected_mode, name
    # Optional submission accepts these same YAMLs and snapshots their values.
    args = validated_args(
        ["train", "--config", str(ROOT / "configs/top_small_scratch.yaml")]
    )
    assert "--config" not in args
    assert args[args.index("--save-tag") + 1] == "top_small_scratch"


def test_yaml_launchers(tmp_path):
    capture = tmp_path / "args"
    fake = tmp_path / "omnilearned"
    fake.write_text(
        '#!/bin/bash\nprintf "%s\\0" "$@" > "$CAPTURE"\ntest "$RANK" = 0 && test "$WORLD_SIZE" = 4\n'
    )
    fake.chmod(0o755)
    bootstrap = """
module() { test "$*" = "load pytorch"; }
scontrol() { echo test-node; }
srun() { while [ "$1" != bash ]; do shift; done; "$@"; }
export -f module scontrol srun
bash "$@"
"""
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        CAPTURE=str(capture),
        SLURM_JOB_NODELIST="test-node",
        SLURM_PROCID="0",
        SLURM_LOCALID="0",
        SLURM_NTASKS="4",
    )
    for script, command, name in [
        ("run.sh", "train", "top_small_scratch.yaml"),
        ("evaluate.sh", "evaluate", "evaluate_top_small_scratch.yaml"),
    ]:
        subprocess.run(["bash", "-n", str(ROOT / "shell" / script)], check=True)
        external = tmp_path / "custom config.yaml"
        external.write_text((ROOT / "configs" / name).read_text())
        for argument, expected in [
            (name, ROOT / "configs" / name),
            ("../configs/" + name, ROOT / "configs" / name),
            (str(external), external),
        ]:
            subprocess.run(
                [
                    "bash",
                    "-c",
                    bootstrap,
                    "bash",
                    str(ROOT / "shell" / script),
                    argument,
                    "--save-tag",
                    "trial_7",
                    "--seed",
                    "49",
                ],
                cwd=ROOT / "shell",
                env=env,
                check=True,
            )
            assert capture.read_bytes().decode().rstrip("\0").split("\0") == [
                command,
                "--config",
                str(expected),
                "--save-tag",
                "trial_7",
                "--seed",
                "49",
            ]
        failed = subprocess.run(
            ["bash", str(ROOT / "shell" / script), "absent.yaml"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        assert failed.returncode == 2 and "Config not found" in failed.stderr
        failed = subprocess.run(
            ["bash", str(ROOT / "shell" / script)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert failed.returncode == 2 and "Usage:" in failed.stderr
