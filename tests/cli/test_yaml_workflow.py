"""Validate the config matrix, pairing and real shell argument forwarding."""

import os
import subprocess
from pathlib import Path

import typer

from omnilearned.cli import app
from omnilearned.config import resolve
from omnilearned.submit import validated_args

ROOT = Path(__file__).resolve().parents[2]


def test_config_matrix():
    commands = typer.main.get_command(app).commands
    expected = set()
    for size in ("small", "medium", "large"):
        expected.add(f"pretrain_{size}")
        for dataset in (
            "top",
            "qg",
            "jetnet30",
            "jetnet150",
            "atlas_flav",
            "microboone",
            "minerva",
            "camels",
            "quijote",
        ):
            modes = (
                ("regression", "segmentation")
                if dataset in ("camels", "quijote")
                else (None,)
            )
            for mode in modes:
                for kind in ("scratch", "finetune"):
                    expected.add("_".join(x for x in (dataset, mode, size, kind) if x))
    assert {
        p.stem
        for p in (ROOT / "configs").glob("*.yaml")
        if not p.stem.startswith("evaluate_")
    } == expected
    assert {
        p.stem.removeprefix("evaluate_")
        for p in (ROOT / "configs").glob("evaluate_*.yaml")
    } == expected
    for stem in expected:
        resolved = {}
        for command, filename in [("train", stem), ("evaluate", "evaluate_" + stem)]:
            path = ROOT / "configs" / f"{filename}.yaml"
            with commands[command].make_context(
                command, ["--config", str(path)]
            ) as ctx:
                resolved[command] = resolve(ctx.params)
        train, evaluation = resolved["train"], resolved["evaluate"]
        assert train["save_tag"] == evaluation["save_tag"] == stem
        assert train["outdir"] == evaluation["indir"]
        assert train["dataset"] == evaluation["dataset"]
        assert train["size"] == evaluation["size"]
        assert not train["resuming"]
        if stem.startswith("pretrain_"):
            assert train["dataset"] == train["mode"] == "pretrain"
        else:
            assert train["fine_tune"] == stem.endswith("_finetune")
            if train["fine_tune"]:
                assert train["pretrain_tag"] == "pretrain_" + train["size"]
        if train["dataset"] in ("jetnet30", "jetnet150"):
            assert train["mode"] == evaluation["mode"] == "generator"
        if train["dataset"] in ("top", "qg", "microboone", "minerva"):
            assert train["mode"] == evaluation["mode"] == "classifier"
        if train["dataset"] == "top":
            assert train["num_classes"] == evaluation["num_classes"] == 2
        if train["dataset"] == "atlas_flav":
            assert train["mode"] == "ftag" and evaluation["mode"] == "classifier"
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
module() { test "$*" = "load pytorch/2.11"; }
scontrol() { echo test-node; }
srun() { while [ "$1" != bash ]; do shift; done; "$@"; }
export -f module scontrol srun
bash "$1" "$2"
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
                ],
                cwd=ROOT / "shell",
                env=env,
                check=True,
            )
            assert capture.read_bytes().decode().rstrip("\0").split("\0") == [
                command,
                "--config",
                str(expected),
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
