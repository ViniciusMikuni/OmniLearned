import os
import socket
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.distributed
@pytest.mark.skipif(
    os.environ.get("OMNILEARNED_SKIP_DISTRIBUTED") == "1",
    reason="Runtime prohibits Gloo operations; enabled by default in CI",
)
def test_two_rank_training_evaluation_and_failure(toy_local, tmp_path):
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "GLOO_SOCKET_IFNAME": "lo",
    }
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    prefix = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master-addr=127.0.0.1",
        "--master-port",
        port,
        "--nnodes=1",
        "--nproc-per-node=2",
        "-m",
        "omnilearned.cli",
    ]

    def run(args, okay=True):
        r = subprocess.run(
            prefix + args, env=env, text=True, capture_output=True, timeout=90
        )
        assert (r.returncode == 0) == okay, r.stdout + r.stderr
        return r

    shared = [
        "--dataset",
        "toy",
        "--path",
        str(toy_local),
        "--no-check-remote",
        "--batch",
        "2",
        "--num-workers",
        "0",
    ]
    model, output = tmp_path / "ddp", tmp_path / "out"
    run(
        [
            "train",
            *shared,
            "--output_dir",
            str(model),
            "--size",
            "lite",
            "--epoch",
            "1",
            "--iterations",
            "1",
            "--val-iterations",
            "1",
        ]
    )
    run(["evaluate", *shared, "--input_dir", str(model), "--output_dir", str(output)])
    rows = []
    for file in sorted(output.glob("*.npz")):
        with np.load(file) as data:
            rows.extend(
                zip(data["file_id"].tolist(), data["row_id"].tolist(), strict=False)
            )
    assert rows == [(f, r) for f in range(2) for r in range(4)]
    r = run(
        [
            "dataloader",
            "--dataset",
            "absent",
            "--folder",
            str(tmp_path / "absent"),
            "--base-url",
            "http://127.0.0.1:1/",
        ],
        okay=False,
    )
    assert "Could not check remote dataset" in r.stdout + r.stderr
