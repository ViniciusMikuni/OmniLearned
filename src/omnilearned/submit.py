"""NERSC Perlmutter batch scripts, one Slurm process per GPU."""

import re
import shlex
import subprocess
from pathlib import Path


def render(
    arguments,
    *,
    account="m3246",
    nodes=16,
    gpus_per_node=4,
    cpus_per_task=32,
    qos="regular",
    time="04:00:00",
    job_name="omnilearned",
    output="job_log_%j.out",
    constraint="gpu",
    module="pytorch",
    workdir=None,
    master_port=29500,
):
    if nodes < 1 or not 1 <= gpus_per_node <= 4 or cpus_per_task < 1:
        raise ValueError("Require nodes >= 1, 1..4 GPUs/node, CPUs/task >= 1")
    if not 1024 <= master_port <= 65535:
        raise ValueError("master_port must be 1024..65535")
    if not re.fullmatch(r"(?:\d+-)?\d+:\d{2}:\d{2}", time):
        raise ValueError("time must be HH:MM:SS or D-HH:MM:SS")
    for value in (account, qos, job_name, constraint):
        if not re.fullmatch(r"[A-Za-z0-9_.:+-]+", value):
            raise ValueError(f"Invalid Slurm value: {value!r}")
    if any(c in output for c in "\n\r\x00"):
        raise ValueError("output must be a single-line path")
    workdir = str(Path(workdir or Path.cwd()).resolve())
    worker = 'export RANK=$SLURM_PROCID WORLD_RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$SLURM_NTASKS; exec "$@"'
    lines = [
        "#!/bin/bash",
        f"#SBATCH --constraint={constraint}",
        f"#SBATCH --account={account}",
        f"#SBATCH --qos={qos}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks-per-node={gpus_per_node}",
        f"#SBATCH --gpus-per-node={gpus_per_node}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={shlex.quote(output)}",
        f"#SBATCH --time={time}",
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(workdir)}",
    ]
    if module:
        lines.append(f"module load {shlex.quote(module)}")
    lines += [
        "export HDF5_USE_FILE_LOCKING=FALSE",
        "export NCCL_NET_GDR_LEVEL=PHB",
        'export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)',
        f"export MASTER_PORT={master_port}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
        "srun -l -u --cpu-bind=cores --gpu-bind=none bash -c "
        + shlex.quote(worker)
        + " bash "
        + shlex.join(["omnilearned", *arguments]),
        "",
    ]
    return "\n".join(lines)


def submit(script, contents, dry_run=False):
    path = Path(script).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an existing script by accident.
    with path.open("x") as stream:
        stream.write(contents)
    print(f"Slurm script: {path}", flush=True)
    if dry_run:
        print(contents)
        return
    result = subprocess.run(
        ["sbatch", "--parsable", str(path)], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    print(f"Submitted job {result.stdout.strip()}", flush=True)


def validated_args(arguments):
    import typer

    from omnilearned.cli import app
    from omnilearned.config import resolve

    arguments = list(arguments)
    if arguments[0].startswith("-"):
        arguments.insert(0, "train")
    name = arguments[0]
    if name not in ("train", "train-hl", "evaluate", "evaluate-hl", "dataloader"):
        raise ValueError(f"Unsupported job command: {name}")
    command = typer.main.get_command(app).commands[name]
    with command.make_context(name, arguments[1:]) as ctx:
        values = resolve(ctx.params)
    result = [name]
    for parameter in command.params:
        if parameter.name not in values:
            continue
        value = values[parameter.name]
        if parameter.is_flag:
            result.append((parameter.opts if value else parameter.secondary_opts)[0])
        else:
            result.extend([parameter.opts[0], str(value)])
    return result
