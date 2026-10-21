"""Sequential high-level generation."""

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist

from omnilearned.diffusion import generate_hl
from omnilearned.evaluate import append, load_for_evaluation
from omnilearned.runtime import seed_everything
from omnilearned.utils import ddp_setup


def run(**c):
    local, rank, size = ddp_setup()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    seed_everything(c["seed"] + rank)
    model = load_for_evaluation(c, device, high_level=True)
    with np.load(Path(c["path"]) / c["condition_file"]) as data:
        conditions = data["cond"]
    if conditions.ndim == 1:
        conditions = conditions[:, None]
    if conditions.shape[1] != c["num_cond"]:
        raise ValueError("Condition file must have exactly num_cond columns")
    out = Path(c["outdir"] or ".")
    out.mkdir(parents=True, exist_ok=True)
    start, stop = len(conditions) * rank // size, len(conditions) * (rank + 1) // size
    with h5py.File(
        out / f"generated_{c['save_tag']}_{c['dataset']}_{rank}.h5", "w"
    ) as target:
        for i in range(start, stop, c["batch"]):
            cond = torch.as_tensor(
                conditions[i : min(stop, i + c["batch"])],
                device=device,
                dtype=torch.float32,
            )
            prediction = generate_hl(
                model, (len(cond), c["num_feat"]), cond, c["nsteps"], device
            )
            prediction[:, 0].clamp_(min=np.log(450))
            prediction[:, -1] = (100 * prediction[:, -1]).round().clamp(
                1, c["max_part"]
            ) / 100
            append(
                target,
                "global",
                torch.cat((prediction[:, :1], cond, prediction[:, 1:]), -1).cpu(),
            )
            append(target, "data", np.zeros((len(cond), c["max_part"], 4), np.float32))
            append(target, "pid", np.zeros(len(cond), np.int64))
            append(target, "source_row", np.arange(i, i + len(cond)))
    print(f"Evaluation complete: {out}")
    if dist.is_initialized():
        dist.destroy_process_group()
