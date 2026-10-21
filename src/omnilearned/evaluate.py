"""Sequential streamed evaluation and schema-preserving generation."""

import json
import zipfile
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from omnilearned.checkpoint import load_weights
from omnilearned.data_spec import FeatureLayout, multiplicity_counts
from omnilearned.diffusion import estimate_max_logp, generate
from omnilearned.packing import dense_batch
from omnilearned.runtime import make_loader, make_model, seed_everything, to_device
from omnilearned.utils import ddp_setup, get_checkpoint_name


class BatchProgress(tqdm):
    """Use flushed lines for Slurm pipes, an updating bar for terminals."""

    def display(self, msg=None, pos=None):
        if self.fp.isatty():
            return super().display(msg=msg, pos=pos)
        self.fp.write((str(self) if msg is None else msg) + "\n")
        self.fp.flush()
        return True


def append(group, name, value):
    value = np.asarray(value)
    if name not in group:
        group.create_dataset(
            name,
            shape=(0, *value.shape[1:]),
            maxshape=(None, *value.shape[1:]),
            dtype=value.dtype,
            chunks=True,
        )
    data = group[name]
    start = len(data)
    data.resize(start + len(value), axis=0)
    data[start:] = value


def export_npz(source, destination):
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for name, array in source.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as target:
                np.lib.format.write_array_header_2_0(
                    target,
                    {
                        "descr": np.lib.format.dtype_to_descr(array.dtype),
                        "fortran_order": False,
                        "shape": array.shape,
                    },
                )
                for start in range(0, len(array), 1024):
                    target.write(
                        np.ascontiguousarray(array[start : start + 1024]).tobytes()
                    )


def copied_fields(source, rows):
    return {
        k: v[rows]
        for k, v in source.items()
        if isinstance(v, h5py.Dataset) and v.ndim and len(v) == len(source["data"])
    }


def load_for_evaluation(c, device, high_level=False):
    file = Path(
        c.get("checkpoint_path")
        or Path(c["indir"] or ".") / get_checkpoint_name(c["save_tag"])
    )
    saved = torch.load(file, map_location="cpu", weights_only=False)
    architecture = (
        "size num_feat num_part num_tokens interaction local_interaction num_coord K "
        "interaction_type conditional num_cond use_pid pid_idx pid_dim use_add "
        "num_add num_classes num_gen_classes qk_norm"
    ).split()
    model_config = dict(c)
    input_layout = {"use_pid", "pid_idx", "use_add", "num_add"}

    for key in architecture:
        if key in saved.get("config", {}):
            model_config[key] = saved["config"][key]
            if key not in input_layout:
                c[key] = saved["config"][key]

    mode = saved.get("config", {}).get("mode")
    if mode in ("pretrain", "pretrain_regression", "pretrain_decoder", "ftag"):
        model_config["mode"] = mode
    model = make_model(model_config, high_level).to(device)
    use_ema = c.get("use_ema", True) and (c.get("mode") == "generator" or high_level)
    if high_level and "ema_model" in saved and use_ema:
        saved["ema"] = saved["ema_model"]
    load_weights(model, saved, ema=use_ema)
    return model.eval()


def model_inputs(batch):
    return {
        k: batch[k] for k in ("cond", "pid", "add_info") if batch.get(k) is not None
    }


def generated_fields(source, row, prediction, i, counts, width, c):
    mode = c["mode"]
    fields = copied_fields(source, slice(row, row + 1))
    if mode == "generator":
        data = np.zeros_like(fields["data"])
        layout = FeatureLayout(
            data.shape[-1],
            c["pid_idx"] % data.shape[-1] if c.get("use_pid") else None,
            c["num_add"] if c.get("use_add") else 0,
        )
        count = int(counts[i])
        positions = np.flatnonzero(fields["data"][0, :, 2] != 0)
        positions = positions if len(positions) == count else np.arange(count)
        data[0][np.ix_(positions, layout.physical)] = (
            prediction[i, :count].cpu().numpy()
        )
        for column in (
            [layout.pid_idx] if layout.pid_idx is not None else []
        ) + layout.additional:
            data[0, positions, column] = fields["data"][0, positions, column]
        fields["data"] = data
    else:
        fields["data"] = prediction[i : i + 1].cpu().numpy()

    # Preserve the latent token count; only pad source particle arrays.
    for key in ("data", "data_pid") if mode == "generator" else ("data_pid",):
        if key in fields and fields[key].ndim >= 2:
            value = fields[key]
            padded_value = np.zeros(
                (len(value), width, *value.shape[2:]), dtype=value.dtype
            )
            padded_value[:, : value.shape[1]] = value
            fields[key] = padded_value
    return fields


def run(**c):
    local, rank, size = ddp_setup()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    seed_everything(c.get("seed", 42) + rank)
    mode, sbi = c["mode"], c.get("sbi", False)
    if mode not in ("classifier", "regression", "segmentation", "generator", "encoder"):
        raise ValueError(
            "Use classifier evaluation for ftag; supported modes are "
            "classifier, regression, segmentation, generator and encoder"
        )
    if sbi and mode != "generator":
        raise ValueError("SBI requires generator mode")
    model = load_for_evaluation(c, device)
    loader = make_loader(c, "test", rank, size)
    outdir = Path(c["outdir"] or ".")
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"outputs_{c['save_tag']}_{c['dataset']}_{rank}"
    sources, width = loader.dataset.file_paths, 0
    tabular = mode in ("classifier", "regression", "segmentation") or sbi
    if mode in ("segmentation", "generator", "encoder"):
        for path in sources:
            with h5py.File(path) as source:
                width = max(width, source["data"].shape[1])
    try:
        with ExitStack() as stack:
            spool = target = None
            for raw in BatchProgress(
                loader,
                total=len(loader),
                desc="Evaluating",
                unit="batch",
                disable=rank != 0,
                dynamic_ncols=True,
                miniters=1,
                mininterval=1.0,
            ):
                if tabular and spool is None:
                    spool = stack.enter_context(h5py.File(outdir / f"{stem}.h5", "w"))
                batch = to_device(raw, device)
                dense = dense_batch(batch)
                kwargs = model_inputs(dense)
                results = {}
                with torch.no_grad():
                    if mode in ("classifier", "regression"):
                        body, mask = model.body(
                            dense["X"], mask=dense["mask"], return_mask=True, **kwargs
                        )
                        prediction = model.classifier(body, mask=mask)
                        if mode == "classifier":
                            if c.get("use_event_loss"):
                                results["event_prediction"] = (
                                    prediction[:, 200:].softmax(-1).cpu()
                                )
                                prediction = prediction[:, :200]
                            prediction = prediction.softmax(-1)
                    elif mode in ("segmentation", "encoder"):
                        output = model(
                            batch["X"],
                            batch["y"],
                            offsets=batch["offsets"],
                            **model_inputs(batch),
                        )
                        prediction = (
                            output["z_pred"]
                            if mode == "segmentation"
                            else output["latent"]
                        )
                    else:
                        maximum = int(batch["stored_shape"][:, 0].max())
                        mult = dense["cond"][:, -1]
                        if "camels" in c["dataset"] or "quijote" in c["dataset"]:
                            mult = maximum * torch.ones_like(mult) / 100.0
                        counts = multiplicity_counts(mult, maximum)
                        if (counts > batch["stored_shape"][:, 0]).any():
                            raise ValueError(
                                "Multiplicity exceeds source file particle capacity"
                            )
                        if c.get("sbi"):
                            ranges = json.loads(c["scan_ranges"])
                            result = estimate_max_logp(
                                model,
                                dense["X"],
                                batch["y"].long(),
                                **kwargs,
                                multiplicity=mult,
                                dataset=c["dataset"],
                                device=device,
                                scan_ranges=ranges,
                                num_param=len(ranges),
                                scan_points=c["scan_points"],
                                nsteps=c["scan_steps"],
                                refinements=c["scan_refinements"],
                                probes=c["scan_probes"],
                                candidate_batch=c["scan_candidate_batch"],
                                seed=c["seed"],
                            )
                            prediction = result["theta"]
                        else:
                            for key in ("pid", "add_info"):
                                if key in kwargs:
                                    value = kwargs[key]
                                    pad = value.new_zeros(
                                        (len(value), maximum, *value.shape[2:])
                                    )
                                    pad[:, : value.shape[1]] = value
                                    kwargs[key] = pad
                            prediction = generate(
                                model,
                                batch["y"].long(),
                                (len(batch["y"]), maximum, c["num_feat"]),
                                **kwargs,
                                nsteps=c["nsteps"],
                                dataset=c["dataset"],
                                device=device,
                            )

                rows = list(
                    zip(raw["file_id"].tolist(), raw["row_id"].tolist(), strict=True)
                )
                if mode == "segmentation":
                    values = np.zeros(
                        (len(rows), width, prediction.shape[-1]), np.float32
                    )
                    targets = np.zeros_like(values)
                    for i, (fid, row) in enumerate(rows):
                        start, stop = raw["offsets"][i : i + 2]
                        positions = raw["positions"][start:stop].numpy()
                        values[i, positions] = (
                            prediction[i, : len(positions)].cpu().numpy()
                        )
                        targets[i, positions] = loader.dataset._get_file(fid)[
                            "data_pid"
                        ][row][positions]
                    results.update(
                        prediction=values, pid=targets, lengths=raw["lengths"]
                    )
                elif tabular:
                    results.update(prediction=prediction.cpu(), pid=batch["y"].cpu())
                    conds = []
                    for fid, row in rows:
                        source = loader.dataset._get_file(fid)
                        if "global" in source:
                            conds.append(source["global"][row])
                    if conds:
                        results["cond"] = np.asarray(conds)
                if tabular:
                    results.update(file_id=raw["file_id"], row_id=raw["row_id"])
                    for name, value in results.items():
                        append(spool, name, value)
                    continue
                for i, (fid, row) in enumerate(rows):
                    source = loader.dataset._get_file(fid)
                    if target is None:
                        destination = (
                            outdir / f"generated_{c['save_tag']}_rank{rank}.h5"
                        )
                        target = stack.enter_context(h5py.File(destination, "w"))
                        target.attrs.update(source.attrs)
                    fields = generated_fields(
                        source,
                        row,
                        prediction,
                        i,
                        counts if mode == "generator" else None,
                        width,
                        c,
                    )
                    fields.update(
                        source_file=np.asarray(
                            [sources[fid]], dtype=h5py.string_dtype()
                        ),
                        source_row=np.asarray([row], np.int64),
                    )
                    for name, value in fields.items():
                        append(target, name, value)
                        if name in source:
                            target[name].attrs.update(source[name].attrs)
            if spool is not None and "prediction" in spool:
                if "cond" not in spool:
                    spool.create_dataset(
                        "cond", data=np.empty((len(spool["prediction"]), 0), np.float32)
                    )
                export_npz(spool, outdir / f"{stem}.npz")
        print(f"Evaluation complete: {outdir}")
    finally:
        loader.dataset.close()
        if dist.is_initialized():
            dist.destroy_process_group()
