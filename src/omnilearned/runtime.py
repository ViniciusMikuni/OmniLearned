"""Shared runtime construction and provenance."""

import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch

from omnilearned.data_spec import LABEL_OFFSETS
from omnilearned.dataloader import BASE_URL, load_data
from omnilearned.network import MLPGEN, PET2
from omnilearned.utils import get_model_parameters


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(c, high_level=False):
    if high_level:
        return MLPGEN(
            c.get("num_feat", 3),
            256,
            mlp_drop=c.get("mlp_drop", 0.0),
            conditional=c.get("conditional", False),
            cond_dim=c.get("num_cond", 1),
        )
    mapping = {
        "num_feat": "input_dim",
        "interaction": "use_int",
        "local_interaction": "local_int",
        "interaction_type": "int_type",
        "num_cond": "cond_dim",
        "use_pid": "pid",
        "use_add": "add_info",
        "num_add": "add_dim",
        "num_part": "num_particles",
    }
    keys = [
        "num_feat",
        "interaction",
        "local_interaction",
        "interaction_type",
        "conditional",
        "num_cond",
        "use_pid",
        "pid_dim",
        "use_add",
        "num_add",
        "mode",
        "num_classes",
        "num_gen_classes",
        "mlp_drop",
        "attn_drop",
        "feature_drop",
        "num_coord",
        "K",
        "num_part",
        "num_tokens",
        "dataset",
        "qk_norm",
        "attention_backend",
        "activation_checkpointing",
    ]
    options = {mapping.get(k, k): c[k] for k in keys if k in c}
    return PET2(**(get_model_parameters(c.get("size", "small")) | options))


def make_loader(c, split, rank, size, high_level=False):
    keys = [
        "use_pid",
        "pid_idx",
        "use_add",
        "num_add",
        "zero_add",
        "num_workers",
        "clip_inputs",
        "mode",
        "seed",
        "persistent_workers",
        "prefetch_factor",
        "max_open_files",
        "check_remote",
        "bucket_size",
    ]
    options = {k: c[k] for k in keys if k in c}
    return load_data(
        c["dataset"],
        c["path"],
        batch=c["batch"],
        dataset_type=split,
        rank=rank,
        size=size,
        use_cond=high_level
        or c.get("conditional", False)
        or c.get("mode") == "generator",
        global_only=high_level,
        shuffle=split in ("train", "val"),
        nevts=c.get("nevts", -1) if split == "train" else -1,
        base_url=c.get("base_url", BASE_URL),
        **options,
    )


def to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def metadata(c, model, loader, size, device):
    paths = sorted(
        {str(Path(p).parent / "file_manifest.json") for p in loader.dataset.file_paths}
    )
    manifests = {p: json.loads(Path(p).read_text()) for p in paths}
    architecture = (
        {"base_dim": 256, "num_layers": 3}
        if isinstance(model, MLPGEN)
        else get_model_parameters(c.get("size", "small"))
    )
    if not isinstance(model, MLPGEN):
        architecture["num_tokens"] = model.num_tokens
    return {
        **c,
        "architecture": architecture,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "runtime": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "world_size": size,
        },
        "data": {
            "manifest_sha256": hashlib.sha256(
                json.dumps(manifests, sort_keys=True).encode()
            ).hexdigest(),
            "files": manifests,
            "label_offsets": LABEL_OFFSETS,
            "validity_feature": 2,
            "multiplicity_scale": 100,
            "training_tail": "drop at most world_size-1",
            "evaluation_order": "file then row, contiguous rank slices",
        },
    }
