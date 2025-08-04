import torch
from omnilearned.network import PET2
from omnilearned.dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omnilearned.utils import (
    is_master_node,
    ddp_setup,
    get_checkpoint_name,
    print_metrics,
    gather_tensors,
)
from omnilearned.diffusion import generate
import os
import time
import numpy as np
import h5py
from tqdm.auto import tqdm


def eval_model(
    model,
    test_loader,
    dataset,
    mode,
    use_event_loss,
    device="cpu",
    save_tag="pretrain",
    rank=0,
):
    start = time.time()
    prediction, cond, labels = test_step(model, test_loader, mode, device)

    if dist.is_initialized() and mode == "classifier" and prediction.shape[-1] == 2:
        prediction, cond, labels = [
            gather_tensors(t) for t in (prediction, cond, labels)
        ]
        if is_master_node():
            print_metrics(prediction.softmax(-1).numpy(), labels.numpy())
            print("Time taken for evaluation is {} sec".format(time.time() - start))

    if mode == "classifier":
        if use_event_loss:
            np.savez(
                f"/pscratch/sd/v/vmikuni/Omnilearned/outputs_{save_tag}_{dataset}_{rank}.npz",
                prediction=prediction[:, :200].softmax(-1).cpu().numpy(),
                event_prediction=prediction[:, 200:].softmax(-1).cpu().numpy(),
                pid=labels.cpu().numpy(),
                cond=cond.cpu().numpy(),
            )
        else:
            np.savez(
                f"/pscratch/sd/v/vmikuni/Omnilearned/outputs_{save_tag}_{dataset}_{rank}.npz",
                prediction=prediction.softmax(-1).cpu().numpy(),
                pid=labels.cpu().numpy(),
                cond=cond.cpu().numpy(),
            )
    else:
        with h5py.File(
            f"/pscratch/sd/v/vmikuni/Omnilearned/generated_{save_tag}_{dataset}_{rank}.h5",
            "w",
        ) as fh5:
            fh5.create_dataset("data", data=prediction.cpu().numpy())
            fh5.create_dataset("global", data=cond.cpu().numpy())
            fh5.create_dataset("pid", data=labels.cpu().numpy() + 1)


def pad_array(tensor_list, M: int = 150) -> torch.Tensor:
    """
    Given a list of torch tensors, each of shape (B, N_i, F),
    pads or truncates each along dimension N to length M,
    and returns a single tensor of shape (I, M, F), where
      H = sum over list of B,
      M = target length,
      F = feature dimension.
    """
    # Determine total number of samples and feature dim
    H = sum(t.shape[0] for t in tensor_list)
    _, _, F = tensor_list[0].shape

    # Use the dtype/device of the first tensor
    device = tensor_list[0].device
    dtype = tensor_list[0].dtype

    # Allocate output buffer
    out = torch.zeros((H, M, F), dtype=dtype, device=device)

    idx = 0
    for t in tensor_list:
        B, N, F_ = t.shape
        assert F_ == F, "All tensors must have the same feature dimension F"

        if N < M:
            # create a (B, M, F) zero tensor and copy `t` into its first N slots
            padded = torch.zeros((B, M, F), dtype=dtype, device=device)
            padded[:, :N, :] = t
        else:
            # truncate to the first M points
            padded = t[:, :M, :]

        out[idx : idx + B] = padded
        idx += B

    return out


def test_step(
    model,
    dataloader,
    mode,
    device,
):
    model.eval()

    preds = []
    labels = []
    conds = []

    for ib, batch in enumerate(
        tqdm(dataloader, desc="Iterating", total=len(dataloader))
        if is_master_node()
        else dataloader
    ):
        # if ib > 30000: break
        # for ib, batch in enumerate(dataloader):
        # if ib > 40000: break
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }

        with torch.no_grad():
            if mode == "classifier":
                outputs = model(X, y, **model_kwargs)
                preds.append(outputs["y_pred"])
            elif mode == "generator":
                assert "cond" in model_kwargs, (
                    "ERROR, conditioning variables not passed to model"
                )
                preds.append(generate(model, y, X.shape, **model_kwargs))
        labels.append(y)
        conds.append(batch["cond"])
        if mode == "generator":
            if batch["pid"] is not None:
                preds[-1] = torch.cat(
                    [preds[-1], model_kwargs["pid"].unsqueeze(-1).float()], -1
                )
            if batch["add_info"] is not None:
                preds[-1] = torch.cat([preds[-1], model_kwargs["add_info"]], -1)

    if mode == "generator":
        preds = pad_array(preds)
    else:
        preds = torch.cat(preds).to(device)
    return (
        preds,
        torch.cat(conds).to(device),
        torch.cat(labels).to(device),
    )


def restore_checkpoint(
    model,
    mode,
    checkpoint_dir,
    checkpoint_name,
    device,
    is_main_node=False,
):
    device = "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name),
        map_location=device,
    )

    base_model = model.module if hasattr(model, "module") else model
    base_model.to(device)

    if mode == "generator" and "ema_generator" in checkpoint:
        body_name = "ema_body"
        generator_name = "ema_generator"
        if is_main_node:
            print("Will load EMA models for evaluation")
    else:
        body_name = "body"
        generator_name = "generator_head"

    if base_model.body is not None and body_name in checkpoint:
        body_state = checkpoint[body_name]
        model_state = base_model.body.state_dict()
        filtered_state = {}
        for k, v in body_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                if is_main_node:
                    print(
                        f"Skipping {k}: shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape if k in model_state else 'missing'})"
                    )

        base_model.body.load_state_dict(filtered_state, strict=False)

    if base_model.classifier is not None and "classifier_head" in checkpoint:
        classifier_state = checkpoint["classifier_head"]
        model_state = base_model.classifier.state_dict()
        filtered_state = {}
        for k, v in classifier_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                if is_main_node:
                    print(
                        f"Skipping {k}: shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape if k in model_state else 'missing'})"
                    )

        base_model.classifier.load_state_dict(filtered_state, strict=False)

    if base_model.generator is not None:
        generator_state = checkpoint[generator_name]
        model_state = base_model.generator.state_dict()
        filtered_state = {}
        for k, v in generator_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                if is_main_node:
                    print(
                        f"Skipping {k}: shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape if k in model_state else 'missing'})"
                    )

        base_model.generator.load_state_dict(filtered_state, strict=False)


def run(
    indir: str = "",
    save_tag: str = "",
    dataset: str = "top",
    path: str = "/pscratch/sd/v/vmikuni/datasets",
    num_feat: int = 4,
    conditional: bool = False,
    num_cond: int = 3,
    use_pid: bool = False,
    pid_idx: int = -1,
    use_add: bool = False,
    num_add: int = 4,
    zero_add: bool = False,
    use_event_loss: bool = False,
    num_classes: int = 2,
    mode: str = "classifier",
    batch: int = 64,
    num_transf: int = 6,
    num_transf_head: int = 2,
    num_tokens: int = 4,
    num_head: int = 8,
    K: int = 15,
    base_dim: int = 64,
    mlp_ratio: int = 2,
    attn_drop: float = 0.1,
    mlp_drop: float = 0.1,
    feature_drop: float = 0.0,
    num_workers: int = 16,
    clip_inputs: bool = False,
):
    local_rank, rank, size = ddp_setup()
    # set up model
    model = PET2(
        input_dim=num_feat,
        hidden_size=base_dim,
        num_transformers=num_transf,
        num_transformers_head=num_transf_head,
        num_heads=num_head,
        mlp_ratio=mlp_ratio,
        mlp_drop=mlp_drop,
        attn_drop=attn_drop,
        feature_drop=feature_drop,
        num_tokens=num_tokens,
        K=K,
        conditional=conditional,
        cond_dim=num_cond,
        pid=use_pid,
        add_info=use_add,
        add_dim=num_add,
        use_time=mode == "generator",
        mode=mode,
        num_classes=num_classes,
    )

    if rank == 0:
        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("**** Setup ****")
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        print(f"Evaluating on device: {d}, with {size} GPUs")
        print("************")

    # load in train data
    test_loader = load_data(
        dataset,
        dataset_type="test",
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
        zero_add=zero_add,
        clip_inputs=clip_inputs,
    )
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(test_loader)}")
        print("************")
    if os.path.isfile(os.path.join(indir, get_checkpoint_name(save_tag))):
        if is_master_node():
            print(
                f"Loading checkpoint from {os.path.join(indir, get_checkpoint_name(save_tag))}"
            )

        restore_checkpoint(
            model,
            mode,
            indir,
            get_checkpoint_name(save_tag),
            local_rank,
            rank == 0,
        )

    else:
        raise ValueError(
            f"Error loading checkpoint: {os.path.join(indir, get_checkpoint_name(save_tag))}"
        )

    # Transfer model to GPU if available
    kwarg = {}
    if torch.cuda.is_available():
        device = local_rank
        model.to(local_rank)
        kwarg["device_ids"] = [device]
    else:
        model.cpu()
        device = "cpu"

    model = DDP(
        model,
        **kwarg,
    )

    eval_model(
        model,
        test_loader,
        dataset,
        mode=mode,
        use_event_loss=use_event_loss,
        device=device,
        rank=rank,
        save_tag=save_tag,
    )
    dist.barrier()
    dist.destroy_process_group()
