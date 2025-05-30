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
)
import os
import time


def gather_tensors(x):
    """
    If running under DDP, all_gather x from every rank, concat, then return as numpy.
    Otherwise just .cpu().numpy().
    """
    if dist.is_initialized():
        ws = dist.get_world_size()
        # pre‐allocate one buffer per rank
        buf = [torch.zeros_like(x) for _ in range(ws)]
        dist.all_gather(buf, x)
        x = torch.cat(buf, dim=0)
    return x.cpu()


def eval_model(
    model,
    val_loader,
    dataset,
    use_event_loss,
    device="cpu",
):
    start = time.time()
    prediction, mass, pt, labels = test_step(model, val_loader, device)

    if dist.is_initialized():
        prediction, mass, pt, labels = [
            gather_tensors(t) for t in (prediction, mass, pt, labels)
        ]

    if is_master_node():
        print_metrics(prediction.softmax(-1).numpy(), labels.numpy())
    if is_master_node():
        print("Time taken for evaluation is {} sec".format(time.time() - start))
    #     if use_event_loss:
    #         np.savez(f"/pscratch/sd/v/vmikuni/outputs_{dataset}.npz",
    #                  # prediction= torch.nn.functional.log_softmax(prediction[:,:200],dim=-1).numpy(),
    #                  # event_prediction = torch.nn.functional.log_softmax(prediction[:,200:],dim=-1).numpy(),
    #                  prediction= prediction[:,:200].softmax(-1).numpy(),
    #                  event_prediction = prediction[:,200:].softmax(-1).numpy(),
    #                  mass=mass.numpy(), pt=pt.numpy())
    #     else:
    #         np.savez(f"/pscratch/sd/v/vmikuni/outputs_{dataset}.npz",
    #                  prediction= torch.nn.functional.log_softmax(prediction,dim=-1).numpy(),
    #                  mass=mass.numpy(), pt=pt.numpy())


def test_step(
    model,
    dataloader,
    device,
):
    model.eval()

    preds = []
    labels = []
    masses = []
    pts = []

    for ib, batch in enumerate(dataloader):
        if ib > 30000:
            break
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }
        with torch.no_grad():
            outputs = model(X, y, **model_kwargs)

        preds.append(outputs["y_pred"])
        labels.append(y)
        masses.append(torch.exp(batch["cond"][:, 1]))
        pts.append(torch.exp(batch["cond"][:, 0]))

    return (
        torch.cat(preds).to(device),
        torch.cat(masses).to(device),
        torch.cat(pts).to(device),
        torch.cat(labels).to(device),
    )


def restore_checkpoint(
    model,
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
    base_model.body.load_state_dict(checkpoint["body"], strict=False)

    if base_model.classifier is not None and "classifier_head" in checkpoint:
        base_model.classifier.load_state_dict(checkpoint["classifier_head"])

    if base_model.generator is not None:
        base_model.generator.load_state_dict(checkpoint["generator_head"])


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
        use_time=False,
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
    val_loader = load_data(
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
    )
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(val_loader)}")
        print("************")
    if os.path.isfile(os.path.join(indir, get_checkpoint_name(save_tag))):
        if is_master_node():
            print(
                f"Loading checkpoint from {os.path.join(indir, get_checkpoint_name(save_tag))}"
            )

        restore_checkpoint(
            model,
            indir,
            get_checkpoint_name(save_tag),
            local_rank,
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

    eval_model(model, val_loader, dataset, use_event_loss=use_event_loss, device=device)
    dist.destroy_process_group()
